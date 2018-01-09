
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Machine limits for Float32 and Float64 and (long double) if available...
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: __all__ = ['finfo', 'iinfo']
7: 
8: from .machar import MachAr
9: from . import numeric
10: from . import numerictypes as ntypes
11: from .numeric import array
12: 
13: def _frz(a):
14:     '''fix rank-0 --> rank-1'''
15:     if a.ndim == 0:
16:         a.shape = (1,)
17:     return a
18: 
19: _convert_to_float = {
20:     ntypes.csingle: ntypes.single,
21:     ntypes.complex_: ntypes.float_,
22:     ntypes.clongfloat: ntypes.longfloat
23:     }
24: 
25: class finfo(object):
26:     '''
27:     finfo(dtype)
28: 
29:     Machine limits for floating point types.
30: 
31:     Attributes
32:     ----------
33:     eps : float
34:         The smallest representable positive number such that
35:         ``1.0 + eps != 1.0``.  Type of `eps` is an appropriate floating
36:         point type.
37:     epsneg : floating point number of the appropriate type
38:         The smallest representable positive number such that
39:         ``1.0 - epsneg != 1.0``.
40:     iexp : int
41:         The number of bits in the exponent portion of the floating point
42:         representation.
43:     machar : MachAr
44:         The object which calculated these parameters and holds more
45:         detailed information.
46:     machep : int
47:         The exponent that yields `eps`.
48:     max : floating point number of the appropriate type
49:         The largest representable number.
50:     maxexp : int
51:         The smallest positive power of the base (2) that causes overflow.
52:     min : floating point number of the appropriate type
53:         The smallest representable number, typically ``-max``.
54:     minexp : int
55:         The most negative power of the base (2) consistent with there
56:         being no leading 0's in the mantissa.
57:     negep : int
58:         The exponent that yields `epsneg`.
59:     nexp : int
60:         The number of bits in the exponent including its sign and bias.
61:     nmant : int
62:         The number of bits in the mantissa.
63:     precision : int
64:         The approximate number of decimal digits to which this kind of
65:         float is precise.
66:     resolution : floating point number of the appropriate type
67:         The approximate decimal resolution of this type, i.e.,
68:         ``10**-precision``.
69:     tiny : float
70:         The smallest positive usable number.  Type of `tiny` is an
71:         appropriate floating point type.
72: 
73:     Parameters
74:     ----------
75:     dtype : float, dtype, or instance
76:         Kind of floating point data-type about which to get information.
77: 
78:     See Also
79:     --------
80:     MachAr : The implementation of the tests that produce this information.
81:     iinfo : The equivalent for integer data types.
82: 
83:     Notes
84:     -----
85:     For developers of NumPy: do not instantiate this at the module level.
86:     The initial calculation of these parameters is expensive and negatively
87:     impacts import times.  These objects are cached, so calling ``finfo()``
88:     repeatedly inside your functions is not a problem.
89: 
90:     '''
91: 
92:     _finfo_cache = {}
93: 
94:     def __new__(cls, dtype):
95:         try:
96:             dtype = numeric.dtype(dtype)
97:         except TypeError:
98:             # In case a float instance was given
99:             dtype = numeric.dtype(type(dtype))
100: 
101:         obj = cls._finfo_cache.get(dtype, None)
102:         if obj is not None:
103:             return obj
104:         dtypes = [dtype]
105:         newdtype = numeric.obj2sctype(dtype)
106:         if newdtype is not dtype:
107:             dtypes.append(newdtype)
108:             dtype = newdtype
109:         if not issubclass(dtype, numeric.inexact):
110:             raise ValueError("data type %r not inexact" % (dtype))
111:         obj = cls._finfo_cache.get(dtype, None)
112:         if obj is not None:
113:             return obj
114:         if not issubclass(dtype, numeric.floating):
115:             newdtype = _convert_to_float[dtype]
116:             if newdtype is not dtype:
117:                 dtypes.append(newdtype)
118:                 dtype = newdtype
119:         obj = cls._finfo_cache.get(dtype, None)
120:         if obj is not None:
121:             return obj
122:         obj = object.__new__(cls)._init(dtype)
123:         for dt in dtypes:
124:             cls._finfo_cache[dt] = obj
125:         return obj
126: 
127:     def _init(self, dtype):
128:         self.dtype = numeric.dtype(dtype)
129:         if dtype is ntypes.double:
130:             itype = ntypes.int64
131:             fmt = '%24.16e'
132:             precname = 'double'
133:         elif dtype is ntypes.single:
134:             itype = ntypes.int32
135:             fmt = '%15.7e'
136:             precname = 'single'
137:         elif dtype is ntypes.longdouble:
138:             itype = ntypes.longlong
139:             fmt = '%s'
140:             precname = 'long double'
141:         elif dtype is ntypes.half:
142:             itype = ntypes.int16
143:             fmt = '%12.5e'
144:             precname = 'half'
145:         else:
146:             raise ValueError(repr(dtype))
147: 
148:         machar = MachAr(lambda v:array([v], dtype),
149:                         lambda v:_frz(v.astype(itype))[0],
150:                         lambda v:array(_frz(v)[0], dtype),
151:                         lambda v: fmt % array(_frz(v)[0], dtype),
152:                         'numpy %s precision floating point number' % precname)
153: 
154:         for word in ['precision', 'iexp',
155:                      'maxexp', 'minexp', 'negep',
156:                      'machep']:
157:             setattr(self, word, getattr(machar, word))
158:         for word in ['tiny', 'resolution', 'epsneg']:
159:             setattr(self, word, getattr(machar, word).flat[0])
160:         self.max = machar.huge.flat[0]
161:         self.min = -self.max
162:         self.eps = machar.eps.flat[0]
163:         self.nexp = machar.iexp
164:         self.nmant = machar.it
165:         self.machar = machar
166:         self._str_tiny = machar._str_xmin.strip()
167:         self._str_max = machar._str_xmax.strip()
168:         self._str_epsneg = machar._str_epsneg.strip()
169:         self._str_eps = machar._str_eps.strip()
170:         self._str_resolution = machar._str_resolution.strip()
171:         return self
172: 
173:     def __str__(self):
174:         fmt = (
175:             'Machine parameters for %(dtype)s\n'
176:             '---------------------------------------------------------------\n'
177:             'precision=%(precision)3s   resolution= %(_str_resolution)s\n'
178:             'machep=%(machep)6s   eps=        %(_str_eps)s\n'
179:             'negep =%(negep)6s   epsneg=     %(_str_epsneg)s\n'
180:             'minexp=%(minexp)6s   tiny=       %(_str_tiny)s\n'
181:             'maxexp=%(maxexp)6s   max=        %(_str_max)s\n'
182:             'nexp  =%(nexp)6s   min=        -max\n'
183:             '---------------------------------------------------------------\n'
184:             )
185:         return fmt % self.__dict__
186: 
187:     def __repr__(self):
188:         c = self.__class__.__name__
189:         d = self.__dict__.copy()
190:         d['klass'] = c
191:         return (("%(klass)s(resolution=%(resolution)s, min=-%(_str_max)s,"
192:                  " max=%(_str_max)s, dtype=%(dtype)s)") % d)
193: 
194: 
195: class iinfo(object):
196:     '''
197:     iinfo(type)
198: 
199:     Machine limits for integer types.
200: 
201:     Attributes
202:     ----------
203:     min : int
204:         The smallest integer expressible by the type.
205:     max : int
206:         The largest integer expressible by the type.
207: 
208:     Parameters
209:     ----------
210:     int_type : integer type, dtype, or instance
211:         The kind of integer data type to get information about.
212: 
213:     See Also
214:     --------
215:     finfo : The equivalent for floating point data types.
216: 
217:     Examples
218:     --------
219:     With types:
220: 
221:     >>> ii16 = np.iinfo(np.int16)
222:     >>> ii16.min
223:     -32768
224:     >>> ii16.max
225:     32767
226:     >>> ii32 = np.iinfo(np.int32)
227:     >>> ii32.min
228:     -2147483648
229:     >>> ii32.max
230:     2147483647
231: 
232:     With instances:
233: 
234:     >>> ii32 = np.iinfo(np.int32(10))
235:     >>> ii32.min
236:     -2147483648
237:     >>> ii32.max
238:     2147483647
239: 
240:     '''
241: 
242:     _min_vals = {}
243:     _max_vals = {}
244: 
245:     def __init__(self, int_type):
246:         try:
247:             self.dtype = numeric.dtype(int_type)
248:         except TypeError:
249:             self.dtype = numeric.dtype(type(int_type))
250:         self.kind = self.dtype.kind
251:         self.bits = self.dtype.itemsize * 8
252:         self.key = "%s%d" % (self.kind, self.bits)
253:         if self.kind not in 'iu':
254:             raise ValueError("Invalid integer data type.")
255: 
256:     def min(self):
257:         '''Minimum value of given dtype.'''
258:         if self.kind == 'u':
259:             return 0
260:         else:
261:             try:
262:                 val = iinfo._min_vals[self.key]
263:             except KeyError:
264:                 val = int(-(1 << (self.bits-1)))
265:                 iinfo._min_vals[self.key] = val
266:             return val
267: 
268:     min = property(min)
269: 
270:     def max(self):
271:         '''Maximum value of given dtype.'''
272:         try:
273:             val = iinfo._max_vals[self.key]
274:         except KeyError:
275:             if self.kind == 'u':
276:                 val = int((1 << self.bits) - 1)
277:             else:
278:                 val = int((1 << (self.bits-1)) - 1)
279:             iinfo._max_vals[self.key] = val
280:         return val
281: 
282:     max = property(max)
283: 
284:     def __str__(self):
285:         '''String representation.'''
286:         fmt = (
287:             'Machine parameters for %(dtype)s\n'
288:             '---------------------------------------------------------------\n'
289:             'min = %(min)s\n'
290:             'max = %(max)s\n'
291:             '---------------------------------------------------------------\n'
292:             )
293:         return fmt % {'dtype': self.dtype, 'min': self.min, 'max': self.max}
294: 
295:     def __repr__(self):
296:         return "%s(min=%s, max=%s, dtype=%s)" % (self.__class__.__name__,
297:                                     self.min, self.max, self.dtype)
298: 
299: if __name__ == '__main__':
300:     f = finfo(ntypes.single)
301:     print('single epsilon:', f.eps)
302:     print('single tiny:', f.tiny)
303:     f = finfo(ntypes.float)
304:     print('float epsilon:', f.eps)
305:     print('float tiny:', f.tiny)
306:     f = finfo(ntypes.longfloat)
307:     print('longfloat epsilon:', f.eps)
308:     print('longfloat tiny:', f.tiny)
309: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_5763 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'Machine limits for Float32 and Float64 and (long double) if available...\n\n')

# Assigning a List to a Name (line 6):
__all__ = ['finfo', 'iinfo']
module_type_store.set_exportable_members(['finfo', 'iinfo'])

# Obtaining an instance of the builtin type 'list' (line 6)
list_5764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_5765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 11), 'str', 'finfo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_5764, str_5765)
# Adding element type (line 6)
str_5766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 20), 'str', 'iinfo')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 10), list_5764, str_5766)

# Assigning a type to the variable '__all__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__all__', list_5764)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.core.machar import MachAr' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5767 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.machar')

if (type(import_5767) is not StypyTypeError):

    if (import_5767 != 'pyd_module'):
        __import__(import_5767)
        sys_modules_5768 = sys.modules[import_5767]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.machar', sys_modules_5768.module_type_store, module_type_store, ['MachAr'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_5768, sys_modules_5768.module_type_store, module_type_store)
    else:
        from numpy.core.machar import MachAr

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.machar', None, module_type_store, ['MachAr'], [MachAr])

else:
    # Assigning a type to the variable 'numpy.core.machar' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.core.machar', import_5767)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from numpy.core import numeric' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5769 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core')

if (type(import_5769) is not StypyTypeError):

    if (import_5769 != 'pyd_module'):
        __import__(import_5769)
        sys_modules_5770 = sys.modules[import_5769]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core', sys_modules_5770.module_type_store, module_type_store, ['numeric'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_5770, sys_modules_5770.module_type_store, module_type_store)
    else:
        from numpy.core import numeric

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core', None, module_type_store, ['numeric'], [numeric])

else:
    # Assigning a type to the variable 'numpy.core' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'numpy.core', import_5769)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.core import ntypes' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5771 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core')

if (type(import_5771) is not StypyTypeError):

    if (import_5771 != 'pyd_module'):
        __import__(import_5771)
        sys_modules_5772 = sys.modules[import_5771]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', sys_modules_5772.module_type_store, module_type_store, ['numerictypes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_5772, sys_modules_5772.module_type_store, module_type_store)
    else:
        from numpy.core import numerictypes as ntypes

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', None, module_type_store, ['numerictypes'], [ntypes])

else:
    # Assigning a type to the variable 'numpy.core' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', import_5771)

# Adding an alias
module_type_store.add_alias('ntypes', 'numerictypes')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from numpy.core.numeric import array' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/core/')
import_5773 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core.numeric')

if (type(import_5773) is not StypyTypeError):

    if (import_5773 != 'pyd_module'):
        __import__(import_5773)
        sys_modules_5774 = sys.modules[import_5773]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core.numeric', sys_modules_5774.module_type_store, module_type_store, ['array'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_5774, sys_modules_5774.module_type_store, module_type_store)
    else:
        from numpy.core.numeric import array

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core.numeric', None, module_type_store, ['array'], [array])

else:
    # Assigning a type to the variable 'numpy.core.numeric' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'numpy.core.numeric', import_5773)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/core/')


@norecursion
def _frz(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_frz'
    module_type_store = module_type_store.open_function_context('_frz', 13, 0, False)
    
    # Passed parameters checking function
    _frz.stypy_localization = localization
    _frz.stypy_type_of_self = None
    _frz.stypy_type_store = module_type_store
    _frz.stypy_function_name = '_frz'
    _frz.stypy_param_names_list = ['a']
    _frz.stypy_varargs_param_name = None
    _frz.stypy_kwargs_param_name = None
    _frz.stypy_call_defaults = defaults
    _frz.stypy_call_varargs = varargs
    _frz.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_frz', ['a'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_frz', localization, ['a'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_frz(...)' code ##################

    str_5775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'fix rank-0 --> rank-1')
    
    
    # Getting the type of 'a' (line 15)
    a_5776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 7), 'a')
    # Obtaining the member 'ndim' of a type (line 15)
    ndim_5777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 7), a_5776, 'ndim')
    int_5778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 17), 'int')
    # Applying the binary operator '==' (line 15)
    result_eq_5779 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 7), '==', ndim_5777, int_5778)
    
    # Testing the type of an if condition (line 15)
    if_condition_5780 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 15, 4), result_eq_5779)
    # Assigning a type to the variable 'if_condition_5780' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'if_condition_5780', if_condition_5780)
    # SSA begins for if statement (line 15)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Tuple to a Attribute (line 16):
    
    # Obtaining an instance of the builtin type 'tuple' (line 16)
    tuple_5781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 16)
    # Adding element type (line 16)
    int_5782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 19), tuple_5781, int_5782)
    
    # Getting the type of 'a' (line 16)
    a_5783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'a')
    # Setting the type of the member 'shape' of a type (line 16)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), a_5783, 'shape', tuple_5781)
    # SSA join for if statement (line 15)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'a' (line 17)
    a_5784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 11), 'a')
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type', a_5784)
    
    # ################# End of '_frz(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_frz' in the type store
    # Getting the type of 'stypy_return_type' (line 13)
    stypy_return_type_5785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_5785)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_frz'
    return stypy_return_type_5785

# Assigning a type to the variable '_frz' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_frz', _frz)

# Assigning a Dict to a Name (line 19):

# Obtaining an instance of the builtin type 'dict' (line 19)
dict_5786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 20), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 19)
# Adding element type (key, value) (line 19)
# Getting the type of 'ntypes' (line 20)
ntypes_5787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'ntypes')
# Obtaining the member 'csingle' of a type (line 20)
csingle_5788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 4), ntypes_5787, 'csingle')
# Getting the type of 'ntypes' (line 20)
ntypes_5789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 20), 'ntypes')
# Obtaining the member 'single' of a type (line 20)
single_5790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 20), ntypes_5789, 'single')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), dict_5786, (csingle_5788, single_5790))
# Adding element type (key, value) (line 19)
# Getting the type of 'ntypes' (line 21)
ntypes_5791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'ntypes')
# Obtaining the member 'complex_' of a type (line 21)
complex__5792 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 4), ntypes_5791, 'complex_')
# Getting the type of 'ntypes' (line 21)
ntypes_5793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 21), 'ntypes')
# Obtaining the member 'float_' of a type (line 21)
float__5794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 21), ntypes_5793, 'float_')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), dict_5786, (complex__5792, float__5794))
# Adding element type (key, value) (line 19)
# Getting the type of 'ntypes' (line 22)
ntypes_5795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'ntypes')
# Obtaining the member 'clongfloat' of a type (line 22)
clongfloat_5796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 4), ntypes_5795, 'clongfloat')
# Getting the type of 'ntypes' (line 22)
ntypes_5797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 23), 'ntypes')
# Obtaining the member 'longfloat' of a type (line 22)
longfloat_5798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 23), ntypes_5797, 'longfloat')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 20), dict_5786, (clongfloat_5796, longfloat_5798))

# Assigning a type to the variable '_convert_to_float' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '_convert_to_float', dict_5786)
# Declaration of the 'finfo' class

class finfo(object, ):
    str_5799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, (-1)), 'str', "\n    finfo(dtype)\n\n    Machine limits for floating point types.\n\n    Attributes\n    ----------\n    eps : float\n        The smallest representable positive number such that\n        ``1.0 + eps != 1.0``.  Type of `eps` is an appropriate floating\n        point type.\n    epsneg : floating point number of the appropriate type\n        The smallest representable positive number such that\n        ``1.0 - epsneg != 1.0``.\n    iexp : int\n        The number of bits in the exponent portion of the floating point\n        representation.\n    machar : MachAr\n        The object which calculated these parameters and holds more\n        detailed information.\n    machep : int\n        The exponent that yields `eps`.\n    max : floating point number of the appropriate type\n        The largest representable number.\n    maxexp : int\n        The smallest positive power of the base (2) that causes overflow.\n    min : floating point number of the appropriate type\n        The smallest representable number, typically ``-max``.\n    minexp : int\n        The most negative power of the base (2) consistent with there\n        being no leading 0's in the mantissa.\n    negep : int\n        The exponent that yields `epsneg`.\n    nexp : int\n        The number of bits in the exponent including its sign and bias.\n    nmant : int\n        The number of bits in the mantissa.\n    precision : int\n        The approximate number of decimal digits to which this kind of\n        float is precise.\n    resolution : floating point number of the appropriate type\n        The approximate decimal resolution of this type, i.e.,\n        ``10**-precision``.\n    tiny : float\n        The smallest positive usable number.  Type of `tiny` is an\n        appropriate floating point type.\n\n    Parameters\n    ----------\n    dtype : float, dtype, or instance\n        Kind of floating point data-type about which to get information.\n\n    See Also\n    --------\n    MachAr : The implementation of the tests that produce this information.\n    iinfo : The equivalent for integer data types.\n\n    Notes\n    -----\n    For developers of NumPy: do not instantiate this at the module level.\n    The initial calculation of these parameters is expensive and negatively\n    impacts import times.  These objects are cached, so calling ``finfo()``\n    repeatedly inside your functions is not a problem.\n\n    ")

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 94, 4, False)
        # Assigning a type to the variable 'self' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        finfo.__new__.__dict__.__setitem__('stypy_localization', localization)
        finfo.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        finfo.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        finfo.__new__.__dict__.__setitem__('stypy_function_name', 'finfo.__new__')
        finfo.__new__.__dict__.__setitem__('stypy_param_names_list', ['dtype'])
        finfo.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        finfo.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        finfo.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        finfo.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        finfo.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        finfo.__new__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'finfo.__new__', ['dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 95)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 96):
        
        # Call to dtype(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'dtype' (line 96)
        dtype_5802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 34), 'dtype', False)
        # Processing the call keyword arguments (line 96)
        kwargs_5803 = {}
        # Getting the type of 'numeric' (line 96)
        numeric_5800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 20), 'numeric', False)
        # Obtaining the member 'dtype' of a type (line 96)
        dtype_5801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 20), numeric_5800, 'dtype')
        # Calling dtype(args, kwargs) (line 96)
        dtype_call_result_5804 = invoke(stypy.reporting.localization.Localization(__file__, 96, 20), dtype_5801, *[dtype_5802], **kwargs_5803)
        
        # Assigning a type to the variable 'dtype' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'dtype', dtype_call_result_5804)
        # SSA branch for the except part of a try statement (line 95)
        # SSA branch for the except 'TypeError' branch of a try statement (line 95)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 99):
        
        # Call to dtype(...): (line 99)
        # Processing the call arguments (line 99)
        
        # Call to type(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'dtype' (line 99)
        dtype_5808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 39), 'dtype', False)
        # Processing the call keyword arguments (line 99)
        kwargs_5809 = {}
        # Getting the type of 'type' (line 99)
        type_5807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'type', False)
        # Calling type(args, kwargs) (line 99)
        type_call_result_5810 = invoke(stypy.reporting.localization.Localization(__file__, 99, 34), type_5807, *[dtype_5808], **kwargs_5809)
        
        # Processing the call keyword arguments (line 99)
        kwargs_5811 = {}
        # Getting the type of 'numeric' (line 99)
        numeric_5805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 20), 'numeric', False)
        # Obtaining the member 'dtype' of a type (line 99)
        dtype_5806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 20), numeric_5805, 'dtype')
        # Calling dtype(args, kwargs) (line 99)
        dtype_call_result_5812 = invoke(stypy.reporting.localization.Localization(__file__, 99, 20), dtype_5806, *[type_call_result_5810], **kwargs_5811)
        
        # Assigning a type to the variable 'dtype' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'dtype', dtype_call_result_5812)
        # SSA join for try-except statement (line 95)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 101):
        
        # Call to get(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'dtype' (line 101)
        dtype_5816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 35), 'dtype', False)
        # Getting the type of 'None' (line 101)
        None_5817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 42), 'None', False)
        # Processing the call keyword arguments (line 101)
        kwargs_5818 = {}
        # Getting the type of 'cls' (line 101)
        cls_5813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'cls', False)
        # Obtaining the member '_finfo_cache' of a type (line 101)
        _finfo_cache_5814 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 14), cls_5813, '_finfo_cache')
        # Obtaining the member 'get' of a type (line 101)
        get_5815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 14), _finfo_cache_5814, 'get')
        # Calling get(args, kwargs) (line 101)
        get_call_result_5819 = invoke(stypy.reporting.localization.Localization(__file__, 101, 14), get_5815, *[dtype_5816, None_5817], **kwargs_5818)
        
        # Assigning a type to the variable 'obj' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'obj', get_call_result_5819)
        
        # Type idiom detected: calculating its left and rigth part (line 102)
        # Getting the type of 'obj' (line 102)
        obj_5820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'obj')
        # Getting the type of 'None' (line 102)
        None_5821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 22), 'None')
        
        (may_be_5822, more_types_in_union_5823) = may_not_be_none(obj_5820, None_5821)

        if may_be_5822:

            if more_types_in_union_5823:
                # Runtime conditional SSA (line 102)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'obj' (line 103)
            obj_5824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 19), 'obj')
            # Assigning a type to the variable 'stypy_return_type' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'stypy_return_type', obj_5824)

            if more_types_in_union_5823:
                # SSA join for if statement (line 102)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 104):
        
        # Obtaining an instance of the builtin type 'list' (line 104)
        list_5825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 104)
        # Adding element type (line 104)
        # Getting the type of 'dtype' (line 104)
        dtype_5826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 18), 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 17), list_5825, dtype_5826)
        
        # Assigning a type to the variable 'dtypes' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'dtypes', list_5825)
        
        # Assigning a Call to a Name (line 105):
        
        # Call to obj2sctype(...): (line 105)
        # Processing the call arguments (line 105)
        # Getting the type of 'dtype' (line 105)
        dtype_5829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 38), 'dtype', False)
        # Processing the call keyword arguments (line 105)
        kwargs_5830 = {}
        # Getting the type of 'numeric' (line 105)
        numeric_5827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'numeric', False)
        # Obtaining the member 'obj2sctype' of a type (line 105)
        obj2sctype_5828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 19), numeric_5827, 'obj2sctype')
        # Calling obj2sctype(args, kwargs) (line 105)
        obj2sctype_call_result_5831 = invoke(stypy.reporting.localization.Localization(__file__, 105, 19), obj2sctype_5828, *[dtype_5829], **kwargs_5830)
        
        # Assigning a type to the variable 'newdtype' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'newdtype', obj2sctype_call_result_5831)
        
        
        # Getting the type of 'newdtype' (line 106)
        newdtype_5832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'newdtype')
        # Getting the type of 'dtype' (line 106)
        dtype_5833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 27), 'dtype')
        # Applying the binary operator 'isnot' (line 106)
        result_is_not_5834 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), 'isnot', newdtype_5832, dtype_5833)
        
        # Testing the type of an if condition (line 106)
        if_condition_5835 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_is_not_5834)
        # Assigning a type to the variable 'if_condition_5835' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_5835', if_condition_5835)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'newdtype' (line 107)
        newdtype_5838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 26), 'newdtype', False)
        # Processing the call keyword arguments (line 107)
        kwargs_5839 = {}
        # Getting the type of 'dtypes' (line 107)
        dtypes_5836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'dtypes', False)
        # Obtaining the member 'append' of a type (line 107)
        append_5837 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), dtypes_5836, 'append')
        # Calling append(args, kwargs) (line 107)
        append_call_result_5840 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), append_5837, *[newdtype_5838], **kwargs_5839)
        
        
        # Assigning a Name to a Name (line 108):
        # Getting the type of 'newdtype' (line 108)
        newdtype_5841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'newdtype')
        # Assigning a type to the variable 'dtype' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'dtype', newdtype_5841)
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to issubclass(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'dtype' (line 109)
        dtype_5843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 26), 'dtype', False)
        # Getting the type of 'numeric' (line 109)
        numeric_5844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 33), 'numeric', False)
        # Obtaining the member 'inexact' of a type (line 109)
        inexact_5845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 33), numeric_5844, 'inexact')
        # Processing the call keyword arguments (line 109)
        kwargs_5846 = {}
        # Getting the type of 'issubclass' (line 109)
        issubclass_5842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 15), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 109)
        issubclass_call_result_5847 = invoke(stypy.reporting.localization.Localization(__file__, 109, 15), issubclass_5842, *[dtype_5843, inexact_5845], **kwargs_5846)
        
        # Applying the 'not' unary operator (line 109)
        result_not__5848 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 11), 'not', issubclass_call_result_5847)
        
        # Testing the type of an if condition (line 109)
        if_condition_5849 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 8), result_not__5848)
        # Assigning a type to the variable 'if_condition_5849' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'if_condition_5849', if_condition_5849)
        # SSA begins for if statement (line 109)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 110)
        # Processing the call arguments (line 110)
        str_5851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 29), 'str', 'data type %r not inexact')
        # Getting the type of 'dtype' (line 110)
        dtype_5852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 59), 'dtype', False)
        # Applying the binary operator '%' (line 110)
        result_mod_5853 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 29), '%', str_5851, dtype_5852)
        
        # Processing the call keyword arguments (line 110)
        kwargs_5854 = {}
        # Getting the type of 'ValueError' (line 110)
        ValueError_5850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 110)
        ValueError_call_result_5855 = invoke(stypy.reporting.localization.Localization(__file__, 110, 18), ValueError_5850, *[result_mod_5853], **kwargs_5854)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 110, 12), ValueError_call_result_5855, 'raise parameter', BaseException)
        # SSA join for if statement (line 109)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 111):
        
        # Call to get(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'dtype' (line 111)
        dtype_5859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 35), 'dtype', False)
        # Getting the type of 'None' (line 111)
        None_5860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 42), 'None', False)
        # Processing the call keyword arguments (line 111)
        kwargs_5861 = {}
        # Getting the type of 'cls' (line 111)
        cls_5856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 14), 'cls', False)
        # Obtaining the member '_finfo_cache' of a type (line 111)
        _finfo_cache_5857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 14), cls_5856, '_finfo_cache')
        # Obtaining the member 'get' of a type (line 111)
        get_5858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 14), _finfo_cache_5857, 'get')
        # Calling get(args, kwargs) (line 111)
        get_call_result_5862 = invoke(stypy.reporting.localization.Localization(__file__, 111, 14), get_5858, *[dtype_5859, None_5860], **kwargs_5861)
        
        # Assigning a type to the variable 'obj' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'obj', get_call_result_5862)
        
        # Type idiom detected: calculating its left and rigth part (line 112)
        # Getting the type of 'obj' (line 112)
        obj_5863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'obj')
        # Getting the type of 'None' (line 112)
        None_5864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 22), 'None')
        
        (may_be_5865, more_types_in_union_5866) = may_not_be_none(obj_5863, None_5864)

        if may_be_5865:

            if more_types_in_union_5866:
                # Runtime conditional SSA (line 112)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'obj' (line 113)
            obj_5867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'obj')
            # Assigning a type to the variable 'stypy_return_type' (line 113)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'stypy_return_type', obj_5867)

            if more_types_in_union_5866:
                # SSA join for if statement (line 112)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        
        # Call to issubclass(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'dtype' (line 114)
        dtype_5869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 26), 'dtype', False)
        # Getting the type of 'numeric' (line 114)
        numeric_5870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 33), 'numeric', False)
        # Obtaining the member 'floating' of a type (line 114)
        floating_5871 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 33), numeric_5870, 'floating')
        # Processing the call keyword arguments (line 114)
        kwargs_5872 = {}
        # Getting the type of 'issubclass' (line 114)
        issubclass_5868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'issubclass', False)
        # Calling issubclass(args, kwargs) (line 114)
        issubclass_call_result_5873 = invoke(stypy.reporting.localization.Localization(__file__, 114, 15), issubclass_5868, *[dtype_5869, floating_5871], **kwargs_5872)
        
        # Applying the 'not' unary operator (line 114)
        result_not__5874 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 11), 'not', issubclass_call_result_5873)
        
        # Testing the type of an if condition (line 114)
        if_condition_5875 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 8), result_not__5874)
        # Assigning a type to the variable 'if_condition_5875' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'if_condition_5875', if_condition_5875)
        # SSA begins for if statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Name (line 115):
        
        # Obtaining the type of the subscript
        # Getting the type of 'dtype' (line 115)
        dtype_5876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 41), 'dtype')
        # Getting the type of '_convert_to_float' (line 115)
        _convert_to_float_5877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 23), '_convert_to_float')
        # Obtaining the member '__getitem__' of a type (line 115)
        getitem___5878 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 23), _convert_to_float_5877, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 115)
        subscript_call_result_5879 = invoke(stypy.reporting.localization.Localization(__file__, 115, 23), getitem___5878, dtype_5876)
        
        # Assigning a type to the variable 'newdtype' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'newdtype', subscript_call_result_5879)
        
        
        # Getting the type of 'newdtype' (line 116)
        newdtype_5880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'newdtype')
        # Getting the type of 'dtype' (line 116)
        dtype_5881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'dtype')
        # Applying the binary operator 'isnot' (line 116)
        result_is_not_5882 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 15), 'isnot', newdtype_5880, dtype_5881)
        
        # Testing the type of an if condition (line 116)
        if_condition_5883 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 116, 12), result_is_not_5882)
        # Assigning a type to the variable 'if_condition_5883' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'if_condition_5883', if_condition_5883)
        # SSA begins for if statement (line 116)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'newdtype' (line 117)
        newdtype_5886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 30), 'newdtype', False)
        # Processing the call keyword arguments (line 117)
        kwargs_5887 = {}
        # Getting the type of 'dtypes' (line 117)
        dtypes_5884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'dtypes', False)
        # Obtaining the member 'append' of a type (line 117)
        append_5885 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), dtypes_5884, 'append')
        # Calling append(args, kwargs) (line 117)
        append_call_result_5888 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), append_5885, *[newdtype_5886], **kwargs_5887)
        
        
        # Assigning a Name to a Name (line 118):
        # Getting the type of 'newdtype' (line 118)
        newdtype_5889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 24), 'newdtype')
        # Assigning a type to the variable 'dtype' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'dtype', newdtype_5889)
        # SSA join for if statement (line 116)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 114)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 119):
        
        # Call to get(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'dtype' (line 119)
        dtype_5893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'dtype', False)
        # Getting the type of 'None' (line 119)
        None_5894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 42), 'None', False)
        # Processing the call keyword arguments (line 119)
        kwargs_5895 = {}
        # Getting the type of 'cls' (line 119)
        cls_5890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 14), 'cls', False)
        # Obtaining the member '_finfo_cache' of a type (line 119)
        _finfo_cache_5891 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 14), cls_5890, '_finfo_cache')
        # Obtaining the member 'get' of a type (line 119)
        get_5892 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 14), _finfo_cache_5891, 'get')
        # Calling get(args, kwargs) (line 119)
        get_call_result_5896 = invoke(stypy.reporting.localization.Localization(__file__, 119, 14), get_5892, *[dtype_5893, None_5894], **kwargs_5895)
        
        # Assigning a type to the variable 'obj' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'obj', get_call_result_5896)
        
        # Type idiom detected: calculating its left and rigth part (line 120)
        # Getting the type of 'obj' (line 120)
        obj_5897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'obj')
        # Getting the type of 'None' (line 120)
        None_5898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'None')
        
        (may_be_5899, more_types_in_union_5900) = may_not_be_none(obj_5897, None_5898)

        if may_be_5899:

            if more_types_in_union_5900:
                # Runtime conditional SSA (line 120)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'obj' (line 121)
            obj_5901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 19), 'obj')
            # Assigning a type to the variable 'stypy_return_type' (line 121)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'stypy_return_type', obj_5901)

            if more_types_in_union_5900:
                # SSA join for if statement (line 120)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a Call to a Name (line 122):
        
        # Call to _init(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'dtype' (line 122)
        dtype_5908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 40), 'dtype', False)
        # Processing the call keyword arguments (line 122)
        kwargs_5909 = {}
        
        # Call to __new__(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'cls' (line 122)
        cls_5904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'cls', False)
        # Processing the call keyword arguments (line 122)
        kwargs_5905 = {}
        # Getting the type of 'object' (line 122)
        object_5902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 14), 'object', False)
        # Obtaining the member '__new__' of a type (line 122)
        new___5903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 14), object_5902, '__new__')
        # Calling __new__(args, kwargs) (line 122)
        new___call_result_5906 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), new___5903, *[cls_5904], **kwargs_5905)
        
        # Obtaining the member '_init' of a type (line 122)
        _init_5907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 14), new___call_result_5906, '_init')
        # Calling _init(args, kwargs) (line 122)
        _init_call_result_5910 = invoke(stypy.reporting.localization.Localization(__file__, 122, 14), _init_5907, *[dtype_5908], **kwargs_5909)
        
        # Assigning a type to the variable 'obj' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'obj', _init_call_result_5910)
        
        # Getting the type of 'dtypes' (line 123)
        dtypes_5911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 18), 'dtypes')
        # Testing the type of a for loop iterable (line 123)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 8), dtypes_5911)
        # Getting the type of the for loop variable (line 123)
        for_loop_var_5912 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 8), dtypes_5911)
        # Assigning a type to the variable 'dt' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'dt', for_loop_var_5912)
        # SSA begins for a for statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Name to a Subscript (line 124):
        # Getting the type of 'obj' (line 124)
        obj_5913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 35), 'obj')
        # Getting the type of 'cls' (line 124)
        cls_5914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'cls')
        # Obtaining the member '_finfo_cache' of a type (line 124)
        _finfo_cache_5915 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 12), cls_5914, '_finfo_cache')
        # Getting the type of 'dt' (line 124)
        dt_5916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 29), 'dt')
        # Storing an element on a container (line 124)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 12), _finfo_cache_5915, (dt_5916, obj_5913))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'obj' (line 125)
        obj_5917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 15), 'obj')
        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', obj_5917)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 94)
        stypy_return_type_5918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5918)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_5918


    @norecursion
    def _init(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_init'
        module_type_store = module_type_store.open_function_context('_init', 127, 4, False)
        # Assigning a type to the variable 'self' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        finfo._init.__dict__.__setitem__('stypy_localization', localization)
        finfo._init.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        finfo._init.__dict__.__setitem__('stypy_type_store', module_type_store)
        finfo._init.__dict__.__setitem__('stypy_function_name', 'finfo._init')
        finfo._init.__dict__.__setitem__('stypy_param_names_list', ['dtype'])
        finfo._init.__dict__.__setitem__('stypy_varargs_param_name', None)
        finfo._init.__dict__.__setitem__('stypy_kwargs_param_name', None)
        finfo._init.__dict__.__setitem__('stypy_call_defaults', defaults)
        finfo._init.__dict__.__setitem__('stypy_call_varargs', varargs)
        finfo._init.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        finfo._init.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'finfo._init', ['dtype'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_init', localization, ['dtype'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_init(...)' code ##################

        
        # Assigning a Call to a Attribute (line 128):
        
        # Call to dtype(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'dtype' (line 128)
        dtype_5921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 35), 'dtype', False)
        # Processing the call keyword arguments (line 128)
        kwargs_5922 = {}
        # Getting the type of 'numeric' (line 128)
        numeric_5919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'numeric', False)
        # Obtaining the member 'dtype' of a type (line 128)
        dtype_5920 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 21), numeric_5919, 'dtype')
        # Calling dtype(args, kwargs) (line 128)
        dtype_call_result_5923 = invoke(stypy.reporting.localization.Localization(__file__, 128, 21), dtype_5920, *[dtype_5921], **kwargs_5922)
        
        # Getting the type of 'self' (line 128)
        self_5924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'self')
        # Setting the type of the member 'dtype' of a type (line 128)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), self_5924, 'dtype', dtype_call_result_5923)
        
        
        # Getting the type of 'dtype' (line 129)
        dtype_5925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'dtype')
        # Getting the type of 'ntypes' (line 129)
        ntypes_5926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 20), 'ntypes')
        # Obtaining the member 'double' of a type (line 129)
        double_5927 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 20), ntypes_5926, 'double')
        # Applying the binary operator 'is' (line 129)
        result_is__5928 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 11), 'is', dtype_5925, double_5927)
        
        # Testing the type of an if condition (line 129)
        if_condition_5929 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), result_is__5928)
        # Assigning a type to the variable 'if_condition_5929' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_5929', if_condition_5929)
        # SSA begins for if statement (line 129)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 130):
        # Getting the type of 'ntypes' (line 130)
        ntypes_5930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'ntypes')
        # Obtaining the member 'int64' of a type (line 130)
        int64_5931 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 20), ntypes_5930, 'int64')
        # Assigning a type to the variable 'itype' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'itype', int64_5931)
        
        # Assigning a Str to a Name (line 131):
        str_5932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 18), 'str', '%24.16e')
        # Assigning a type to the variable 'fmt' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'fmt', str_5932)
        
        # Assigning a Str to a Name (line 132):
        str_5933 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 23), 'str', 'double')
        # Assigning a type to the variable 'precname' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'precname', str_5933)
        # SSA branch for the else part of an if statement (line 129)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'dtype' (line 133)
        dtype_5934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'dtype')
        # Getting the type of 'ntypes' (line 133)
        ntypes_5935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'ntypes')
        # Obtaining the member 'single' of a type (line 133)
        single_5936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), ntypes_5935, 'single')
        # Applying the binary operator 'is' (line 133)
        result_is__5937 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 13), 'is', dtype_5934, single_5936)
        
        # Testing the type of an if condition (line 133)
        if_condition_5938 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 133, 13), result_is__5937)
        # Assigning a type to the variable 'if_condition_5938' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 13), 'if_condition_5938', if_condition_5938)
        # SSA begins for if statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 134):
        # Getting the type of 'ntypes' (line 134)
        ntypes_5939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'ntypes')
        # Obtaining the member 'int32' of a type (line 134)
        int32_5940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 20), ntypes_5939, 'int32')
        # Assigning a type to the variable 'itype' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'itype', int32_5940)
        
        # Assigning a Str to a Name (line 135):
        str_5941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 18), 'str', '%15.7e')
        # Assigning a type to the variable 'fmt' (line 135)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'fmt', str_5941)
        
        # Assigning a Str to a Name (line 136):
        str_5942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 23), 'str', 'single')
        # Assigning a type to the variable 'precname' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 12), 'precname', str_5942)
        # SSA branch for the else part of an if statement (line 133)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'dtype' (line 137)
        dtype_5943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'dtype')
        # Getting the type of 'ntypes' (line 137)
        ntypes_5944 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'ntypes')
        # Obtaining the member 'longdouble' of a type (line 137)
        longdouble_5945 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 22), ntypes_5944, 'longdouble')
        # Applying the binary operator 'is' (line 137)
        result_is__5946 = python_operator(stypy.reporting.localization.Localization(__file__, 137, 13), 'is', dtype_5943, longdouble_5945)
        
        # Testing the type of an if condition (line 137)
        if_condition_5947 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 13), result_is__5946)
        # Assigning a type to the variable 'if_condition_5947' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'if_condition_5947', if_condition_5947)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 138):
        # Getting the type of 'ntypes' (line 138)
        ntypes_5948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'ntypes')
        # Obtaining the member 'longlong' of a type (line 138)
        longlong_5949 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 20), ntypes_5948, 'longlong')
        # Assigning a type to the variable 'itype' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'itype', longlong_5949)
        
        # Assigning a Str to a Name (line 139):
        str_5950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 18), 'str', '%s')
        # Assigning a type to the variable 'fmt' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 12), 'fmt', str_5950)
        
        # Assigning a Str to a Name (line 140):
        str_5951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 23), 'str', 'long double')
        # Assigning a type to the variable 'precname' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'precname', str_5951)
        # SSA branch for the else part of an if statement (line 137)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'dtype' (line 141)
        dtype_5952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'dtype')
        # Getting the type of 'ntypes' (line 141)
        ntypes_5953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'ntypes')
        # Obtaining the member 'half' of a type (line 141)
        half_5954 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 22), ntypes_5953, 'half')
        # Applying the binary operator 'is' (line 141)
        result_is__5955 = python_operator(stypy.reporting.localization.Localization(__file__, 141, 13), 'is', dtype_5952, half_5954)
        
        # Testing the type of an if condition (line 141)
        if_condition_5956 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 13), result_is__5955)
        # Assigning a type to the variable 'if_condition_5956' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'if_condition_5956', if_condition_5956)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Name (line 142):
        # Getting the type of 'ntypes' (line 142)
        ntypes_5957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'ntypes')
        # Obtaining the member 'int16' of a type (line 142)
        int16_5958 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 20), ntypes_5957, 'int16')
        # Assigning a type to the variable 'itype' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'itype', int16_5958)
        
        # Assigning a Str to a Name (line 143):
        str_5959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 18), 'str', '%12.5e')
        # Assigning a type to the variable 'fmt' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 12), 'fmt', str_5959)
        
        # Assigning a Str to a Name (line 144):
        str_5960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 23), 'str', 'half')
        # Assigning a type to the variable 'precname' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'precname', str_5960)
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Call to repr(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'dtype' (line 146)
        dtype_5963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 34), 'dtype', False)
        # Processing the call keyword arguments (line 146)
        kwargs_5964 = {}
        # Getting the type of 'repr' (line 146)
        repr_5962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'repr', False)
        # Calling repr(args, kwargs) (line 146)
        repr_call_result_5965 = invoke(stypy.reporting.localization.Localization(__file__, 146, 29), repr_5962, *[dtype_5963], **kwargs_5964)
        
        # Processing the call keyword arguments (line 146)
        kwargs_5966 = {}
        # Getting the type of 'ValueError' (line 146)
        ValueError_5961 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 146)
        ValueError_call_result_5967 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), ValueError_5961, *[repr_call_result_5965], **kwargs_5966)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 12), ValueError_call_result_5967, 'raise parameter', BaseException)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 133)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 129)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 148):
        
        # Call to MachAr(...): (line 148)
        # Processing the call arguments (line 148)

        @norecursion
        def _stypy_temp_lambda_1(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_1'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_1', 148, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_1.stypy_localization = localization
            _stypy_temp_lambda_1.stypy_type_of_self = None
            _stypy_temp_lambda_1.stypy_type_store = module_type_store
            _stypy_temp_lambda_1.stypy_function_name = '_stypy_temp_lambda_1'
            _stypy_temp_lambda_1.stypy_param_names_list = ['v']
            _stypy_temp_lambda_1.stypy_varargs_param_name = None
            _stypy_temp_lambda_1.stypy_kwargs_param_name = None
            _stypy_temp_lambda_1.stypy_call_defaults = defaults
            _stypy_temp_lambda_1.stypy_call_varargs = varargs
            _stypy_temp_lambda_1.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_1', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_1', ['v'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 148)
            # Processing the call arguments (line 148)
            
            # Obtaining an instance of the builtin type 'list' (line 148)
            list_5970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 39), 'list')
            # Adding type elements to the builtin type 'list' instance (line 148)
            # Adding element type (line 148)
            # Getting the type of 'v' (line 148)
            v_5971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 40), 'v', False)
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 148, 39), list_5970, v_5971)
            
            # Getting the type of 'dtype' (line 148)
            dtype_5972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 44), 'dtype', False)
            # Processing the call keyword arguments (line 148)
            kwargs_5973 = {}
            # Getting the type of 'array' (line 148)
            array_5969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 33), 'array', False)
            # Calling array(args, kwargs) (line 148)
            array_call_result_5974 = invoke(stypy.reporting.localization.Localization(__file__, 148, 33), array_5969, *[list_5970, dtype_5972], **kwargs_5973)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 148)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'stypy_return_type', array_call_result_5974)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_1' in the type store
            # Getting the type of 'stypy_return_type' (line 148)
            stypy_return_type_5975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5975)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_1'
            return stypy_return_type_5975

        # Assigning a type to the variable '_stypy_temp_lambda_1' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), '_stypy_temp_lambda_1', _stypy_temp_lambda_1)
        # Getting the type of '_stypy_temp_lambda_1' (line 148)
        _stypy_temp_lambda_1_5976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), '_stypy_temp_lambda_1')

        @norecursion
        def _stypy_temp_lambda_2(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_2'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_2', 149, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_2.stypy_localization = localization
            _stypy_temp_lambda_2.stypy_type_of_self = None
            _stypy_temp_lambda_2.stypy_type_store = module_type_store
            _stypy_temp_lambda_2.stypy_function_name = '_stypy_temp_lambda_2'
            _stypy_temp_lambda_2.stypy_param_names_list = ['v']
            _stypy_temp_lambda_2.stypy_varargs_param_name = None
            _stypy_temp_lambda_2.stypy_kwargs_param_name = None
            _stypy_temp_lambda_2.stypy_call_defaults = defaults
            _stypy_temp_lambda_2.stypy_call_varargs = varargs
            _stypy_temp_lambda_2.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_2', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_2', ['v'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Obtaining the type of the subscript
            int_5977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 55), 'int')
            
            # Call to _frz(...): (line 149)
            # Processing the call arguments (line 149)
            
            # Call to astype(...): (line 149)
            # Processing the call arguments (line 149)
            # Getting the type of 'itype' (line 149)
            itype_5981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 47), 'itype', False)
            # Processing the call keyword arguments (line 149)
            kwargs_5982 = {}
            # Getting the type of 'v' (line 149)
            v_5979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 38), 'v', False)
            # Obtaining the member 'astype' of a type (line 149)
            astype_5980 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 38), v_5979, 'astype')
            # Calling astype(args, kwargs) (line 149)
            astype_call_result_5983 = invoke(stypy.reporting.localization.Localization(__file__, 149, 38), astype_5980, *[itype_5981], **kwargs_5982)
            
            # Processing the call keyword arguments (line 149)
            kwargs_5984 = {}
            # Getting the type of '_frz' (line 149)
            _frz_5978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 33), '_frz', False)
            # Calling _frz(args, kwargs) (line 149)
            _frz_call_result_5985 = invoke(stypy.reporting.localization.Localization(__file__, 149, 33), _frz_5978, *[astype_call_result_5983], **kwargs_5984)
            
            # Obtaining the member '__getitem__' of a type (line 149)
            getitem___5986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 33), _frz_call_result_5985, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 149)
            subscript_call_result_5987 = invoke(stypy.reporting.localization.Localization(__file__, 149, 33), getitem___5986, int_5977)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 149)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'stypy_return_type', subscript_call_result_5987)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_2' in the type store
            # Getting the type of 'stypy_return_type' (line 149)
            stypy_return_type_5988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5988)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_2'
            return stypy_return_type_5988

        # Assigning a type to the variable '_stypy_temp_lambda_2' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), '_stypy_temp_lambda_2', _stypy_temp_lambda_2)
        # Getting the type of '_stypy_temp_lambda_2' (line 149)
        _stypy_temp_lambda_2_5989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 24), '_stypy_temp_lambda_2')

        @norecursion
        def _stypy_temp_lambda_3(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_3'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_3', 150, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_3.stypy_localization = localization
            _stypy_temp_lambda_3.stypy_type_of_self = None
            _stypy_temp_lambda_3.stypy_type_store = module_type_store
            _stypy_temp_lambda_3.stypy_function_name = '_stypy_temp_lambda_3'
            _stypy_temp_lambda_3.stypy_param_names_list = ['v']
            _stypy_temp_lambda_3.stypy_varargs_param_name = None
            _stypy_temp_lambda_3.stypy_kwargs_param_name = None
            _stypy_temp_lambda_3.stypy_call_defaults = defaults
            _stypy_temp_lambda_3.stypy_call_varargs = varargs
            _stypy_temp_lambda_3.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_3', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_3', ['v'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to array(...): (line 150)
            # Processing the call arguments (line 150)
            
            # Obtaining the type of the subscript
            int_5991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 47), 'int')
            
            # Call to _frz(...): (line 150)
            # Processing the call arguments (line 150)
            # Getting the type of 'v' (line 150)
            v_5993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 44), 'v', False)
            # Processing the call keyword arguments (line 150)
            kwargs_5994 = {}
            # Getting the type of '_frz' (line 150)
            _frz_5992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 39), '_frz', False)
            # Calling _frz(args, kwargs) (line 150)
            _frz_call_result_5995 = invoke(stypy.reporting.localization.Localization(__file__, 150, 39), _frz_5992, *[v_5993], **kwargs_5994)
            
            # Obtaining the member '__getitem__' of a type (line 150)
            getitem___5996 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 39), _frz_call_result_5995, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 150)
            subscript_call_result_5997 = invoke(stypy.reporting.localization.Localization(__file__, 150, 39), getitem___5996, int_5991)
            
            # Getting the type of 'dtype' (line 150)
            dtype_5998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 51), 'dtype', False)
            # Processing the call keyword arguments (line 150)
            kwargs_5999 = {}
            # Getting the type of 'array' (line 150)
            array_5990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 33), 'array', False)
            # Calling array(args, kwargs) (line 150)
            array_call_result_6000 = invoke(stypy.reporting.localization.Localization(__file__, 150, 33), array_5990, *[subscript_call_result_5997, dtype_5998], **kwargs_5999)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'stypy_return_type', array_call_result_6000)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_3' in the type store
            # Getting the type of 'stypy_return_type' (line 150)
            stypy_return_type_6001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6001)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_3'
            return stypy_return_type_6001

        # Assigning a type to the variable '_stypy_temp_lambda_3' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), '_stypy_temp_lambda_3', _stypy_temp_lambda_3)
        # Getting the type of '_stypy_temp_lambda_3' (line 150)
        _stypy_temp_lambda_3_6002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), '_stypy_temp_lambda_3')

        @norecursion
        def _stypy_temp_lambda_4(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_4'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_4', 151, 24, True)
            # Passed parameters checking function
            _stypy_temp_lambda_4.stypy_localization = localization
            _stypy_temp_lambda_4.stypy_type_of_self = None
            _stypy_temp_lambda_4.stypy_type_store = module_type_store
            _stypy_temp_lambda_4.stypy_function_name = '_stypy_temp_lambda_4'
            _stypy_temp_lambda_4.stypy_param_names_list = ['v']
            _stypy_temp_lambda_4.stypy_varargs_param_name = None
            _stypy_temp_lambda_4.stypy_kwargs_param_name = None
            _stypy_temp_lambda_4.stypy_call_defaults = defaults
            _stypy_temp_lambda_4.stypy_call_varargs = varargs
            _stypy_temp_lambda_4.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_4', ['v'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_4', ['v'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            # Getting the type of 'fmt' (line 151)
            fmt_6003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 34), 'fmt', False)
            
            # Call to array(...): (line 151)
            # Processing the call arguments (line 151)
            
            # Obtaining the type of the subscript
            int_6005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 54), 'int')
            
            # Call to _frz(...): (line 151)
            # Processing the call arguments (line 151)
            # Getting the type of 'v' (line 151)
            v_6007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 51), 'v', False)
            # Processing the call keyword arguments (line 151)
            kwargs_6008 = {}
            # Getting the type of '_frz' (line 151)
            _frz_6006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 46), '_frz', False)
            # Calling _frz(args, kwargs) (line 151)
            _frz_call_result_6009 = invoke(stypy.reporting.localization.Localization(__file__, 151, 46), _frz_6006, *[v_6007], **kwargs_6008)
            
            # Obtaining the member '__getitem__' of a type (line 151)
            getitem___6010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 46), _frz_call_result_6009, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 151)
            subscript_call_result_6011 = invoke(stypy.reporting.localization.Localization(__file__, 151, 46), getitem___6010, int_6005)
            
            # Getting the type of 'dtype' (line 151)
            dtype_6012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 58), 'dtype', False)
            # Processing the call keyword arguments (line 151)
            kwargs_6013 = {}
            # Getting the type of 'array' (line 151)
            array_6004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 40), 'array', False)
            # Calling array(args, kwargs) (line 151)
            array_call_result_6014 = invoke(stypy.reporting.localization.Localization(__file__, 151, 40), array_6004, *[subscript_call_result_6011, dtype_6012], **kwargs_6013)
            
            # Applying the binary operator '%' (line 151)
            result_mod_6015 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 34), '%', fmt_6003, array_call_result_6014)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 151)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'stypy_return_type', result_mod_6015)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_4' in the type store
            # Getting the type of 'stypy_return_type' (line 151)
            stypy_return_type_6016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6016)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_4'
            return stypy_return_type_6016

        # Assigning a type to the variable '_stypy_temp_lambda_4' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), '_stypy_temp_lambda_4', _stypy_temp_lambda_4)
        # Getting the type of '_stypy_temp_lambda_4' (line 151)
        _stypy_temp_lambda_4_6017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 24), '_stypy_temp_lambda_4')
        str_6018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 24), 'str', 'numpy %s precision floating point number')
        # Getting the type of 'precname' (line 152)
        precname_6019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 69), 'precname', False)
        # Applying the binary operator '%' (line 152)
        result_mod_6020 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 24), '%', str_6018, precname_6019)
        
        # Processing the call keyword arguments (line 148)
        kwargs_6021 = {}
        # Getting the type of 'MachAr' (line 148)
        MachAr_5968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 17), 'MachAr', False)
        # Calling MachAr(args, kwargs) (line 148)
        MachAr_call_result_6022 = invoke(stypy.reporting.localization.Localization(__file__, 148, 17), MachAr_5968, *[_stypy_temp_lambda_1_5976, _stypy_temp_lambda_2_5989, _stypy_temp_lambda_3_6002, _stypy_temp_lambda_4_6017, result_mod_6020], **kwargs_6021)
        
        # Assigning a type to the variable 'machar' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'machar', MachAr_call_result_6022)
        
        
        # Obtaining an instance of the builtin type 'list' (line 154)
        list_6023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 154)
        # Adding element type (line 154)
        str_6024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'str', 'precision')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), list_6023, str_6024)
        # Adding element type (line 154)
        str_6025 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 34), 'str', 'iexp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), list_6023, str_6025)
        # Adding element type (line 154)
        str_6026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 21), 'str', 'maxexp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), list_6023, str_6026)
        # Adding element type (line 154)
        str_6027 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 31), 'str', 'minexp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), list_6023, str_6027)
        # Adding element type (line 154)
        str_6028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 41), 'str', 'negep')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), list_6023, str_6028)
        # Adding element type (line 154)
        str_6029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'str', 'machep')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 20), list_6023, str_6029)
        
        # Testing the type of a for loop iterable (line 154)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 154, 8), list_6023)
        # Getting the type of the for loop variable (line 154)
        for_loop_var_6030 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 154, 8), list_6023)
        # Assigning a type to the variable 'word' (line 154)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'word', for_loop_var_6030)
        # SSA begins for a for statement (line 154)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'self' (line 157)
        self_6032 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 20), 'self', False)
        # Getting the type of 'word' (line 157)
        word_6033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 26), 'word', False)
        
        # Call to getattr(...): (line 157)
        # Processing the call arguments (line 157)
        # Getting the type of 'machar' (line 157)
        machar_6035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 40), 'machar', False)
        # Getting the type of 'word' (line 157)
        word_6036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 48), 'word', False)
        # Processing the call keyword arguments (line 157)
        kwargs_6037 = {}
        # Getting the type of 'getattr' (line 157)
        getattr_6034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 32), 'getattr', False)
        # Calling getattr(args, kwargs) (line 157)
        getattr_call_result_6038 = invoke(stypy.reporting.localization.Localization(__file__, 157, 32), getattr_6034, *[machar_6035, word_6036], **kwargs_6037)
        
        # Processing the call keyword arguments (line 157)
        kwargs_6039 = {}
        # Getting the type of 'setattr' (line 157)
        setattr_6031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 157)
        setattr_call_result_6040 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), setattr_6031, *[self_6032, word_6033, getattr_call_result_6038], **kwargs_6039)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Obtaining an instance of the builtin type 'list' (line 158)
        list_6041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 158)
        # Adding element type (line 158)
        str_6042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 21), 'str', 'tiny')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 20), list_6041, str_6042)
        # Adding element type (line 158)
        str_6043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 29), 'str', 'resolution')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 20), list_6041, str_6043)
        # Adding element type (line 158)
        str_6044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 43), 'str', 'epsneg')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 158, 20), list_6041, str_6044)
        
        # Testing the type of a for loop iterable (line 158)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 158, 8), list_6041)
        # Getting the type of the for loop variable (line 158)
        for_loop_var_6045 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 158, 8), list_6041)
        # Assigning a type to the variable 'word' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'word', for_loop_var_6045)
        # SSA begins for a for statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'self' (line 159)
        self_6047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 20), 'self', False)
        # Getting the type of 'word' (line 159)
        word_6048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 26), 'word', False)
        
        # Obtaining the type of the subscript
        int_6049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 59), 'int')
        
        # Call to getattr(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'machar' (line 159)
        machar_6051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 40), 'machar', False)
        # Getting the type of 'word' (line 159)
        word_6052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 48), 'word', False)
        # Processing the call keyword arguments (line 159)
        kwargs_6053 = {}
        # Getting the type of 'getattr' (line 159)
        getattr_6050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 32), 'getattr', False)
        # Calling getattr(args, kwargs) (line 159)
        getattr_call_result_6054 = invoke(stypy.reporting.localization.Localization(__file__, 159, 32), getattr_6050, *[machar_6051, word_6052], **kwargs_6053)
        
        # Obtaining the member 'flat' of a type (line 159)
        flat_6055 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 32), getattr_call_result_6054, 'flat')
        # Obtaining the member '__getitem__' of a type (line 159)
        getitem___6056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 32), flat_6055, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 159)
        subscript_call_result_6057 = invoke(stypy.reporting.localization.Localization(__file__, 159, 32), getitem___6056, int_6049)
        
        # Processing the call keyword arguments (line 159)
        kwargs_6058 = {}
        # Getting the type of 'setattr' (line 159)
        setattr_6046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 159)
        setattr_call_result_6059 = invoke(stypy.reporting.localization.Localization(__file__, 159, 12), setattr_6046, *[self_6047, word_6048, subscript_call_result_6057], **kwargs_6058)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Attribute (line 160):
        
        # Obtaining the type of the subscript
        int_6060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 36), 'int')
        # Getting the type of 'machar' (line 160)
        machar_6061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 19), 'machar')
        # Obtaining the member 'huge' of a type (line 160)
        huge_6062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), machar_6061, 'huge')
        # Obtaining the member 'flat' of a type (line 160)
        flat_6063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), huge_6062, 'flat')
        # Obtaining the member '__getitem__' of a type (line 160)
        getitem___6064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 19), flat_6063, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 160)
        subscript_call_result_6065 = invoke(stypy.reporting.localization.Localization(__file__, 160, 19), getitem___6064, int_6060)
        
        # Getting the type of 'self' (line 160)
        self_6066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'self')
        # Setting the type of the member 'max' of a type (line 160)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), self_6066, 'max', subscript_call_result_6065)
        
        # Assigning a UnaryOp to a Attribute (line 161):
        
        # Getting the type of 'self' (line 161)
        self_6067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 20), 'self')
        # Obtaining the member 'max' of a type (line 161)
        max_6068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 20), self_6067, 'max')
        # Applying the 'usub' unary operator (line 161)
        result___neg___6069 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 19), 'usub', max_6068)
        
        # Getting the type of 'self' (line 161)
        self_6070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'self')
        # Setting the type of the member 'min' of a type (line 161)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), self_6070, 'min', result___neg___6069)
        
        # Assigning a Subscript to a Attribute (line 162):
        
        # Obtaining the type of the subscript
        int_6071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 35), 'int')
        # Getting the type of 'machar' (line 162)
        machar_6072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 19), 'machar')
        # Obtaining the member 'eps' of a type (line 162)
        eps_6073 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 19), machar_6072, 'eps')
        # Obtaining the member 'flat' of a type (line 162)
        flat_6074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 19), eps_6073, 'flat')
        # Obtaining the member '__getitem__' of a type (line 162)
        getitem___6075 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 19), flat_6074, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 162)
        subscript_call_result_6076 = invoke(stypy.reporting.localization.Localization(__file__, 162, 19), getitem___6075, int_6071)
        
        # Getting the type of 'self' (line 162)
        self_6077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'self')
        # Setting the type of the member 'eps' of a type (line 162)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), self_6077, 'eps', subscript_call_result_6076)
        
        # Assigning a Attribute to a Attribute (line 163):
        # Getting the type of 'machar' (line 163)
        machar_6078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 20), 'machar')
        # Obtaining the member 'iexp' of a type (line 163)
        iexp_6079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 20), machar_6078, 'iexp')
        # Getting the type of 'self' (line 163)
        self_6080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'self')
        # Setting the type of the member 'nexp' of a type (line 163)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 8), self_6080, 'nexp', iexp_6079)
        
        # Assigning a Attribute to a Attribute (line 164):
        # Getting the type of 'machar' (line 164)
        machar_6081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 21), 'machar')
        # Obtaining the member 'it' of a type (line 164)
        it_6082 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 21), machar_6081, 'it')
        # Getting the type of 'self' (line 164)
        self_6083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'self')
        # Setting the type of the member 'nmant' of a type (line 164)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 8), self_6083, 'nmant', it_6082)
        
        # Assigning a Name to a Attribute (line 165):
        # Getting the type of 'machar' (line 165)
        machar_6084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'machar')
        # Getting the type of 'self' (line 165)
        self_6085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 8), 'self')
        # Setting the type of the member 'machar' of a type (line 165)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 8), self_6085, 'machar', machar_6084)
        
        # Assigning a Call to a Attribute (line 166):
        
        # Call to strip(...): (line 166)
        # Processing the call keyword arguments (line 166)
        kwargs_6089 = {}
        # Getting the type of 'machar' (line 166)
        machar_6086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 25), 'machar', False)
        # Obtaining the member '_str_xmin' of a type (line 166)
        _str_xmin_6087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), machar_6086, '_str_xmin')
        # Obtaining the member 'strip' of a type (line 166)
        strip_6088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 25), _str_xmin_6087, 'strip')
        # Calling strip(args, kwargs) (line 166)
        strip_call_result_6090 = invoke(stypy.reporting.localization.Localization(__file__, 166, 25), strip_6088, *[], **kwargs_6089)
        
        # Getting the type of 'self' (line 166)
        self_6091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'self')
        # Setting the type of the member '_str_tiny' of a type (line 166)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 8), self_6091, '_str_tiny', strip_call_result_6090)
        
        # Assigning a Call to a Attribute (line 167):
        
        # Call to strip(...): (line 167)
        # Processing the call keyword arguments (line 167)
        kwargs_6095 = {}
        # Getting the type of 'machar' (line 167)
        machar_6092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 24), 'machar', False)
        # Obtaining the member '_str_xmax' of a type (line 167)
        _str_xmax_6093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 24), machar_6092, '_str_xmax')
        # Obtaining the member 'strip' of a type (line 167)
        strip_6094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 24), _str_xmax_6093, 'strip')
        # Calling strip(args, kwargs) (line 167)
        strip_call_result_6096 = invoke(stypy.reporting.localization.Localization(__file__, 167, 24), strip_6094, *[], **kwargs_6095)
        
        # Getting the type of 'self' (line 167)
        self_6097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 8), 'self')
        # Setting the type of the member '_str_max' of a type (line 167)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 8), self_6097, '_str_max', strip_call_result_6096)
        
        # Assigning a Call to a Attribute (line 168):
        
        # Call to strip(...): (line 168)
        # Processing the call keyword arguments (line 168)
        kwargs_6101 = {}
        # Getting the type of 'machar' (line 168)
        machar_6098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'machar', False)
        # Obtaining the member '_str_epsneg' of a type (line 168)
        _str_epsneg_6099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 27), machar_6098, '_str_epsneg')
        # Obtaining the member 'strip' of a type (line 168)
        strip_6100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 27), _str_epsneg_6099, 'strip')
        # Calling strip(args, kwargs) (line 168)
        strip_call_result_6102 = invoke(stypy.reporting.localization.Localization(__file__, 168, 27), strip_6100, *[], **kwargs_6101)
        
        # Getting the type of 'self' (line 168)
        self_6103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self')
        # Setting the type of the member '_str_epsneg' of a type (line 168)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_6103, '_str_epsneg', strip_call_result_6102)
        
        # Assigning a Call to a Attribute (line 169):
        
        # Call to strip(...): (line 169)
        # Processing the call keyword arguments (line 169)
        kwargs_6107 = {}
        # Getting the type of 'machar' (line 169)
        machar_6104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'machar', False)
        # Obtaining the member '_str_eps' of a type (line 169)
        _str_eps_6105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 24), machar_6104, '_str_eps')
        # Obtaining the member 'strip' of a type (line 169)
        strip_6106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 24), _str_eps_6105, 'strip')
        # Calling strip(args, kwargs) (line 169)
        strip_call_result_6108 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), strip_6106, *[], **kwargs_6107)
        
        # Getting the type of 'self' (line 169)
        self_6109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'self')
        # Setting the type of the member '_str_eps' of a type (line 169)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 8), self_6109, '_str_eps', strip_call_result_6108)
        
        # Assigning a Call to a Attribute (line 170):
        
        # Call to strip(...): (line 170)
        # Processing the call keyword arguments (line 170)
        kwargs_6113 = {}
        # Getting the type of 'machar' (line 170)
        machar_6110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 31), 'machar', False)
        # Obtaining the member '_str_resolution' of a type (line 170)
        _str_resolution_6111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 31), machar_6110, '_str_resolution')
        # Obtaining the member 'strip' of a type (line 170)
        strip_6112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 31), _str_resolution_6111, 'strip')
        # Calling strip(args, kwargs) (line 170)
        strip_call_result_6114 = invoke(stypy.reporting.localization.Localization(__file__, 170, 31), strip_6112, *[], **kwargs_6113)
        
        # Getting the type of 'self' (line 170)
        self_6115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'self')
        # Setting the type of the member '_str_resolution' of a type (line 170)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), self_6115, '_str_resolution', strip_call_result_6114)
        # Getting the type of 'self' (line 171)
        self_6116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 8), 'stypy_return_type', self_6116)
        
        # ################# End of '_init(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_init' in the type store
        # Getting the type of 'stypy_return_type' (line 127)
        stypy_return_type_6117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_init'
        return stypy_return_type_6117


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 173, 4, False)
        # Assigning a type to the variable 'self' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        finfo.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        finfo.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        finfo.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        finfo.stypy__str__.__dict__.__setitem__('stypy_function_name', 'finfo.__str__')
        finfo.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        finfo.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        finfo.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        finfo.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        finfo.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        finfo.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        finfo.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'finfo.__str__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Str to a Name (line 174):
        str_6118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 12), 'str', 'Machine parameters for %(dtype)s\n---------------------------------------------------------------\nprecision=%(precision)3s   resolution= %(_str_resolution)s\nmachep=%(machep)6s   eps=        %(_str_eps)s\nnegep =%(negep)6s   epsneg=     %(_str_epsneg)s\nminexp=%(minexp)6s   tiny=       %(_str_tiny)s\nmaxexp=%(maxexp)6s   max=        %(_str_max)s\nnexp  =%(nexp)6s   min=        -max\n---------------------------------------------------------------\n')
        # Assigning a type to the variable 'fmt' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 8), 'fmt', str_6118)
        # Getting the type of 'fmt' (line 185)
        fmt_6119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 15), 'fmt')
        # Getting the type of 'self' (line 185)
        self_6120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 21), 'self')
        # Obtaining the member '__dict__' of a type (line 185)
        dict___6121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 21), self_6120, '__dict__')
        # Applying the binary operator '%' (line 185)
        result_mod_6122 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 15), '%', fmt_6119, dict___6121)
        
        # Assigning a type to the variable 'stypy_return_type' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', result_mod_6122)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 173)
        stypy_return_type_6123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6123)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_6123


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 187, 4, False)
        # Assigning a type to the variable 'self' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        finfo.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        finfo.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        finfo.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        finfo.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'finfo.__repr__')
        finfo.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        finfo.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        finfo.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        finfo.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        finfo.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        finfo.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        finfo.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'finfo.__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        
        # Assigning a Attribute to a Name (line 188):
        # Getting the type of 'self' (line 188)
        self_6124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 12), 'self')
        # Obtaining the member '__class__' of a type (line 188)
        class___6125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), self_6124, '__class__')
        # Obtaining the member '__name__' of a type (line 188)
        name___6126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 12), class___6125, '__name__')
        # Assigning a type to the variable 'c' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'c', name___6126)
        
        # Assigning a Call to a Name (line 189):
        
        # Call to copy(...): (line 189)
        # Processing the call keyword arguments (line 189)
        kwargs_6130 = {}
        # Getting the type of 'self' (line 189)
        self_6127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'self', False)
        # Obtaining the member '__dict__' of a type (line 189)
        dict___6128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), self_6127, '__dict__')
        # Obtaining the member 'copy' of a type (line 189)
        copy_6129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), dict___6128, 'copy')
        # Calling copy(args, kwargs) (line 189)
        copy_call_result_6131 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), copy_6129, *[], **kwargs_6130)
        
        # Assigning a type to the variable 'd' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'd', copy_call_result_6131)
        
        # Assigning a Name to a Subscript (line 190):
        # Getting the type of 'c' (line 190)
        c_6132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 21), 'c')
        # Getting the type of 'd' (line 190)
        d_6133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 8), 'd')
        str_6134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 10), 'str', 'klass')
        # Storing an element on a container (line 190)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 190, 8), d_6133, (str_6134, c_6132))
        str_6135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 17), 'str', '%(klass)s(resolution=%(resolution)s, min=-%(_str_max)s, max=%(_str_max)s, dtype=%(dtype)s)')
        # Getting the type of 'd' (line 192)
        d_6136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 58), 'd')
        # Applying the binary operator '%' (line 191)
        result_mod_6137 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 16), '%', str_6135, d_6136)
        
        # Assigning a type to the variable 'stypy_return_type' (line 191)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'stypy_return_type', result_mod_6137)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 187)
        stypy_return_type_6138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6138)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_6138


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 25, 0, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'finfo.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'finfo' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'finfo', finfo)

# Assigning a Dict to a Name (line 92):

# Obtaining an instance of the builtin type 'dict' (line 92)
dict_6139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 92)

# Getting the type of 'finfo'
finfo_6140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'finfo')
# Setting the type of the member '_finfo_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), finfo_6140, '_finfo_cache', dict_6139)
# Declaration of the 'iinfo' class

class iinfo(object, ):
    str_6141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, (-1)), 'str', '\n    iinfo(type)\n\n    Machine limits for integer types.\n\n    Attributes\n    ----------\n    min : int\n        The smallest integer expressible by the type.\n    max : int\n        The largest integer expressible by the type.\n\n    Parameters\n    ----------\n    int_type : integer type, dtype, or instance\n        The kind of integer data type to get information about.\n\n    See Also\n    --------\n    finfo : The equivalent for floating point data types.\n\n    Examples\n    --------\n    With types:\n\n    >>> ii16 = np.iinfo(np.int16)\n    >>> ii16.min\n    -32768\n    >>> ii16.max\n    32767\n    >>> ii32 = np.iinfo(np.int32)\n    >>> ii32.min\n    -2147483648\n    >>> ii32.max\n    2147483647\n\n    With instances:\n\n    >>> ii32 = np.iinfo(np.int32(10))\n    >>> ii32.min\n    -2147483648\n    >>> ii32.max\n    2147483647\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 245, 4, False)
        # Assigning a type to the variable 'self' (line 246)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'iinfo.__init__', ['int_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['int_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        
        # SSA begins for try-except statement (line 246)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Attribute (line 247):
        
        # Call to dtype(...): (line 247)
        # Processing the call arguments (line 247)
        # Getting the type of 'int_type' (line 247)
        int_type_6144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 39), 'int_type', False)
        # Processing the call keyword arguments (line 247)
        kwargs_6145 = {}
        # Getting the type of 'numeric' (line 247)
        numeric_6142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 25), 'numeric', False)
        # Obtaining the member 'dtype' of a type (line 247)
        dtype_6143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 25), numeric_6142, 'dtype')
        # Calling dtype(args, kwargs) (line 247)
        dtype_call_result_6146 = invoke(stypy.reporting.localization.Localization(__file__, 247, 25), dtype_6143, *[int_type_6144], **kwargs_6145)
        
        # Getting the type of 'self' (line 247)
        self_6147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 247, 12), 'self')
        # Setting the type of the member 'dtype' of a type (line 247)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 247, 12), self_6147, 'dtype', dtype_call_result_6146)
        # SSA branch for the except part of a try statement (line 246)
        # SSA branch for the except 'TypeError' branch of a try statement (line 246)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Attribute (line 249):
        
        # Call to dtype(...): (line 249)
        # Processing the call arguments (line 249)
        
        # Call to type(...): (line 249)
        # Processing the call arguments (line 249)
        # Getting the type of 'int_type' (line 249)
        int_type_6151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 44), 'int_type', False)
        # Processing the call keyword arguments (line 249)
        kwargs_6152 = {}
        # Getting the type of 'type' (line 249)
        type_6150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 39), 'type', False)
        # Calling type(args, kwargs) (line 249)
        type_call_result_6153 = invoke(stypy.reporting.localization.Localization(__file__, 249, 39), type_6150, *[int_type_6151], **kwargs_6152)
        
        # Processing the call keyword arguments (line 249)
        kwargs_6154 = {}
        # Getting the type of 'numeric' (line 249)
        numeric_6148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 25), 'numeric', False)
        # Obtaining the member 'dtype' of a type (line 249)
        dtype_6149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 25), numeric_6148, 'dtype')
        # Calling dtype(args, kwargs) (line 249)
        dtype_call_result_6155 = invoke(stypy.reporting.localization.Localization(__file__, 249, 25), dtype_6149, *[type_call_result_6153], **kwargs_6154)
        
        # Getting the type of 'self' (line 249)
        self_6156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 249, 12), 'self')
        # Setting the type of the member 'dtype' of a type (line 249)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 249, 12), self_6156, 'dtype', dtype_call_result_6155)
        # SSA join for try-except statement (line 246)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Attribute (line 250):
        # Getting the type of 'self' (line 250)
        self_6157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 20), 'self')
        # Obtaining the member 'dtype' of a type (line 250)
        dtype_6158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 20), self_6157, 'dtype')
        # Obtaining the member 'kind' of a type (line 250)
        kind_6159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 20), dtype_6158, 'kind')
        # Getting the type of 'self' (line 250)
        self_6160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 8), 'self')
        # Setting the type of the member 'kind' of a type (line 250)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 250, 8), self_6160, 'kind', kind_6159)
        
        # Assigning a BinOp to a Attribute (line 251):
        # Getting the type of 'self' (line 251)
        self_6161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 20), 'self')
        # Obtaining the member 'dtype' of a type (line 251)
        dtype_6162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 20), self_6161, 'dtype')
        # Obtaining the member 'itemsize' of a type (line 251)
        itemsize_6163 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 20), dtype_6162, 'itemsize')
        int_6164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 42), 'int')
        # Applying the binary operator '*' (line 251)
        result_mul_6165 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 20), '*', itemsize_6163, int_6164)
        
        # Getting the type of 'self' (line 251)
        self_6166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'self')
        # Setting the type of the member 'bits' of a type (line 251)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 251, 8), self_6166, 'bits', result_mul_6165)
        
        # Assigning a BinOp to a Attribute (line 252):
        str_6167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 19), 'str', '%s%d')
        
        # Obtaining an instance of the builtin type 'tuple' (line 252)
        tuple_6168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 252, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 252)
        # Adding element type (line 252)
        # Getting the type of 'self' (line 252)
        self_6169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 29), 'self')
        # Obtaining the member 'kind' of a type (line 252)
        kind_6170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 29), self_6169, 'kind')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 29), tuple_6168, kind_6170)
        # Adding element type (line 252)
        # Getting the type of 'self' (line 252)
        self_6171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 40), 'self')
        # Obtaining the member 'bits' of a type (line 252)
        bits_6172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 40), self_6171, 'bits')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 252, 29), tuple_6168, bits_6172)
        
        # Applying the binary operator '%' (line 252)
        result_mod_6173 = python_operator(stypy.reporting.localization.Localization(__file__, 252, 19), '%', str_6167, tuple_6168)
        
        # Getting the type of 'self' (line 252)
        self_6174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 8), 'self')
        # Setting the type of the member 'key' of a type (line 252)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 252, 8), self_6174, 'key', result_mod_6173)
        
        
        # Getting the type of 'self' (line 253)
        self_6175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'self')
        # Obtaining the member 'kind' of a type (line 253)
        kind_6176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 253, 11), self_6175, 'kind')
        str_6177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 253, 28), 'str', 'iu')
        # Applying the binary operator 'notin' (line 253)
        result_contains_6178 = python_operator(stypy.reporting.localization.Localization(__file__, 253, 11), 'notin', kind_6176, str_6177)
        
        # Testing the type of an if condition (line 253)
        if_condition_6179 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 253, 8), result_contains_6178)
        # Assigning a type to the variable 'if_condition_6179' (line 253)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 8), 'if_condition_6179', if_condition_6179)
        # SSA begins for if statement (line 253)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 254)
        # Processing the call arguments (line 254)
        str_6181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 254, 29), 'str', 'Invalid integer data type.')
        # Processing the call keyword arguments (line 254)
        kwargs_6182 = {}
        # Getting the type of 'ValueError' (line 254)
        ValueError_6180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 254)
        ValueError_call_result_6183 = invoke(stypy.reporting.localization.Localization(__file__, 254, 18), ValueError_6180, *[str_6181], **kwargs_6182)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 254, 12), ValueError_call_result_6183, 'raise parameter', BaseException)
        # SSA join for if statement (line 253)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def min(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'min'
        module_type_store = module_type_store.open_function_context('min', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        iinfo.min.__dict__.__setitem__('stypy_localization', localization)
        iinfo.min.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        iinfo.min.__dict__.__setitem__('stypy_type_store', module_type_store)
        iinfo.min.__dict__.__setitem__('stypy_function_name', 'iinfo.min')
        iinfo.min.__dict__.__setitem__('stypy_param_names_list', [])
        iinfo.min.__dict__.__setitem__('stypy_varargs_param_name', None)
        iinfo.min.__dict__.__setitem__('stypy_kwargs_param_name', None)
        iinfo.min.__dict__.__setitem__('stypy_call_defaults', defaults)
        iinfo.min.__dict__.__setitem__('stypy_call_varargs', varargs)
        iinfo.min.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        iinfo.min.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'iinfo.min', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'min', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'min(...)' code ##################

        str_6184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 8), 'str', 'Minimum value of given dtype.')
        
        
        # Getting the type of 'self' (line 258)
        self_6185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 258, 11), 'self')
        # Obtaining the member 'kind' of a type (line 258)
        kind_6186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 258, 11), self_6185, 'kind')
        str_6187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 258, 24), 'str', 'u')
        # Applying the binary operator '==' (line 258)
        result_eq_6188 = python_operator(stypy.reporting.localization.Localization(__file__, 258, 11), '==', kind_6186, str_6187)
        
        # Testing the type of an if condition (line 258)
        if_condition_6189 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 258, 8), result_eq_6188)
        # Assigning a type to the variable 'if_condition_6189' (line 258)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 258, 8), 'if_condition_6189', if_condition_6189)
        # SSA begins for if statement (line 258)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        int_6190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 259, 19), 'int')
        # Assigning a type to the variable 'stypy_return_type' (line 259)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 259, 12), 'stypy_return_type', int_6190)
        # SSA branch for the else part of an if statement (line 258)
        module_type_store.open_ssa_branch('else')
        
        
        # SSA begins for try-except statement (line 261)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 262):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 262)
        self_6191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 38), 'self')
        # Obtaining the member 'key' of a type (line 262)
        key_6192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 38), self_6191, 'key')
        # Getting the type of 'iinfo' (line 262)
        iinfo_6193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 262, 22), 'iinfo')
        # Obtaining the member '_min_vals' of a type (line 262)
        _min_vals_6194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 22), iinfo_6193, '_min_vals')
        # Obtaining the member '__getitem__' of a type (line 262)
        getitem___6195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 262, 22), _min_vals_6194, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 262)
        subscript_call_result_6196 = invoke(stypy.reporting.localization.Localization(__file__, 262, 22), getitem___6195, key_6192)
        
        # Assigning a type to the variable 'val' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 16), 'val', subscript_call_result_6196)
        # SSA branch for the except part of a try statement (line 261)
        # SSA branch for the except 'KeyError' branch of a try statement (line 261)
        module_type_store.open_ssa_branch('except')
        
        # Assigning a Call to a Name (line 264):
        
        # Call to int(...): (line 264)
        # Processing the call arguments (line 264)
        
        int_6198 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 28), 'int')
        # Getting the type of 'self' (line 264)
        self_6199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 34), 'self', False)
        # Obtaining the member 'bits' of a type (line 264)
        bits_6200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 34), self_6199, 'bits')
        int_6201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 264, 44), 'int')
        # Applying the binary operator '-' (line 264)
        result_sub_6202 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 34), '-', bits_6200, int_6201)
        
        # Applying the binary operator '<<' (line 264)
        result_lshift_6203 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 28), '<<', int_6198, result_sub_6202)
        
        # Applying the 'usub' unary operator (line 264)
        result___neg___6204 = python_operator(stypy.reporting.localization.Localization(__file__, 264, 26), 'usub', result_lshift_6203)
        
        # Processing the call keyword arguments (line 264)
        kwargs_6205 = {}
        # Getting the type of 'int' (line 264)
        int_6197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 22), 'int', False)
        # Calling int(args, kwargs) (line 264)
        int_call_result_6206 = invoke(stypy.reporting.localization.Localization(__file__, 264, 22), int_6197, *[result___neg___6204], **kwargs_6205)
        
        # Assigning a type to the variable 'val' (line 264)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 264, 16), 'val', int_call_result_6206)
        
        # Assigning a Name to a Subscript (line 265):
        # Getting the type of 'val' (line 265)
        val_6207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 44), 'val')
        # Getting the type of 'iinfo' (line 265)
        iinfo_6208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 16), 'iinfo')
        # Obtaining the member '_min_vals' of a type (line 265)
        _min_vals_6209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 16), iinfo_6208, '_min_vals')
        # Getting the type of 'self' (line 265)
        self_6210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 265, 32), 'self')
        # Obtaining the member 'key' of a type (line 265)
        key_6211 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 265, 32), self_6210, 'key')
        # Storing an element on a container (line 265)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 265, 16), _min_vals_6209, (key_6211, val_6207))
        # SSA join for try-except statement (line 261)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'val' (line 266)
        val_6212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 19), 'val')
        # Assigning a type to the variable 'stypy_return_type' (line 266)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'stypy_return_type', val_6212)
        # SSA join for if statement (line 258)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'min(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'min' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_6213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'min'
        return stypy_return_type_6213


    @norecursion
    def max(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'max'
        module_type_store = module_type_store.open_function_context('max', 270, 4, False)
        # Assigning a type to the variable 'self' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        iinfo.max.__dict__.__setitem__('stypy_localization', localization)
        iinfo.max.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        iinfo.max.__dict__.__setitem__('stypy_type_store', module_type_store)
        iinfo.max.__dict__.__setitem__('stypy_function_name', 'iinfo.max')
        iinfo.max.__dict__.__setitem__('stypy_param_names_list', [])
        iinfo.max.__dict__.__setitem__('stypy_varargs_param_name', None)
        iinfo.max.__dict__.__setitem__('stypy_kwargs_param_name', None)
        iinfo.max.__dict__.__setitem__('stypy_call_defaults', defaults)
        iinfo.max.__dict__.__setitem__('stypy_call_varargs', varargs)
        iinfo.max.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        iinfo.max.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'iinfo.max', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'max', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'max(...)' code ##################

        str_6214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 271, 8), 'str', 'Maximum value of given dtype.')
        
        
        # SSA begins for try-except statement (line 272)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 273):
        
        # Obtaining the type of the subscript
        # Getting the type of 'self' (line 273)
        self_6215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 34), 'self')
        # Obtaining the member 'key' of a type (line 273)
        key_6216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 34), self_6215, 'key')
        # Getting the type of 'iinfo' (line 273)
        iinfo_6217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 273, 18), 'iinfo')
        # Obtaining the member '_max_vals' of a type (line 273)
        _max_vals_6218 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 18), iinfo_6217, '_max_vals')
        # Obtaining the member '__getitem__' of a type (line 273)
        getitem___6219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 273, 18), _max_vals_6218, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 273)
        subscript_call_result_6220 = invoke(stypy.reporting.localization.Localization(__file__, 273, 18), getitem___6219, key_6216)
        
        # Assigning a type to the variable 'val' (line 273)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 273, 12), 'val', subscript_call_result_6220)
        # SSA branch for the except part of a try statement (line 272)
        # SSA branch for the except 'KeyError' branch of a try statement (line 272)
        module_type_store.open_ssa_branch('except')
        
        
        # Getting the type of 'self' (line 275)
        self_6221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 275, 15), 'self')
        # Obtaining the member 'kind' of a type (line 275)
        kind_6222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 275, 15), self_6221, 'kind')
        str_6223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 275, 28), 'str', 'u')
        # Applying the binary operator '==' (line 275)
        result_eq_6224 = python_operator(stypy.reporting.localization.Localization(__file__, 275, 15), '==', kind_6222, str_6223)
        
        # Testing the type of an if condition (line 275)
        if_condition_6225 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 275, 12), result_eq_6224)
        # Assigning a type to the variable 'if_condition_6225' (line 275)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 275, 12), 'if_condition_6225', if_condition_6225)
        # SSA begins for if statement (line 275)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 276):
        
        # Call to int(...): (line 276)
        # Processing the call arguments (line 276)
        int_6227 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 27), 'int')
        # Getting the type of 'self' (line 276)
        self_6228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 32), 'self', False)
        # Obtaining the member 'bits' of a type (line 276)
        bits_6229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 276, 32), self_6228, 'bits')
        # Applying the binary operator '<<' (line 276)
        result_lshift_6230 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 27), '<<', int_6227, bits_6229)
        
        int_6231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 276, 45), 'int')
        # Applying the binary operator '-' (line 276)
        result_sub_6232 = python_operator(stypy.reporting.localization.Localization(__file__, 276, 26), '-', result_lshift_6230, int_6231)
        
        # Processing the call keyword arguments (line 276)
        kwargs_6233 = {}
        # Getting the type of 'int' (line 276)
        int_6226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 276, 22), 'int', False)
        # Calling int(args, kwargs) (line 276)
        int_call_result_6234 = invoke(stypy.reporting.localization.Localization(__file__, 276, 22), int_6226, *[result_sub_6232], **kwargs_6233)
        
        # Assigning a type to the variable 'val' (line 276)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 276, 16), 'val', int_call_result_6234)
        # SSA branch for the else part of an if statement (line 275)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 278):
        
        # Call to int(...): (line 278)
        # Processing the call arguments (line 278)
        int_6236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 27), 'int')
        # Getting the type of 'self' (line 278)
        self_6237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 33), 'self', False)
        # Obtaining the member 'bits' of a type (line 278)
        bits_6238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 278, 33), self_6237, 'bits')
        int_6239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 43), 'int')
        # Applying the binary operator '-' (line 278)
        result_sub_6240 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 33), '-', bits_6238, int_6239)
        
        # Applying the binary operator '<<' (line 278)
        result_lshift_6241 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 27), '<<', int_6236, result_sub_6240)
        
        int_6242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 49), 'int')
        # Applying the binary operator '-' (line 278)
        result_sub_6243 = python_operator(stypy.reporting.localization.Localization(__file__, 278, 26), '-', result_lshift_6241, int_6242)
        
        # Processing the call keyword arguments (line 278)
        kwargs_6244 = {}
        # Getting the type of 'int' (line 278)
        int_6235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 22), 'int', False)
        # Calling int(args, kwargs) (line 278)
        int_call_result_6245 = invoke(stypy.reporting.localization.Localization(__file__, 278, 22), int_6235, *[result_sub_6243], **kwargs_6244)
        
        # Assigning a type to the variable 'val' (line 278)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 16), 'val', int_call_result_6245)
        # SSA join for if statement (line 275)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Subscript (line 279):
        # Getting the type of 'val' (line 279)
        val_6246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 40), 'val')
        # Getting the type of 'iinfo' (line 279)
        iinfo_6247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 12), 'iinfo')
        # Obtaining the member '_max_vals' of a type (line 279)
        _max_vals_6248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 12), iinfo_6247, '_max_vals')
        # Getting the type of 'self' (line 279)
        self_6249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 279, 28), 'self')
        # Obtaining the member 'key' of a type (line 279)
        key_6250 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 279, 28), self_6249, 'key')
        # Storing an element on a container (line 279)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 279, 12), _max_vals_6248, (key_6250, val_6246))
        # SSA join for try-except statement (line 272)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'val' (line 280)
        val_6251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 15), 'val')
        # Assigning a type to the variable 'stypy_return_type' (line 280)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 8), 'stypy_return_type', val_6251)
        
        # ################# End of 'max(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'max' in the type store
        # Getting the type of 'stypy_return_type' (line 270)
        stypy_return_type_6252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6252)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'max'
        return stypy_return_type_6252


    @norecursion
    def stypy__str__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__str__'
        module_type_store = module_type_store.open_function_context('__str__', 284, 4, False)
        # Assigning a type to the variable 'self' (line 285)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        iinfo.stypy__str__.__dict__.__setitem__('stypy_localization', localization)
        iinfo.stypy__str__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        iinfo.stypy__str__.__dict__.__setitem__('stypy_type_store', module_type_store)
        iinfo.stypy__str__.__dict__.__setitem__('stypy_function_name', 'iinfo.__str__')
        iinfo.stypy__str__.__dict__.__setitem__('stypy_param_names_list', [])
        iinfo.stypy__str__.__dict__.__setitem__('stypy_varargs_param_name', None)
        iinfo.stypy__str__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        iinfo.stypy__str__.__dict__.__setitem__('stypy_call_defaults', defaults)
        iinfo.stypy__str__.__dict__.__setitem__('stypy_call_varargs', varargs)
        iinfo.stypy__str__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        iinfo.stypy__str__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'iinfo.__str__', [], None, None, defaults, varargs, kwargs)

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

        str_6253 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 8), 'str', 'String representation.')
        
        # Assigning a Str to a Name (line 286):
        str_6254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 287, 12), 'str', 'Machine parameters for %(dtype)s\n---------------------------------------------------------------\nmin = %(min)s\nmax = %(max)s\n---------------------------------------------------------------\n')
        # Assigning a type to the variable 'fmt' (line 286)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 286, 8), 'fmt', str_6254)
        # Getting the type of 'fmt' (line 293)
        fmt_6255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 15), 'fmt')
        
        # Obtaining an instance of the builtin type 'dict' (line 293)
        dict_6256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 21), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 293)
        # Adding element type (key, value) (line 293)
        str_6257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 22), 'str', 'dtype')
        # Getting the type of 'self' (line 293)
        self_6258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 31), 'self')
        # Obtaining the member 'dtype' of a type (line 293)
        dtype_6259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 31), self_6258, 'dtype')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 21), dict_6256, (str_6257, dtype_6259))
        # Adding element type (key, value) (line 293)
        str_6260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 43), 'str', 'min')
        # Getting the type of 'self' (line 293)
        self_6261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 50), 'self')
        # Obtaining the member 'min' of a type (line 293)
        min_6262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 50), self_6261, 'min')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 21), dict_6256, (str_6260, min_6262))
        # Adding element type (key, value) (line 293)
        str_6263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 60), 'str', 'max')
        # Getting the type of 'self' (line 293)
        self_6264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 67), 'self')
        # Obtaining the member 'max' of a type (line 293)
        max_6265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 293, 67), self_6264, 'max')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 21), dict_6256, (str_6263, max_6265))
        
        # Applying the binary operator '%' (line 293)
        result_mod_6266 = python_operator(stypy.reporting.localization.Localization(__file__, 293, 15), '%', fmt_6255, dict_6256)
        
        # Assigning a type to the variable 'stypy_return_type' (line 293)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 293, 8), 'stypy_return_type', result_mod_6266)
        
        # ################# End of '__str__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__str__' in the type store
        # Getting the type of 'stypy_return_type' (line 284)
        stypy_return_type_6267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 284, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6267)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__str__'
        return stypy_return_type_6267


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 295, 4, False)
        # Assigning a type to the variable 'self' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'iinfo.__repr__')
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        iinfo.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'iinfo.__repr__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__repr__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__repr__(...)' code ##################

        str_6268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 15), 'str', '%s(min=%s, max=%s, dtype=%s)')
        
        # Obtaining an instance of the builtin type 'tuple' (line 296)
        tuple_6269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 296, 49), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 296)
        # Adding element type (line 296)
        # Getting the type of 'self' (line 296)
        self_6270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 296, 49), 'self')
        # Obtaining the member '__class__' of a type (line 296)
        class___6271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 49), self_6270, '__class__')
        # Obtaining the member '__name__' of a type (line 296)
        name___6272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 296, 49), class___6271, '__name__')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 49), tuple_6269, name___6272)
        # Adding element type (line 296)
        # Getting the type of 'self' (line 297)
        self_6273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 36), 'self')
        # Obtaining the member 'min' of a type (line 297)
        min_6274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 36), self_6273, 'min')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 49), tuple_6269, min_6274)
        # Adding element type (line 296)
        # Getting the type of 'self' (line 297)
        self_6275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 46), 'self')
        # Obtaining the member 'max' of a type (line 297)
        max_6276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 46), self_6275, 'max')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 49), tuple_6269, max_6276)
        # Adding element type (line 296)
        # Getting the type of 'self' (line 297)
        self_6277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 297, 56), 'self')
        # Obtaining the member 'dtype' of a type (line 297)
        dtype_6278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 297, 56), self_6277, 'dtype')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 296, 49), tuple_6269, dtype_6278)
        
        # Applying the binary operator '%' (line 296)
        result_mod_6279 = python_operator(stypy.reporting.localization.Localization(__file__, 296, 15), '%', str_6268, tuple_6269)
        
        # Assigning a type to the variable 'stypy_return_type' (line 296)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 296, 8), 'stypy_return_type', result_mod_6279)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 295)
        stypy_return_type_6280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 295, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_6280


# Assigning a type to the variable 'iinfo' (line 195)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 0), 'iinfo', iinfo)

# Assigning a Dict to a Name (line 242):

# Obtaining an instance of the builtin type 'dict' (line 242)
dict_6281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 242, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 242)

# Getting the type of 'iinfo'
iinfo_6282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'iinfo')
# Setting the type of the member '_min_vals' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), iinfo_6282, '_min_vals', dict_6281)

# Assigning a Dict to a Name (line 243):

# Obtaining an instance of the builtin type 'dict' (line 243)
dict_6283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 243)

# Getting the type of 'iinfo'
iinfo_6284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'iinfo')
# Setting the type of the member '_max_vals' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), iinfo_6284, '_max_vals', dict_6283)

# Assigning a Call to a Name (line 268):

# Call to property(...): (line 268)
# Processing the call arguments (line 268)
# Getting the type of 'iinfo'
iinfo_6286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'iinfo', False)
# Obtaining the member 'min' of a type
min_6287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), iinfo_6286, 'min')
# Processing the call keyword arguments (line 268)
kwargs_6288 = {}
# Getting the type of 'property' (line 268)
property_6285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 268, 10), 'property', False)
# Calling property(args, kwargs) (line 268)
property_call_result_6289 = invoke(stypy.reporting.localization.Localization(__file__, 268, 10), property_6285, *[min_6287], **kwargs_6288)

# Getting the type of 'iinfo'
iinfo_6290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'iinfo')
# Setting the type of the member 'min' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), iinfo_6290, 'min', property_call_result_6289)

# Assigning a Call to a Name (line 282):

# Call to property(...): (line 282)
# Processing the call arguments (line 282)
# Getting the type of 'iinfo'
iinfo_6292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'iinfo', False)
# Obtaining the member 'max' of a type
max_6293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), iinfo_6292, 'max')
# Processing the call keyword arguments (line 282)
kwargs_6294 = {}
# Getting the type of 'property' (line 282)
property_6291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 10), 'property', False)
# Calling property(args, kwargs) (line 282)
property_call_result_6295 = invoke(stypy.reporting.localization.Localization(__file__, 282, 10), property_6291, *[max_6293], **kwargs_6294)

# Getting the type of 'iinfo'
iinfo_6296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'iinfo')
# Setting the type of the member 'max' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), iinfo_6296, 'max', property_call_result_6295)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 300):
    
    # Call to finfo(...): (line 300)
    # Processing the call arguments (line 300)
    # Getting the type of 'ntypes' (line 300)
    ntypes_6298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 14), 'ntypes', False)
    # Obtaining the member 'single' of a type (line 300)
    single_6299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 300, 14), ntypes_6298, 'single')
    # Processing the call keyword arguments (line 300)
    kwargs_6300 = {}
    # Getting the type of 'finfo' (line 300)
    finfo_6297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 300, 8), 'finfo', False)
    # Calling finfo(args, kwargs) (line 300)
    finfo_call_result_6301 = invoke(stypy.reporting.localization.Localization(__file__, 300, 8), finfo_6297, *[single_6299], **kwargs_6300)
    
    # Assigning a type to the variable 'f' (line 300)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 300, 4), 'f', finfo_call_result_6301)
    
    # Call to print(...): (line 301)
    # Processing the call arguments (line 301)
    str_6303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 301, 10), 'str', 'single epsilon:')
    # Getting the type of 'f' (line 301)
    f_6304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 29), 'f', False)
    # Obtaining the member 'eps' of a type (line 301)
    eps_6305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 301, 29), f_6304, 'eps')
    # Processing the call keyword arguments (line 301)
    kwargs_6306 = {}
    # Getting the type of 'print' (line 301)
    print_6302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 301, 4), 'print', False)
    # Calling print(args, kwargs) (line 301)
    print_call_result_6307 = invoke(stypy.reporting.localization.Localization(__file__, 301, 4), print_6302, *[str_6303, eps_6305], **kwargs_6306)
    
    
    # Call to print(...): (line 302)
    # Processing the call arguments (line 302)
    str_6309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 302, 10), 'str', 'single tiny:')
    # Getting the type of 'f' (line 302)
    f_6310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 26), 'f', False)
    # Obtaining the member 'tiny' of a type (line 302)
    tiny_6311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 302, 26), f_6310, 'tiny')
    # Processing the call keyword arguments (line 302)
    kwargs_6312 = {}
    # Getting the type of 'print' (line 302)
    print_6308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 302, 4), 'print', False)
    # Calling print(args, kwargs) (line 302)
    print_call_result_6313 = invoke(stypy.reporting.localization.Localization(__file__, 302, 4), print_6308, *[str_6309, tiny_6311], **kwargs_6312)
    
    
    # Assigning a Call to a Name (line 303):
    
    # Call to finfo(...): (line 303)
    # Processing the call arguments (line 303)
    # Getting the type of 'ntypes' (line 303)
    ntypes_6315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 14), 'ntypes', False)
    # Obtaining the member 'float' of a type (line 303)
    float_6316 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 303, 14), ntypes_6315, 'float')
    # Processing the call keyword arguments (line 303)
    kwargs_6317 = {}
    # Getting the type of 'finfo' (line 303)
    finfo_6314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 303, 8), 'finfo', False)
    # Calling finfo(args, kwargs) (line 303)
    finfo_call_result_6318 = invoke(stypy.reporting.localization.Localization(__file__, 303, 8), finfo_6314, *[float_6316], **kwargs_6317)
    
    # Assigning a type to the variable 'f' (line 303)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 303, 4), 'f', finfo_call_result_6318)
    
    # Call to print(...): (line 304)
    # Processing the call arguments (line 304)
    str_6320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 304, 10), 'str', 'float epsilon:')
    # Getting the type of 'f' (line 304)
    f_6321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 28), 'f', False)
    # Obtaining the member 'eps' of a type (line 304)
    eps_6322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 304, 28), f_6321, 'eps')
    # Processing the call keyword arguments (line 304)
    kwargs_6323 = {}
    # Getting the type of 'print' (line 304)
    print_6319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 304, 4), 'print', False)
    # Calling print(args, kwargs) (line 304)
    print_call_result_6324 = invoke(stypy.reporting.localization.Localization(__file__, 304, 4), print_6319, *[str_6320, eps_6322], **kwargs_6323)
    
    
    # Call to print(...): (line 305)
    # Processing the call arguments (line 305)
    str_6326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 305, 10), 'str', 'float tiny:')
    # Getting the type of 'f' (line 305)
    f_6327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 25), 'f', False)
    # Obtaining the member 'tiny' of a type (line 305)
    tiny_6328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 305, 25), f_6327, 'tiny')
    # Processing the call keyword arguments (line 305)
    kwargs_6329 = {}
    # Getting the type of 'print' (line 305)
    print_6325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 305, 4), 'print', False)
    # Calling print(args, kwargs) (line 305)
    print_call_result_6330 = invoke(stypy.reporting.localization.Localization(__file__, 305, 4), print_6325, *[str_6326, tiny_6328], **kwargs_6329)
    
    
    # Assigning a Call to a Name (line 306):
    
    # Call to finfo(...): (line 306)
    # Processing the call arguments (line 306)
    # Getting the type of 'ntypes' (line 306)
    ntypes_6332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 14), 'ntypes', False)
    # Obtaining the member 'longfloat' of a type (line 306)
    longfloat_6333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 306, 14), ntypes_6332, 'longfloat')
    # Processing the call keyword arguments (line 306)
    kwargs_6334 = {}
    # Getting the type of 'finfo' (line 306)
    finfo_6331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 306, 8), 'finfo', False)
    # Calling finfo(args, kwargs) (line 306)
    finfo_call_result_6335 = invoke(stypy.reporting.localization.Localization(__file__, 306, 8), finfo_6331, *[longfloat_6333], **kwargs_6334)
    
    # Assigning a type to the variable 'f' (line 306)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 306, 4), 'f', finfo_call_result_6335)
    
    # Call to print(...): (line 307)
    # Processing the call arguments (line 307)
    str_6337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 307, 10), 'str', 'longfloat epsilon:')
    # Getting the type of 'f' (line 307)
    f_6338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 32), 'f', False)
    # Obtaining the member 'eps' of a type (line 307)
    eps_6339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 307, 32), f_6338, 'eps')
    # Processing the call keyword arguments (line 307)
    kwargs_6340 = {}
    # Getting the type of 'print' (line 307)
    print_6336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 307, 4), 'print', False)
    # Calling print(args, kwargs) (line 307)
    print_call_result_6341 = invoke(stypy.reporting.localization.Localization(__file__, 307, 4), print_6336, *[str_6337, eps_6339], **kwargs_6340)
    
    
    # Call to print(...): (line 308)
    # Processing the call arguments (line 308)
    str_6343 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 308, 10), 'str', 'longfloat tiny:')
    # Getting the type of 'f' (line 308)
    f_6344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 29), 'f', False)
    # Obtaining the member 'tiny' of a type (line 308)
    tiny_6345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 308, 29), f_6344, 'tiny')
    # Processing the call keyword arguments (line 308)
    kwargs_6346 = {}
    # Getting the type of 'print' (line 308)
    print_6342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 308, 4), 'print', False)
    # Calling print(args, kwargs) (line 308)
    print_call_result_6347 = invoke(stypy.reporting.localization.Localization(__file__, 308, 4), print_6342, *[str_6343, tiny_6345], **kwargs_6346)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
