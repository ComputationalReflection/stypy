
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Standard container-class for easy multiple-inheritance.
3: 
4: Try to inherit from the ndarray instead of using this class as this is not
5: complete.
6: 
7: '''
8: from __future__ import division, absolute_import, print_function
9: 
10: from numpy.core import (
11:     array, asarray, absolute, add, subtract, multiply, divide,
12:     remainder, power, left_shift, right_shift, bitwise_and, bitwise_or,
13:     bitwise_xor, invert, less, less_equal, not_equal, equal, greater,
14:     greater_equal, shape, reshape, arange, sin, sqrt, transpose
15: )
16: from numpy.compat import long
17: 
18: 
19: class container(object):
20:     '''
21:     container(data, dtype=None, copy=True)
22: 
23:     Standard container-class for easy multiple-inheritance.
24: 
25:     Methods
26:     -------
27:     copy
28:     tostring
29:     byteswap
30:     astype
31: 
32:     '''
33:     def __init__(self, data, dtype=None, copy=True):
34:         self.array = array(data, dtype, copy=copy)
35: 
36:     def __repr__(self):
37:         if len(self.shape) > 0:
38:             return self.__class__.__name__ + repr(self.array)[len("array"):]
39:         else:
40:             return self.__class__.__name__ + "(" + repr(self.array) + ")"
41: 
42:     def __array__(self, t=None):
43:         if t:
44:             return self.array.astype(t)
45:         return self.array
46: 
47:     # Array as sequence
48:     def __len__(self):
49:         return len(self.array)
50: 
51:     def __getitem__(self, index):
52:         return self._rc(self.array[index])
53: 
54:     def __getslice__(self, i, j):
55:         return self._rc(self.array[i:j])
56: 
57:     def __setitem__(self, index, value):
58:         self.array[index] = asarray(value, self.dtype)
59: 
60:     def __setslice__(self, i, j, value):
61:         self.array[i:j] = asarray(value, self.dtype)
62: 
63:     def __abs__(self):
64:         return self._rc(absolute(self.array))
65: 
66:     def __neg__(self):
67:         return self._rc(-self.array)
68: 
69:     def __add__(self, other):
70:         return self._rc(self.array + asarray(other))
71: 
72:     __radd__ = __add__
73: 
74:     def __iadd__(self, other):
75:         add(self.array, other, self.array)
76:         return self
77: 
78:     def __sub__(self, other):
79:         return self._rc(self.array - asarray(other))
80: 
81:     def __rsub__(self, other):
82:         return self._rc(asarray(other) - self.array)
83: 
84:     def __isub__(self, other):
85:         subtract(self.array, other, self.array)
86:         return self
87: 
88:     def __mul__(self, other):
89:         return self._rc(multiply(self.array, asarray(other)))
90: 
91:     __rmul__ = __mul__
92: 
93:     def __imul__(self, other):
94:         multiply(self.array, other, self.array)
95:         return self
96: 
97:     def __div__(self, other):
98:         return self._rc(divide(self.array, asarray(other)))
99: 
100:     def __rdiv__(self, other):
101:         return self._rc(divide(asarray(other), self.array))
102: 
103:     def __idiv__(self, other):
104:         divide(self.array, other, self.array)
105:         return self
106: 
107:     def __mod__(self, other):
108:         return self._rc(remainder(self.array, other))
109: 
110:     def __rmod__(self, other):
111:         return self._rc(remainder(other, self.array))
112: 
113:     def __imod__(self, other):
114:         remainder(self.array, other, self.array)
115:         return self
116: 
117:     def __divmod__(self, other):
118:         return (self._rc(divide(self.array, other)),
119:                 self._rc(remainder(self.array, other)))
120: 
121:     def __rdivmod__(self, other):
122:         return (self._rc(divide(other, self.array)),
123:                 self._rc(remainder(other, self.array)))
124: 
125:     def __pow__(self, other):
126:         return self._rc(power(self.array, asarray(other)))
127: 
128:     def __rpow__(self, other):
129:         return self._rc(power(asarray(other), self.array))
130: 
131:     def __ipow__(self, other):
132:         power(self.array, other, self.array)
133:         return self
134: 
135:     def __lshift__(self, other):
136:         return self._rc(left_shift(self.array, other))
137: 
138:     def __rshift__(self, other):
139:         return self._rc(right_shift(self.array, other))
140: 
141:     def __rlshift__(self, other):
142:         return self._rc(left_shift(other, self.array))
143: 
144:     def __rrshift__(self, other):
145:         return self._rc(right_shift(other, self.array))
146: 
147:     def __ilshift__(self, other):
148:         left_shift(self.array, other, self.array)
149:         return self
150: 
151:     def __irshift__(self, other):
152:         right_shift(self.array, other, self.array)
153:         return self
154: 
155:     def __and__(self, other):
156:         return self._rc(bitwise_and(self.array, other))
157: 
158:     def __rand__(self, other):
159:         return self._rc(bitwise_and(other, self.array))
160: 
161:     def __iand__(self, other):
162:         bitwise_and(self.array, other, self.array)
163:         return self
164: 
165:     def __xor__(self, other):
166:         return self._rc(bitwise_xor(self.array, other))
167: 
168:     def __rxor__(self, other):
169:         return self._rc(bitwise_xor(other, self.array))
170: 
171:     def __ixor__(self, other):
172:         bitwise_xor(self.array, other, self.array)
173:         return self
174: 
175:     def __or__(self, other):
176:         return self._rc(bitwise_or(self.array, other))
177: 
178:     def __ror__(self, other):
179:         return self._rc(bitwise_or(other, self.array))
180: 
181:     def __ior__(self, other):
182:         bitwise_or(self.array, other, self.array)
183:         return self
184: 
185:     def __pos__(self):
186:         return self._rc(self.array)
187: 
188:     def __invert__(self):
189:         return self._rc(invert(self.array))
190: 
191:     def _scalarfunc(self, func):
192:         if len(self.shape) == 0:
193:             return func(self[0])
194:         else:
195:             raise TypeError(
196:                 "only rank-0 arrays can be converted to Python scalars.")
197: 
198:     def __complex__(self):
199:         return self._scalarfunc(complex)
200: 
201:     def __float__(self):
202:         return self._scalarfunc(float)
203: 
204:     def __int__(self):
205:         return self._scalarfunc(int)
206: 
207:     def __long__(self):
208:         return self._scalarfunc(long)
209: 
210:     def __hex__(self):
211:         return self._scalarfunc(hex)
212: 
213:     def __oct__(self):
214:         return self._scalarfunc(oct)
215: 
216:     def __lt__(self, other):
217:         return self._rc(less(self.array, other))
218: 
219:     def __le__(self, other):
220:         return self._rc(less_equal(self.array, other))
221: 
222:     def __eq__(self, other):
223:         return self._rc(equal(self.array, other))
224: 
225:     def __ne__(self, other):
226:         return self._rc(not_equal(self.array, other))
227: 
228:     def __gt__(self, other):
229:         return self._rc(greater(self.array, other))
230: 
231:     def __ge__(self, other):
232:         return self._rc(greater_equal(self.array, other))
233: 
234:     def copy(self):
235:         ""
236:         return self._rc(self.array.copy())
237: 
238:     def tostring(self):
239:         ""
240:         return self.array.tostring()
241: 
242:     def byteswap(self):
243:         ""
244:         return self._rc(self.array.byteswap())
245: 
246:     def astype(self, typecode):
247:         ""
248:         return self._rc(self.array.astype(typecode))
249: 
250:     def _rc(self, a):
251:         if len(shape(a)) == 0:
252:             return a
253:         else:
254:             return self.__class__(a)
255: 
256:     def __array_wrap__(self, *args):
257:         return self.__class__(args[0])
258: 
259:     def __setattr__(self, attr, value):
260:         if attr == 'array':
261:             object.__setattr__(self, attr, value)
262:             return
263:         try:
264:             self.array.__setattr__(attr, value)
265:         except AttributeError:
266:             object.__setattr__(self, attr, value)
267: 
268:     # Only called after other approaches fail.
269:     def __getattr__(self, attr):
270:         if (attr == 'array'):
271:             return object.__getattribute__(self, attr)
272:         return self.array.__getattribute__(attr)
273: 
274: #############################################################
275: # Test of class container
276: #############################################################
277: if __name__ == '__main__':
278:     temp = reshape(arange(10000), (100, 100))
279: 
280:     ua = container(temp)
281:     # new object created begin test
282:     print(dir(ua))
283:     print(shape(ua), ua.shape)  # I have changed Numeric.py
284: 
285:     ua_small = ua[:3, :5]
286:     print(ua_small)
287:     # this did not change ua[0,0], which is not normal behavior
288:     ua_small[0, 0] = 10
289:     print(ua_small[0, 0], ua[0, 0])
290:     print(sin(ua_small) / 3. * 6. + sqrt(ua_small ** 2))
291:     print(less(ua_small, 103), type(less(ua_small, 103)))
292:     print(type(ua_small * reshape(arange(15), shape(ua_small))))
293:     print(reshape(ua_small, (5, 3)))
294:     print(transpose(ua_small))
295: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_128010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nStandard container-class for easy multiple-inheritance.\n\nTry to inherit from the ndarray instead of using this class as this is not\ncomplete.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from numpy.core import array, asarray, absolute, add, subtract, multiply, divide, remainder, power, left_shift, right_shift, bitwise_and, bitwise_or, bitwise_xor, invert, less, less_equal, not_equal, equal, greater, greater_equal, shape, reshape, arange, sin, sqrt, transpose' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_128011 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core')

if (type(import_128011) is not StypyTypeError):

    if (import_128011 != 'pyd_module'):
        __import__(import_128011)
        sys_modules_128012 = sys.modules[import_128011]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', sys_modules_128012.module_type_store, module_type_store, ['array', 'asarray', 'absolute', 'add', 'subtract', 'multiply', 'divide', 'remainder', 'power', 'left_shift', 'right_shift', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert', 'less', 'less_equal', 'not_equal', 'equal', 'greater', 'greater_equal', 'shape', 'reshape', 'arange', 'sin', 'sqrt', 'transpose'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_128012, sys_modules_128012.module_type_store, module_type_store)
    else:
        from numpy.core import array, asarray, absolute, add, subtract, multiply, divide, remainder, power, left_shift, right_shift, bitwise_and, bitwise_or, bitwise_xor, invert, less, less_equal, not_equal, equal, greater, greater_equal, shape, reshape, arange, sin, sqrt, transpose

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', None, module_type_store, ['array', 'asarray', 'absolute', 'add', 'subtract', 'multiply', 'divide', 'remainder', 'power', 'left_shift', 'right_shift', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'invert', 'less', 'less_equal', 'not_equal', 'equal', 'greater', 'greater_equal', 'shape', 'reshape', 'arange', 'sin', 'sqrt', 'transpose'], [array, asarray, absolute, add, subtract, multiply, divide, remainder, power, left_shift, right_shift, bitwise_and, bitwise_or, bitwise_xor, invert, less, less_equal, not_equal, equal, greater, greater_equal, shape, reshape, arange, sin, sqrt, transpose])

else:
    # Assigning a type to the variable 'numpy.core' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy.core', import_128011)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from numpy.compat import long' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_128013 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.compat')

if (type(import_128013) is not StypyTypeError):

    if (import_128013 != 'pyd_module'):
        __import__(import_128013)
        sys_modules_128014 = sys.modules[import_128013]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.compat', sys_modules_128014.module_type_store, module_type_store, ['long'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 0), __file__, sys_modules_128014, sys_modules_128014.module_type_store, module_type_store)
    else:
        from numpy.compat import long

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.compat', None, module_type_store, ['long'], [long])

else:
    # Assigning a type to the variable 'numpy.compat' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'numpy.compat', import_128013)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')

# Declaration of the 'container' class

class container(object, ):
    str_128015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n    container(data, dtype=None, copy=True)\n\n    Standard container-class for easy multiple-inheritance.\n\n    Methods\n    -------\n    copy\n    tostring\n    byteswap\n    astype\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 33)
        None_128016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 35), 'None')
        # Getting the type of 'True' (line 33)
        True_128017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 46), 'True')
        defaults = [None_128016, True_128017]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__init__', ['data', 'dtype', 'copy'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['data', 'dtype', 'copy'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Call to a Attribute (line 34):
        
        # Call to array(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'data' (line 34)
        data_128019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'data', False)
        # Getting the type of 'dtype' (line 34)
        dtype_128020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 33), 'dtype', False)
        # Processing the call keyword arguments (line 34)
        # Getting the type of 'copy' (line 34)
        copy_128021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 45), 'copy', False)
        keyword_128022 = copy_128021
        kwargs_128023 = {'copy': keyword_128022}
        # Getting the type of 'array' (line 34)
        array_128018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 21), 'array', False)
        # Calling array(args, kwargs) (line 34)
        array_call_result_128024 = invoke(stypy.reporting.localization.Localization(__file__, 34, 21), array_128018, *[data_128019, dtype_128020], **kwargs_128023)
        
        # Getting the type of 'self' (line 34)
        self_128025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'array' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_128025, 'array', array_call_result_128024)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        container.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'container.__repr__')
        container.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        container.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        
        
        # Call to len(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'self' (line 37)
        self_128027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'self', False)
        # Obtaining the member 'shape' of a type (line 37)
        shape_128028 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), self_128027, 'shape')
        # Processing the call keyword arguments (line 37)
        kwargs_128029 = {}
        # Getting the type of 'len' (line 37)
        len_128026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'len', False)
        # Calling len(args, kwargs) (line 37)
        len_call_result_128030 = invoke(stypy.reporting.localization.Localization(__file__, 37, 11), len_128026, *[shape_128028], **kwargs_128029)
        
        int_128031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 29), 'int')
        # Applying the binary operator '>' (line 37)
        result_gt_128032 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 11), '>', len_call_result_128030, int_128031)
        
        # Testing the type of an if condition (line 37)
        if_condition_128033 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 8), result_gt_128032)
        # Assigning a type to the variable 'if_condition_128033' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'if_condition_128033', if_condition_128033)
        # SSA begins for if statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'self' (line 38)
        self_128034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 'self')
        # Obtaining the member '__class__' of a type (line 38)
        class___128035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 19), self_128034, '__class__')
        # Obtaining the member '__name__' of a type (line 38)
        name___128036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 19), class___128035, '__name__')
        
        # Obtaining the type of the subscript
        
        # Call to len(...): (line 38)
        # Processing the call arguments (line 38)
        str_128038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 66), 'str', 'array')
        # Processing the call keyword arguments (line 38)
        kwargs_128039 = {}
        # Getting the type of 'len' (line 38)
        len_128037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 62), 'len', False)
        # Calling len(args, kwargs) (line 38)
        len_call_result_128040 = invoke(stypy.reporting.localization.Localization(__file__, 38, 62), len_128037, *[str_128038], **kwargs_128039)
        
        slice_128041 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 38, 45), len_call_result_128040, None, None)
        
        # Call to repr(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_128043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 50), 'self', False)
        # Obtaining the member 'array' of a type (line 38)
        array_128044 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 50), self_128043, 'array')
        # Processing the call keyword arguments (line 38)
        kwargs_128045 = {}
        # Getting the type of 'repr' (line 38)
        repr_128042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 45), 'repr', False)
        # Calling repr(args, kwargs) (line 38)
        repr_call_result_128046 = invoke(stypy.reporting.localization.Localization(__file__, 38, 45), repr_128042, *[array_128044], **kwargs_128045)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___128047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 45), repr_call_result_128046, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_128048 = invoke(stypy.reporting.localization.Localization(__file__, 38, 45), getitem___128047, slice_128041)
        
        # Applying the binary operator '+' (line 38)
        result_add_128049 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 19), '+', name___128036, subscript_call_result_128048)
        
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'stypy_return_type', result_add_128049)
        # SSA branch for the else part of an if statement (line 37)
        module_type_store.open_ssa_branch('else')
        # Getting the type of 'self' (line 40)
        self_128050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'self')
        # Obtaining the member '__class__' of a type (line 40)
        class___128051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), self_128050, '__class__')
        # Obtaining the member '__name__' of a type (line 40)
        name___128052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 19), class___128051, '__name__')
        str_128053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 45), 'str', '(')
        # Applying the binary operator '+' (line 40)
        result_add_128054 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 19), '+', name___128052, str_128053)
        
        
        # Call to repr(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_128056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 56), 'self', False)
        # Obtaining the member 'array' of a type (line 40)
        array_128057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 56), self_128056, 'array')
        # Processing the call keyword arguments (line 40)
        kwargs_128058 = {}
        # Getting the type of 'repr' (line 40)
        repr_128055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 51), 'repr', False)
        # Calling repr(args, kwargs) (line 40)
        repr_call_result_128059 = invoke(stypy.reporting.localization.Localization(__file__, 40, 51), repr_128055, *[array_128057], **kwargs_128058)
        
        # Applying the binary operator '+' (line 40)
        result_add_128060 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 49), '+', result_add_128054, repr_call_result_128059)
        
        str_128061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 70), 'str', ')')
        # Applying the binary operator '+' (line 40)
        result_add_128062 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 68), '+', result_add_128060, str_128061)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 12), 'stypy_return_type', result_add_128062)
        # SSA join for if statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_128063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128063)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_128063


    @norecursion
    def __array__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 42)
        None_128064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 26), 'None')
        defaults = [None_128064]
        # Create a new context for function '__array__'
        module_type_store = module_type_store.open_function_context('__array__', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__array__.__dict__.__setitem__('stypy_localization', localization)
        container.__array__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__array__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__array__.__dict__.__setitem__('stypy_function_name', 'container.__array__')
        container.__array__.__dict__.__setitem__('stypy_param_names_list', ['t'])
        container.__array__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__array__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__array__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__array__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__array__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__array__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__array__', ['t'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array__', localization, ['t'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array__(...)' code ##################

        
        # Getting the type of 't' (line 43)
        t_128065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 't')
        # Testing the type of an if condition (line 43)
        if_condition_128066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), t_128065)
        # Assigning a type to the variable 'if_condition_128066' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_128066', if_condition_128066)
        # SSA begins for if statement (line 43)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to astype(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 't' (line 44)
        t_128070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 37), 't', False)
        # Processing the call keyword arguments (line 44)
        kwargs_128071 = {}
        # Getting the type of 'self' (line 44)
        self_128067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'self', False)
        # Obtaining the member 'array' of a type (line 44)
        array_128068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), self_128067, 'array')
        # Obtaining the member 'astype' of a type (line 44)
        astype_128069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 19), array_128068, 'astype')
        # Calling astype(args, kwargs) (line 44)
        astype_call_result_128072 = invoke(stypy.reporting.localization.Localization(__file__, 44, 19), astype_128069, *[t_128070], **kwargs_128071)
        
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'stypy_return_type', astype_call_result_128072)
        # SSA join for if statement (line 43)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'self' (line 45)
        self_128073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'self')
        # Obtaining the member 'array' of a type (line 45)
        array_128074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), self_128073, 'array')
        # Assigning a type to the variable 'stypy_return_type' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', array_128074)
        
        # ################# End of '__array__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array__' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_128075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128075)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array__'
        return stypy_return_type_128075


    @norecursion
    def __len__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__len__'
        module_type_store = module_type_store.open_function_context('__len__', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__len__.__dict__.__setitem__('stypy_localization', localization)
        container.__len__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__len__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__len__.__dict__.__setitem__('stypy_function_name', 'container.__len__')
        container.__len__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__len__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__len__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__len__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__len__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__len__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__len__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__len__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__len__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__len__(...)' code ##################

        
        # Call to len(...): (line 49)
        # Processing the call arguments (line 49)
        # Getting the type of 'self' (line 49)
        self_128077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'self', False)
        # Obtaining the member 'array' of a type (line 49)
        array_128078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), self_128077, 'array')
        # Processing the call keyword arguments (line 49)
        kwargs_128079 = {}
        # Getting the type of 'len' (line 49)
        len_128076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'len', False)
        # Calling len(args, kwargs) (line 49)
        len_call_result_128080 = invoke(stypy.reporting.localization.Localization(__file__, 49, 15), len_128076, *[array_128078], **kwargs_128079)
        
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', len_call_result_128080)
        
        # ################# End of '__len__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__len__' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_128081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128081)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__len__'
        return stypy_return_type_128081


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        container.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__getitem__.__dict__.__setitem__('stypy_function_name', 'container.__getitem__')
        container.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['index'])
        container.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__getitem__', ['index'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['index'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Call to _rc(...): (line 52)
        # Processing the call arguments (line 52)
        
        # Obtaining the type of the subscript
        # Getting the type of 'index' (line 52)
        index_128084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'index', False)
        # Getting the type of 'self' (line 52)
        self_128085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 24), 'self', False)
        # Obtaining the member 'array' of a type (line 52)
        array_128086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 24), self_128085, 'array')
        # Obtaining the member '__getitem__' of a type (line 52)
        getitem___128087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 24), array_128086, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 52)
        subscript_call_result_128088 = invoke(stypy.reporting.localization.Localization(__file__, 52, 24), getitem___128087, index_128084)
        
        # Processing the call keyword arguments (line 52)
        kwargs_128089 = {}
        # Getting the type of 'self' (line 52)
        self_128082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 52)
        _rc_128083 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), self_128082, '_rc')
        # Calling _rc(args, kwargs) (line 52)
        _rc_call_result_128090 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), _rc_128083, *[subscript_call_result_128088], **kwargs_128089)
        
        # Assigning a type to the variable 'stypy_return_type' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'stypy_return_type', _rc_call_result_128090)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_128091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128091)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_128091


    @norecursion
    def __getslice__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getslice__'
        module_type_store = module_type_store.open_function_context('__getslice__', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__getslice__.__dict__.__setitem__('stypy_localization', localization)
        container.__getslice__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__getslice__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__getslice__.__dict__.__setitem__('stypy_function_name', 'container.__getslice__')
        container.__getslice__.__dict__.__setitem__('stypy_param_names_list', ['i', 'j'])
        container.__getslice__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__getslice__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__getslice__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__getslice__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__getslice__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__getslice__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__getslice__', ['i', 'j'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getslice__', localization, ['i', 'j'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getslice__(...)' code ##################

        
        # Call to _rc(...): (line 55)
        # Processing the call arguments (line 55)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 55)
        i_128094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 35), 'i', False)
        # Getting the type of 'j' (line 55)
        j_128095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 37), 'j', False)
        slice_128096 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 55, 24), i_128094, j_128095, None)
        # Getting the type of 'self' (line 55)
        self_128097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'self', False)
        # Obtaining the member 'array' of a type (line 55)
        array_128098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), self_128097, 'array')
        # Obtaining the member '__getitem__' of a type (line 55)
        getitem___128099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), array_128098, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 55)
        subscript_call_result_128100 = invoke(stypy.reporting.localization.Localization(__file__, 55, 24), getitem___128099, slice_128096)
        
        # Processing the call keyword arguments (line 55)
        kwargs_128101 = {}
        # Getting the type of 'self' (line 55)
        self_128092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 55)
        _rc_128093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 15), self_128092, '_rc')
        # Calling _rc(args, kwargs) (line 55)
        _rc_call_result_128102 = invoke(stypy.reporting.localization.Localization(__file__, 55, 15), _rc_128093, *[subscript_call_result_128100], **kwargs_128101)
        
        # Assigning a type to the variable 'stypy_return_type' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'stypy_return_type', _rc_call_result_128102)
        
        # ################# End of '__getslice__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getslice__' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_128103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128103)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getslice__'
        return stypy_return_type_128103


    @norecursion
    def __setitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setitem__'
        module_type_store = module_type_store.open_function_context('__setitem__', 57, 4, False)
        # Assigning a type to the variable 'self' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__setitem__.__dict__.__setitem__('stypy_localization', localization)
        container.__setitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__setitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__setitem__.__dict__.__setitem__('stypy_function_name', 'container.__setitem__')
        container.__setitem__.__dict__.__setitem__('stypy_param_names_list', ['index', 'value'])
        container.__setitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__setitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__setitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__setitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__setitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__setitem__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__setitem__', ['index', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setitem__', localization, ['index', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setitem__(...)' code ##################

        
        # Assigning a Call to a Subscript (line 58):
        
        # Call to asarray(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'value' (line 58)
        value_128105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 36), 'value', False)
        # Getting the type of 'self' (line 58)
        self_128106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 43), 'self', False)
        # Obtaining the member 'dtype' of a type (line 58)
        dtype_128107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 43), self_128106, 'dtype')
        # Processing the call keyword arguments (line 58)
        kwargs_128108 = {}
        # Getting the type of 'asarray' (line 58)
        asarray_128104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'asarray', False)
        # Calling asarray(args, kwargs) (line 58)
        asarray_call_result_128109 = invoke(stypy.reporting.localization.Localization(__file__, 58, 28), asarray_128104, *[value_128105, dtype_128107], **kwargs_128108)
        
        # Getting the type of 'self' (line 58)
        self_128110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'self')
        # Obtaining the member 'array' of a type (line 58)
        array_128111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 8), self_128110, 'array')
        # Getting the type of 'index' (line 58)
        index_128112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 19), 'index')
        # Storing an element on a container (line 58)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 8), array_128111, (index_128112, asarray_call_result_128109))
        
        # ################# End of '__setitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_128113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128113)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setitem__'
        return stypy_return_type_128113


    @norecursion
    def __setslice__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setslice__'
        module_type_store = module_type_store.open_function_context('__setslice__', 60, 4, False)
        # Assigning a type to the variable 'self' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__setslice__.__dict__.__setitem__('stypy_localization', localization)
        container.__setslice__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__setslice__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__setslice__.__dict__.__setitem__('stypy_function_name', 'container.__setslice__')
        container.__setslice__.__dict__.__setitem__('stypy_param_names_list', ['i', 'j', 'value'])
        container.__setslice__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__setslice__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__setslice__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__setslice__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__setslice__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__setslice__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__setslice__', ['i', 'j', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setslice__', localization, ['i', 'j', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setslice__(...)' code ##################

        
        # Assigning a Call to a Subscript (line 61):
        
        # Call to asarray(...): (line 61)
        # Processing the call arguments (line 61)
        # Getting the type of 'value' (line 61)
        value_128115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'value', False)
        # Getting the type of 'self' (line 61)
        self_128116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 41), 'self', False)
        # Obtaining the member 'dtype' of a type (line 61)
        dtype_128117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 41), self_128116, 'dtype')
        # Processing the call keyword arguments (line 61)
        kwargs_128118 = {}
        # Getting the type of 'asarray' (line 61)
        asarray_128114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 26), 'asarray', False)
        # Calling asarray(args, kwargs) (line 61)
        asarray_call_result_128119 = invoke(stypy.reporting.localization.Localization(__file__, 61, 26), asarray_128114, *[value_128115, dtype_128117], **kwargs_128118)
        
        # Getting the type of 'self' (line 61)
        self_128120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'self')
        # Obtaining the member 'array' of a type (line 61)
        array_128121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 8), self_128120, 'array')
        # Getting the type of 'i' (line 61)
        i_128122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'i')
        # Getting the type of 'j' (line 61)
        j_128123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'j')
        slice_128124 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 61, 8), i_128122, j_128123, None)
        # Storing an element on a container (line 61)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 61, 8), array_128121, (slice_128124, asarray_call_result_128119))
        
        # ################# End of '__setslice__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setslice__' in the type store
        # Getting the type of 'stypy_return_type' (line 60)
        stypy_return_type_128125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setslice__'
        return stypy_return_type_128125


    @norecursion
    def __abs__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__abs__'
        module_type_store = module_type_store.open_function_context('__abs__', 63, 4, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__abs__.__dict__.__setitem__('stypy_localization', localization)
        container.__abs__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__abs__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__abs__.__dict__.__setitem__('stypy_function_name', 'container.__abs__')
        container.__abs__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__abs__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__abs__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__abs__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__abs__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__abs__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__abs__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__abs__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__abs__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__abs__(...)' code ##################

        
        # Call to _rc(...): (line 64)
        # Processing the call arguments (line 64)
        
        # Call to absolute(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'self' (line 64)
        self_128129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 33), 'self', False)
        # Obtaining the member 'array' of a type (line 64)
        array_128130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 33), self_128129, 'array')
        # Processing the call keyword arguments (line 64)
        kwargs_128131 = {}
        # Getting the type of 'absolute' (line 64)
        absolute_128128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 24), 'absolute', False)
        # Calling absolute(args, kwargs) (line 64)
        absolute_call_result_128132 = invoke(stypy.reporting.localization.Localization(__file__, 64, 24), absolute_128128, *[array_128130], **kwargs_128131)
        
        # Processing the call keyword arguments (line 64)
        kwargs_128133 = {}
        # Getting the type of 'self' (line 64)
        self_128126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 64)
        _rc_128127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), self_128126, '_rc')
        # Calling _rc(args, kwargs) (line 64)
        _rc_call_result_128134 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), _rc_128127, *[absolute_call_result_128132], **kwargs_128133)
        
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', _rc_call_result_128134)
        
        # ################# End of '__abs__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__abs__' in the type store
        # Getting the type of 'stypy_return_type' (line 63)
        stypy_return_type_128135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__abs__'
        return stypy_return_type_128135


    @norecursion
    def __neg__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__neg__'
        module_type_store = module_type_store.open_function_context('__neg__', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__neg__.__dict__.__setitem__('stypy_localization', localization)
        container.__neg__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__neg__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__neg__.__dict__.__setitem__('stypy_function_name', 'container.__neg__')
        container.__neg__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__neg__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__neg__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__neg__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__neg__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__neg__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__neg__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__neg__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__neg__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__neg__(...)' code ##################

        
        # Call to _rc(...): (line 67)
        # Processing the call arguments (line 67)
        
        # Getting the type of 'self' (line 67)
        self_128138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 25), 'self', False)
        # Obtaining the member 'array' of a type (line 67)
        array_128139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 25), self_128138, 'array')
        # Applying the 'usub' unary operator (line 67)
        result___neg___128140 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 24), 'usub', array_128139)
        
        # Processing the call keyword arguments (line 67)
        kwargs_128141 = {}
        # Getting the type of 'self' (line 67)
        self_128136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 67)
        _rc_128137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 15), self_128136, '_rc')
        # Calling _rc(args, kwargs) (line 67)
        _rc_call_result_128142 = invoke(stypy.reporting.localization.Localization(__file__, 67, 15), _rc_128137, *[result___neg___128140], **kwargs_128141)
        
        # Assigning a type to the variable 'stypy_return_type' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'stypy_return_type', _rc_call_result_128142)
        
        # ################# End of '__neg__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__neg__' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_128143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128143)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__neg__'
        return stypy_return_type_128143


    @norecursion
    def __add__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__add__'
        module_type_store = module_type_store.open_function_context('__add__', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__add__.__dict__.__setitem__('stypy_localization', localization)
        container.__add__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__add__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__add__.__dict__.__setitem__('stypy_function_name', 'container.__add__')
        container.__add__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__add__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__add__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__add__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__add__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__add__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__add__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__add__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__add__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__add__(...)' code ##################

        
        # Call to _rc(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'self' (line 70)
        self_128146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 24), 'self', False)
        # Obtaining the member 'array' of a type (line 70)
        array_128147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 24), self_128146, 'array')
        
        # Call to asarray(...): (line 70)
        # Processing the call arguments (line 70)
        # Getting the type of 'other' (line 70)
        other_128149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 45), 'other', False)
        # Processing the call keyword arguments (line 70)
        kwargs_128150 = {}
        # Getting the type of 'asarray' (line 70)
        asarray_128148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 37), 'asarray', False)
        # Calling asarray(args, kwargs) (line 70)
        asarray_call_result_128151 = invoke(stypy.reporting.localization.Localization(__file__, 70, 37), asarray_128148, *[other_128149], **kwargs_128150)
        
        # Applying the binary operator '+' (line 70)
        result_add_128152 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 24), '+', array_128147, asarray_call_result_128151)
        
        # Processing the call keyword arguments (line 70)
        kwargs_128153 = {}
        # Getting the type of 'self' (line 70)
        self_128144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 70)
        _rc_128145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 15), self_128144, '_rc')
        # Calling _rc(args, kwargs) (line 70)
        _rc_call_result_128154 = invoke(stypy.reporting.localization.Localization(__file__, 70, 15), _rc_128145, *[result_add_128152], **kwargs_128153)
        
        # Assigning a type to the variable 'stypy_return_type' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'stypy_return_type', _rc_call_result_128154)
        
        # ################# End of '__add__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__add__' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_128155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128155)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__add__'
        return stypy_return_type_128155


    @norecursion
    def __iadd__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iadd__'
        module_type_store = module_type_store.open_function_context('__iadd__', 74, 4, False)
        # Assigning a type to the variable 'self' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__iadd__.__dict__.__setitem__('stypy_localization', localization)
        container.__iadd__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__iadd__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__iadd__.__dict__.__setitem__('stypy_function_name', 'container.__iadd__')
        container.__iadd__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__iadd__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__iadd__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__iadd__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__iadd__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__iadd__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__iadd__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__iadd__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iadd__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iadd__(...)' code ##################

        
        # Call to add(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'self' (line 75)
        self_128157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'self', False)
        # Obtaining the member 'array' of a type (line 75)
        array_128158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 12), self_128157, 'array')
        # Getting the type of 'other' (line 75)
        other_128159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'other', False)
        # Getting the type of 'self' (line 75)
        self_128160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'self', False)
        # Obtaining the member 'array' of a type (line 75)
        array_128161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 31), self_128160, 'array')
        # Processing the call keyword arguments (line 75)
        kwargs_128162 = {}
        # Getting the type of 'add' (line 75)
        add_128156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'add', False)
        # Calling add(args, kwargs) (line 75)
        add_call_result_128163 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), add_128156, *[array_128158, other_128159, array_128161], **kwargs_128162)
        
        # Getting the type of 'self' (line 76)
        self_128164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 76)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'stypy_return_type', self_128164)
        
        # ################# End of '__iadd__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iadd__' in the type store
        # Getting the type of 'stypy_return_type' (line 74)
        stypy_return_type_128165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128165)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iadd__'
        return stypy_return_type_128165


    @norecursion
    def __sub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__sub__'
        module_type_store = module_type_store.open_function_context('__sub__', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__sub__.__dict__.__setitem__('stypy_localization', localization)
        container.__sub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__sub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__sub__.__dict__.__setitem__('stypy_function_name', 'container.__sub__')
        container.__sub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__sub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__sub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__sub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__sub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__sub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__sub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__sub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__sub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__sub__(...)' code ##################

        
        # Call to _rc(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'self' (line 79)
        self_128168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'self', False)
        # Obtaining the member 'array' of a type (line 79)
        array_128169 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), self_128168, 'array')
        
        # Call to asarray(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'other' (line 79)
        other_128171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 45), 'other', False)
        # Processing the call keyword arguments (line 79)
        kwargs_128172 = {}
        # Getting the type of 'asarray' (line 79)
        asarray_128170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'asarray', False)
        # Calling asarray(args, kwargs) (line 79)
        asarray_call_result_128173 = invoke(stypy.reporting.localization.Localization(__file__, 79, 37), asarray_128170, *[other_128171], **kwargs_128172)
        
        # Applying the binary operator '-' (line 79)
        result_sub_128174 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 24), '-', array_128169, asarray_call_result_128173)
        
        # Processing the call keyword arguments (line 79)
        kwargs_128175 = {}
        # Getting the type of 'self' (line 79)
        self_128166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 79)
        _rc_128167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 15), self_128166, '_rc')
        # Calling _rc(args, kwargs) (line 79)
        _rc_call_result_128176 = invoke(stypy.reporting.localization.Localization(__file__, 79, 15), _rc_128167, *[result_sub_128174], **kwargs_128175)
        
        # Assigning a type to the variable 'stypy_return_type' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'stypy_return_type', _rc_call_result_128176)
        
        # ################# End of '__sub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__sub__' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_128177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128177)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__sub__'
        return stypy_return_type_128177


    @norecursion
    def __rsub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rsub__'
        module_type_store = module_type_store.open_function_context('__rsub__', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rsub__.__dict__.__setitem__('stypy_localization', localization)
        container.__rsub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rsub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rsub__.__dict__.__setitem__('stypy_function_name', 'container.__rsub__')
        container.__rsub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rsub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rsub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rsub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rsub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rsub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rsub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rsub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rsub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rsub__(...)' code ##################

        
        # Call to _rc(...): (line 82)
        # Processing the call arguments (line 82)
        
        # Call to asarray(...): (line 82)
        # Processing the call arguments (line 82)
        # Getting the type of 'other' (line 82)
        other_128181 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 32), 'other', False)
        # Processing the call keyword arguments (line 82)
        kwargs_128182 = {}
        # Getting the type of 'asarray' (line 82)
        asarray_128180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 24), 'asarray', False)
        # Calling asarray(args, kwargs) (line 82)
        asarray_call_result_128183 = invoke(stypy.reporting.localization.Localization(__file__, 82, 24), asarray_128180, *[other_128181], **kwargs_128182)
        
        # Getting the type of 'self' (line 82)
        self_128184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 41), 'self', False)
        # Obtaining the member 'array' of a type (line 82)
        array_128185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 41), self_128184, 'array')
        # Applying the binary operator '-' (line 82)
        result_sub_128186 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 24), '-', asarray_call_result_128183, array_128185)
        
        # Processing the call keyword arguments (line 82)
        kwargs_128187 = {}
        # Getting the type of 'self' (line 82)
        self_128178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 82)
        _rc_128179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 15), self_128178, '_rc')
        # Calling _rc(args, kwargs) (line 82)
        _rc_call_result_128188 = invoke(stypy.reporting.localization.Localization(__file__, 82, 15), _rc_128179, *[result_sub_128186], **kwargs_128187)
        
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', _rc_call_result_128188)
        
        # ################# End of '__rsub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rsub__' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_128189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rsub__'
        return stypy_return_type_128189


    @norecursion
    def __isub__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__isub__'
        module_type_store = module_type_store.open_function_context('__isub__', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__isub__.__dict__.__setitem__('stypy_localization', localization)
        container.__isub__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__isub__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__isub__.__dict__.__setitem__('stypy_function_name', 'container.__isub__')
        container.__isub__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__isub__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__isub__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__isub__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__isub__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__isub__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__isub__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__isub__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__isub__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__isub__(...)' code ##################

        
        # Call to subtract(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 85)
        self_128191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 17), 'self', False)
        # Obtaining the member 'array' of a type (line 85)
        array_128192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 17), self_128191, 'array')
        # Getting the type of 'other' (line 85)
        other_128193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'other', False)
        # Getting the type of 'self' (line 85)
        self_128194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'self', False)
        # Obtaining the member 'array' of a type (line 85)
        array_128195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 36), self_128194, 'array')
        # Processing the call keyword arguments (line 85)
        kwargs_128196 = {}
        # Getting the type of 'subtract' (line 85)
        subtract_128190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'subtract', False)
        # Calling subtract(args, kwargs) (line 85)
        subtract_call_result_128197 = invoke(stypy.reporting.localization.Localization(__file__, 85, 8), subtract_128190, *[array_128192, other_128193, array_128195], **kwargs_128196)
        
        # Getting the type of 'self' (line 86)
        self_128198 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'stypy_return_type', self_128198)
        
        # ################# End of '__isub__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__isub__' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_128199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128199)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__isub__'
        return stypy_return_type_128199


    @norecursion
    def __mul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mul__'
        module_type_store = module_type_store.open_function_context('__mul__', 88, 4, False)
        # Assigning a type to the variable 'self' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__mul__.__dict__.__setitem__('stypy_localization', localization)
        container.__mul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__mul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__mul__.__dict__.__setitem__('stypy_function_name', 'container.__mul__')
        container.__mul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__mul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__mul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__mul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__mul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__mul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__mul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__mul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mul__(...)' code ##################

        
        # Call to _rc(...): (line 89)
        # Processing the call arguments (line 89)
        
        # Call to multiply(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'self' (line 89)
        self_128203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 33), 'self', False)
        # Obtaining the member 'array' of a type (line 89)
        array_128204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 33), self_128203, 'array')
        
        # Call to asarray(...): (line 89)
        # Processing the call arguments (line 89)
        # Getting the type of 'other' (line 89)
        other_128206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'other', False)
        # Processing the call keyword arguments (line 89)
        kwargs_128207 = {}
        # Getting the type of 'asarray' (line 89)
        asarray_128205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 45), 'asarray', False)
        # Calling asarray(args, kwargs) (line 89)
        asarray_call_result_128208 = invoke(stypy.reporting.localization.Localization(__file__, 89, 45), asarray_128205, *[other_128206], **kwargs_128207)
        
        # Processing the call keyword arguments (line 89)
        kwargs_128209 = {}
        # Getting the type of 'multiply' (line 89)
        multiply_128202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'multiply', False)
        # Calling multiply(args, kwargs) (line 89)
        multiply_call_result_128210 = invoke(stypy.reporting.localization.Localization(__file__, 89, 24), multiply_128202, *[array_128204, asarray_call_result_128208], **kwargs_128209)
        
        # Processing the call keyword arguments (line 89)
        kwargs_128211 = {}
        # Getting the type of 'self' (line 89)
        self_128200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 89)
        _rc_128201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 15), self_128200, '_rc')
        # Calling _rc(args, kwargs) (line 89)
        _rc_call_result_128212 = invoke(stypy.reporting.localization.Localization(__file__, 89, 15), _rc_128201, *[multiply_call_result_128210], **kwargs_128211)
        
        # Assigning a type to the variable 'stypy_return_type' (line 89)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'stypy_return_type', _rc_call_result_128212)
        
        # ################# End of '__mul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mul__' in the type store
        # Getting the type of 'stypy_return_type' (line 88)
        stypy_return_type_128213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128213)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mul__'
        return stypy_return_type_128213


    @norecursion
    def __imul__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__imul__'
        module_type_store = module_type_store.open_function_context('__imul__', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__imul__.__dict__.__setitem__('stypy_localization', localization)
        container.__imul__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__imul__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__imul__.__dict__.__setitem__('stypy_function_name', 'container.__imul__')
        container.__imul__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__imul__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__imul__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__imul__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__imul__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__imul__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__imul__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__imul__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__imul__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__imul__(...)' code ##################

        
        # Call to multiply(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'self' (line 94)
        self_128215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 17), 'self', False)
        # Obtaining the member 'array' of a type (line 94)
        array_128216 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 17), self_128215, 'array')
        # Getting the type of 'other' (line 94)
        other_128217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 29), 'other', False)
        # Getting the type of 'self' (line 94)
        self_128218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 36), 'self', False)
        # Obtaining the member 'array' of a type (line 94)
        array_128219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 36), self_128218, 'array')
        # Processing the call keyword arguments (line 94)
        kwargs_128220 = {}
        # Getting the type of 'multiply' (line 94)
        multiply_128214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'multiply', False)
        # Calling multiply(args, kwargs) (line 94)
        multiply_call_result_128221 = invoke(stypy.reporting.localization.Localization(__file__, 94, 8), multiply_128214, *[array_128216, other_128217, array_128219], **kwargs_128220)
        
        # Getting the type of 'self' (line 95)
        self_128222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', self_128222)
        
        # ################# End of '__imul__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__imul__' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_128223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128223)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__imul__'
        return stypy_return_type_128223


    @norecursion
    def __div__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__div__'
        module_type_store = module_type_store.open_function_context('__div__', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__div__.__dict__.__setitem__('stypy_localization', localization)
        container.__div__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__div__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__div__.__dict__.__setitem__('stypy_function_name', 'container.__div__')
        container.__div__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__div__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__div__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__div__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__div__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__div__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__div__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__div__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__div__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__div__(...)' code ##################

        
        # Call to _rc(...): (line 98)
        # Processing the call arguments (line 98)
        
        # Call to divide(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'self' (line 98)
        self_128227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 31), 'self', False)
        # Obtaining the member 'array' of a type (line 98)
        array_128228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 31), self_128227, 'array')
        
        # Call to asarray(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'other' (line 98)
        other_128230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 51), 'other', False)
        # Processing the call keyword arguments (line 98)
        kwargs_128231 = {}
        # Getting the type of 'asarray' (line 98)
        asarray_128229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 43), 'asarray', False)
        # Calling asarray(args, kwargs) (line 98)
        asarray_call_result_128232 = invoke(stypy.reporting.localization.Localization(__file__, 98, 43), asarray_128229, *[other_128230], **kwargs_128231)
        
        # Processing the call keyword arguments (line 98)
        kwargs_128233 = {}
        # Getting the type of 'divide' (line 98)
        divide_128226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'divide', False)
        # Calling divide(args, kwargs) (line 98)
        divide_call_result_128234 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), divide_128226, *[array_128228, asarray_call_result_128232], **kwargs_128233)
        
        # Processing the call keyword arguments (line 98)
        kwargs_128235 = {}
        # Getting the type of 'self' (line 98)
        self_128224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 98)
        _rc_128225 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 15), self_128224, '_rc')
        # Calling _rc(args, kwargs) (line 98)
        _rc_call_result_128236 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), _rc_128225, *[divide_call_result_128234], **kwargs_128235)
        
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', _rc_call_result_128236)
        
        # ################# End of '__div__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__div__' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_128237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128237)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__div__'
        return stypy_return_type_128237


    @norecursion
    def __rdiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rdiv__'
        module_type_store = module_type_store.open_function_context('__rdiv__', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rdiv__.__dict__.__setitem__('stypy_localization', localization)
        container.__rdiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rdiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rdiv__.__dict__.__setitem__('stypy_function_name', 'container.__rdiv__')
        container.__rdiv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rdiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rdiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rdiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rdiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rdiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rdiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rdiv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rdiv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rdiv__(...)' code ##################

        
        # Call to _rc(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to divide(...): (line 101)
        # Processing the call arguments (line 101)
        
        # Call to asarray(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'other' (line 101)
        other_128242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 39), 'other', False)
        # Processing the call keyword arguments (line 101)
        kwargs_128243 = {}
        # Getting the type of 'asarray' (line 101)
        asarray_128241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 31), 'asarray', False)
        # Calling asarray(args, kwargs) (line 101)
        asarray_call_result_128244 = invoke(stypy.reporting.localization.Localization(__file__, 101, 31), asarray_128241, *[other_128242], **kwargs_128243)
        
        # Getting the type of 'self' (line 101)
        self_128245 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 47), 'self', False)
        # Obtaining the member 'array' of a type (line 101)
        array_128246 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 47), self_128245, 'array')
        # Processing the call keyword arguments (line 101)
        kwargs_128247 = {}
        # Getting the type of 'divide' (line 101)
        divide_128240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'divide', False)
        # Calling divide(args, kwargs) (line 101)
        divide_call_result_128248 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), divide_128240, *[asarray_call_result_128244, array_128246], **kwargs_128247)
        
        # Processing the call keyword arguments (line 101)
        kwargs_128249 = {}
        # Getting the type of 'self' (line 101)
        self_128238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 101)
        _rc_128239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 15), self_128238, '_rc')
        # Calling _rc(args, kwargs) (line 101)
        _rc_call_result_128250 = invoke(stypy.reporting.localization.Localization(__file__, 101, 15), _rc_128239, *[divide_call_result_128248], **kwargs_128249)
        
        # Assigning a type to the variable 'stypy_return_type' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'stypy_return_type', _rc_call_result_128250)
        
        # ################# End of '__rdiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rdiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_128251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128251)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rdiv__'
        return stypy_return_type_128251


    @norecursion
    def __idiv__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__idiv__'
        module_type_store = module_type_store.open_function_context('__idiv__', 103, 4, False)
        # Assigning a type to the variable 'self' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__idiv__.__dict__.__setitem__('stypy_localization', localization)
        container.__idiv__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__idiv__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__idiv__.__dict__.__setitem__('stypy_function_name', 'container.__idiv__')
        container.__idiv__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__idiv__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__idiv__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__idiv__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__idiv__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__idiv__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__idiv__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__idiv__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__idiv__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__idiv__(...)' code ##################

        
        # Call to divide(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'self' (line 104)
        self_128253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 15), 'self', False)
        # Obtaining the member 'array' of a type (line 104)
        array_128254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 15), self_128253, 'array')
        # Getting the type of 'other' (line 104)
        other_128255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'other', False)
        # Getting the type of 'self' (line 104)
        self_128256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 34), 'self', False)
        # Obtaining the member 'array' of a type (line 104)
        array_128257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 34), self_128256, 'array')
        # Processing the call keyword arguments (line 104)
        kwargs_128258 = {}
        # Getting the type of 'divide' (line 104)
        divide_128252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'divide', False)
        # Calling divide(args, kwargs) (line 104)
        divide_call_result_128259 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), divide_128252, *[array_128254, other_128255, array_128257], **kwargs_128258)
        
        # Getting the type of 'self' (line 105)
        self_128260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'stypy_return_type', self_128260)
        
        # ################# End of '__idiv__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__idiv__' in the type store
        # Getting the type of 'stypy_return_type' (line 103)
        stypy_return_type_128261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__idiv__'
        return stypy_return_type_128261


    @norecursion
    def __mod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__mod__'
        module_type_store = module_type_store.open_function_context('__mod__', 107, 4, False)
        # Assigning a type to the variable 'self' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__mod__.__dict__.__setitem__('stypy_localization', localization)
        container.__mod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__mod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__mod__.__dict__.__setitem__('stypy_function_name', 'container.__mod__')
        container.__mod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__mod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__mod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__mod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__mod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__mod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__mod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__mod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__mod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__mod__(...)' code ##################

        
        # Call to _rc(...): (line 108)
        # Processing the call arguments (line 108)
        
        # Call to remainder(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'self' (line 108)
        self_128265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'self', False)
        # Obtaining the member 'array' of a type (line 108)
        array_128266 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 34), self_128265, 'array')
        # Getting the type of 'other' (line 108)
        other_128267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 46), 'other', False)
        # Processing the call keyword arguments (line 108)
        kwargs_128268 = {}
        # Getting the type of 'remainder' (line 108)
        remainder_128264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 24), 'remainder', False)
        # Calling remainder(args, kwargs) (line 108)
        remainder_call_result_128269 = invoke(stypy.reporting.localization.Localization(__file__, 108, 24), remainder_128264, *[array_128266, other_128267], **kwargs_128268)
        
        # Processing the call keyword arguments (line 108)
        kwargs_128270 = {}
        # Getting the type of 'self' (line 108)
        self_128262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 108)
        _rc_128263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 15), self_128262, '_rc')
        # Calling _rc(args, kwargs) (line 108)
        _rc_call_result_128271 = invoke(stypy.reporting.localization.Localization(__file__, 108, 15), _rc_128263, *[remainder_call_result_128269], **kwargs_128270)
        
        # Assigning a type to the variable 'stypy_return_type' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'stypy_return_type', _rc_call_result_128271)
        
        # ################# End of '__mod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__mod__' in the type store
        # Getting the type of 'stypy_return_type' (line 107)
        stypy_return_type_128272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128272)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__mod__'
        return stypy_return_type_128272


    @norecursion
    def __rmod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rmod__'
        module_type_store = module_type_store.open_function_context('__rmod__', 110, 4, False)
        # Assigning a type to the variable 'self' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rmod__.__dict__.__setitem__('stypy_localization', localization)
        container.__rmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rmod__.__dict__.__setitem__('stypy_function_name', 'container.__rmod__')
        container.__rmod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rmod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rmod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rmod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rmod__(...)' code ##################

        
        # Call to _rc(...): (line 111)
        # Processing the call arguments (line 111)
        
        # Call to remainder(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'other' (line 111)
        other_128276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 34), 'other', False)
        # Getting the type of 'self' (line 111)
        self_128277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'self', False)
        # Obtaining the member 'array' of a type (line 111)
        array_128278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 41), self_128277, 'array')
        # Processing the call keyword arguments (line 111)
        kwargs_128279 = {}
        # Getting the type of 'remainder' (line 111)
        remainder_128275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 24), 'remainder', False)
        # Calling remainder(args, kwargs) (line 111)
        remainder_call_result_128280 = invoke(stypy.reporting.localization.Localization(__file__, 111, 24), remainder_128275, *[other_128276, array_128278], **kwargs_128279)
        
        # Processing the call keyword arguments (line 111)
        kwargs_128281 = {}
        # Getting the type of 'self' (line 111)
        self_128273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 111)
        _rc_128274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 15), self_128273, '_rc')
        # Calling _rc(args, kwargs) (line 111)
        _rc_call_result_128282 = invoke(stypy.reporting.localization.Localization(__file__, 111, 15), _rc_128274, *[remainder_call_result_128280], **kwargs_128281)
        
        # Assigning a type to the variable 'stypy_return_type' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'stypy_return_type', _rc_call_result_128282)
        
        # ################# End of '__rmod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rmod__' in the type store
        # Getting the type of 'stypy_return_type' (line 110)
        stypy_return_type_128283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128283)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rmod__'
        return stypy_return_type_128283


    @norecursion
    def __imod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__imod__'
        module_type_store = module_type_store.open_function_context('__imod__', 113, 4, False)
        # Assigning a type to the variable 'self' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__imod__.__dict__.__setitem__('stypy_localization', localization)
        container.__imod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__imod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__imod__.__dict__.__setitem__('stypy_function_name', 'container.__imod__')
        container.__imod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__imod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__imod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__imod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__imod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__imod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__imod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__imod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__imod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__imod__(...)' code ##################

        
        # Call to remainder(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'self' (line 114)
        self_128285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'self', False)
        # Obtaining the member 'array' of a type (line 114)
        array_128286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 18), self_128285, 'array')
        # Getting the type of 'other' (line 114)
        other_128287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 30), 'other', False)
        # Getting the type of 'self' (line 114)
        self_128288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 37), 'self', False)
        # Obtaining the member 'array' of a type (line 114)
        array_128289 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 37), self_128288, 'array')
        # Processing the call keyword arguments (line 114)
        kwargs_128290 = {}
        # Getting the type of 'remainder' (line 114)
        remainder_128284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'remainder', False)
        # Calling remainder(args, kwargs) (line 114)
        remainder_call_result_128291 = invoke(stypy.reporting.localization.Localization(__file__, 114, 8), remainder_128284, *[array_128286, other_128287, array_128289], **kwargs_128290)
        
        # Getting the type of 'self' (line 115)
        self_128292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'stypy_return_type', self_128292)
        
        # ################# End of '__imod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__imod__' in the type store
        # Getting the type of 'stypy_return_type' (line 113)
        stypy_return_type_128293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128293)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__imod__'
        return stypy_return_type_128293


    @norecursion
    def __divmod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__divmod__'
        module_type_store = module_type_store.open_function_context('__divmod__', 117, 4, False)
        # Assigning a type to the variable 'self' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__divmod__.__dict__.__setitem__('stypy_localization', localization)
        container.__divmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__divmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__divmod__.__dict__.__setitem__('stypy_function_name', 'container.__divmod__')
        container.__divmod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__divmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__divmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__divmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__divmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__divmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__divmod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__divmod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__divmod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__divmod__(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 118)
        tuple_128294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 118)
        # Adding element type (line 118)
        
        # Call to _rc(...): (line 118)
        # Processing the call arguments (line 118)
        
        # Call to divide(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'self' (line 118)
        self_128298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 32), 'self', False)
        # Obtaining the member 'array' of a type (line 118)
        array_128299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 32), self_128298, 'array')
        # Getting the type of 'other' (line 118)
        other_128300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 44), 'other', False)
        # Processing the call keyword arguments (line 118)
        kwargs_128301 = {}
        # Getting the type of 'divide' (line 118)
        divide_128297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 25), 'divide', False)
        # Calling divide(args, kwargs) (line 118)
        divide_call_result_128302 = invoke(stypy.reporting.localization.Localization(__file__, 118, 25), divide_128297, *[array_128299, other_128300], **kwargs_128301)
        
        # Processing the call keyword arguments (line 118)
        kwargs_128303 = {}
        # Getting the type of 'self' (line 118)
        self_128295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'self', False)
        # Obtaining the member '_rc' of a type (line 118)
        _rc_128296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), self_128295, '_rc')
        # Calling _rc(args, kwargs) (line 118)
        _rc_call_result_128304 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), _rc_128296, *[divide_call_result_128302], **kwargs_128303)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 16), tuple_128294, _rc_call_result_128304)
        # Adding element type (line 118)
        
        # Call to _rc(...): (line 119)
        # Processing the call arguments (line 119)
        
        # Call to remainder(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'self' (line 119)
        self_128308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 35), 'self', False)
        # Obtaining the member 'array' of a type (line 119)
        array_128309 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 35), self_128308, 'array')
        # Getting the type of 'other' (line 119)
        other_128310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 47), 'other', False)
        # Processing the call keyword arguments (line 119)
        kwargs_128311 = {}
        # Getting the type of 'remainder' (line 119)
        remainder_128307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'remainder', False)
        # Calling remainder(args, kwargs) (line 119)
        remainder_call_result_128312 = invoke(stypy.reporting.localization.Localization(__file__, 119, 25), remainder_128307, *[array_128309, other_128310], **kwargs_128311)
        
        # Processing the call keyword arguments (line 119)
        kwargs_128313 = {}
        # Getting the type of 'self' (line 119)
        self_128305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'self', False)
        # Obtaining the member '_rc' of a type (line 119)
        _rc_128306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 16), self_128305, '_rc')
        # Calling _rc(args, kwargs) (line 119)
        _rc_call_result_128314 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), _rc_128306, *[remainder_call_result_128312], **kwargs_128313)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 118, 16), tuple_128294, _rc_call_result_128314)
        
        # Assigning a type to the variable 'stypy_return_type' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'stypy_return_type', tuple_128294)
        
        # ################# End of '__divmod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__divmod__' in the type store
        # Getting the type of 'stypy_return_type' (line 117)
        stypy_return_type_128315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128315)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__divmod__'
        return stypy_return_type_128315


    @norecursion
    def __rdivmod__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rdivmod__'
        module_type_store = module_type_store.open_function_context('__rdivmod__', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rdivmod__.__dict__.__setitem__('stypy_localization', localization)
        container.__rdivmod__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rdivmod__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rdivmod__.__dict__.__setitem__('stypy_function_name', 'container.__rdivmod__')
        container.__rdivmod__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rdivmod__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rdivmod__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rdivmod__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rdivmod__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rdivmod__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rdivmod__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rdivmod__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rdivmod__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rdivmod__(...)' code ##################

        
        # Obtaining an instance of the builtin type 'tuple' (line 122)
        tuple_128316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 16), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 122)
        # Adding element type (line 122)
        
        # Call to _rc(...): (line 122)
        # Processing the call arguments (line 122)
        
        # Call to divide(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'other' (line 122)
        other_128320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 32), 'other', False)
        # Getting the type of 'self' (line 122)
        self_128321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 39), 'self', False)
        # Obtaining the member 'array' of a type (line 122)
        array_128322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 39), self_128321, 'array')
        # Processing the call keyword arguments (line 122)
        kwargs_128323 = {}
        # Getting the type of 'divide' (line 122)
        divide_128319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 25), 'divide', False)
        # Calling divide(args, kwargs) (line 122)
        divide_call_result_128324 = invoke(stypy.reporting.localization.Localization(__file__, 122, 25), divide_128319, *[other_128320, array_128322], **kwargs_128323)
        
        # Processing the call keyword arguments (line 122)
        kwargs_128325 = {}
        # Getting the type of 'self' (line 122)
        self_128317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'self', False)
        # Obtaining the member '_rc' of a type (line 122)
        _rc_128318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 16), self_128317, '_rc')
        # Calling _rc(args, kwargs) (line 122)
        _rc_call_result_128326 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), _rc_128318, *[divide_call_result_128324], **kwargs_128325)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), tuple_128316, _rc_call_result_128326)
        # Adding element type (line 122)
        
        # Call to _rc(...): (line 123)
        # Processing the call arguments (line 123)
        
        # Call to remainder(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'other' (line 123)
        other_128330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 35), 'other', False)
        # Getting the type of 'self' (line 123)
        self_128331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 42), 'self', False)
        # Obtaining the member 'array' of a type (line 123)
        array_128332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 42), self_128331, 'array')
        # Processing the call keyword arguments (line 123)
        kwargs_128333 = {}
        # Getting the type of 'remainder' (line 123)
        remainder_128329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'remainder', False)
        # Calling remainder(args, kwargs) (line 123)
        remainder_call_result_128334 = invoke(stypy.reporting.localization.Localization(__file__, 123, 25), remainder_128329, *[other_128330, array_128332], **kwargs_128333)
        
        # Processing the call keyword arguments (line 123)
        kwargs_128335 = {}
        # Getting the type of 'self' (line 123)
        self_128327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'self', False)
        # Obtaining the member '_rc' of a type (line 123)
        _rc_128328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), self_128327, '_rc')
        # Calling _rc(args, kwargs) (line 123)
        _rc_call_result_128336 = invoke(stypy.reporting.localization.Localization(__file__, 123, 16), _rc_128328, *[remainder_call_result_128334], **kwargs_128335)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 16), tuple_128316, _rc_call_result_128336)
        
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'stypy_return_type', tuple_128316)
        
        # ################# End of '__rdivmod__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rdivmod__' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_128337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128337)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rdivmod__'
        return stypy_return_type_128337


    @norecursion
    def __pow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pow__'
        module_type_store = module_type_store.open_function_context('__pow__', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__pow__.__dict__.__setitem__('stypy_localization', localization)
        container.__pow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__pow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__pow__.__dict__.__setitem__('stypy_function_name', 'container.__pow__')
        container.__pow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__pow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__pow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__pow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__pow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__pow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__pow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__pow__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pow__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pow__(...)' code ##################

        
        # Call to _rc(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Call to power(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'self' (line 126)
        self_128341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 30), 'self', False)
        # Obtaining the member 'array' of a type (line 126)
        array_128342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 30), self_128341, 'array')
        
        # Call to asarray(...): (line 126)
        # Processing the call arguments (line 126)
        # Getting the type of 'other' (line 126)
        other_128344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 50), 'other', False)
        # Processing the call keyword arguments (line 126)
        kwargs_128345 = {}
        # Getting the type of 'asarray' (line 126)
        asarray_128343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 42), 'asarray', False)
        # Calling asarray(args, kwargs) (line 126)
        asarray_call_result_128346 = invoke(stypy.reporting.localization.Localization(__file__, 126, 42), asarray_128343, *[other_128344], **kwargs_128345)
        
        # Processing the call keyword arguments (line 126)
        kwargs_128347 = {}
        # Getting the type of 'power' (line 126)
        power_128340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 24), 'power', False)
        # Calling power(args, kwargs) (line 126)
        power_call_result_128348 = invoke(stypy.reporting.localization.Localization(__file__, 126, 24), power_128340, *[array_128342, asarray_call_result_128346], **kwargs_128347)
        
        # Processing the call keyword arguments (line 126)
        kwargs_128349 = {}
        # Getting the type of 'self' (line 126)
        self_128338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 126)
        _rc_128339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 15), self_128338, '_rc')
        # Calling _rc(args, kwargs) (line 126)
        _rc_call_result_128350 = invoke(stypy.reporting.localization.Localization(__file__, 126, 15), _rc_128339, *[power_call_result_128348], **kwargs_128349)
        
        # Assigning a type to the variable 'stypy_return_type' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'stypy_return_type', _rc_call_result_128350)
        
        # ################# End of '__pow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pow__' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_128351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128351)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pow__'
        return stypy_return_type_128351


    @norecursion
    def __rpow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rpow__'
        module_type_store = module_type_store.open_function_context('__rpow__', 128, 4, False)
        # Assigning a type to the variable 'self' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rpow__.__dict__.__setitem__('stypy_localization', localization)
        container.__rpow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rpow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rpow__.__dict__.__setitem__('stypy_function_name', 'container.__rpow__')
        container.__rpow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rpow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rpow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rpow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rpow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rpow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rpow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rpow__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rpow__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rpow__(...)' code ##################

        
        # Call to _rc(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to power(...): (line 129)
        # Processing the call arguments (line 129)
        
        # Call to asarray(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'other' (line 129)
        other_128356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 38), 'other', False)
        # Processing the call keyword arguments (line 129)
        kwargs_128357 = {}
        # Getting the type of 'asarray' (line 129)
        asarray_128355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'asarray', False)
        # Calling asarray(args, kwargs) (line 129)
        asarray_call_result_128358 = invoke(stypy.reporting.localization.Localization(__file__, 129, 30), asarray_128355, *[other_128356], **kwargs_128357)
        
        # Getting the type of 'self' (line 129)
        self_128359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 46), 'self', False)
        # Obtaining the member 'array' of a type (line 129)
        array_128360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 46), self_128359, 'array')
        # Processing the call keyword arguments (line 129)
        kwargs_128361 = {}
        # Getting the type of 'power' (line 129)
        power_128354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 24), 'power', False)
        # Calling power(args, kwargs) (line 129)
        power_call_result_128362 = invoke(stypy.reporting.localization.Localization(__file__, 129, 24), power_128354, *[asarray_call_result_128358, array_128360], **kwargs_128361)
        
        # Processing the call keyword arguments (line 129)
        kwargs_128363 = {}
        # Getting the type of 'self' (line 129)
        self_128352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 129)
        _rc_128353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 15), self_128352, '_rc')
        # Calling _rc(args, kwargs) (line 129)
        _rc_call_result_128364 = invoke(stypy.reporting.localization.Localization(__file__, 129, 15), _rc_128353, *[power_call_result_128362], **kwargs_128363)
        
        # Assigning a type to the variable 'stypy_return_type' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'stypy_return_type', _rc_call_result_128364)
        
        # ################# End of '__rpow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rpow__' in the type store
        # Getting the type of 'stypy_return_type' (line 128)
        stypy_return_type_128365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128365)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rpow__'
        return stypy_return_type_128365


    @norecursion
    def __ipow__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ipow__'
        module_type_store = module_type_store.open_function_context('__ipow__', 131, 4, False)
        # Assigning a type to the variable 'self' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__ipow__.__dict__.__setitem__('stypy_localization', localization)
        container.__ipow__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__ipow__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__ipow__.__dict__.__setitem__('stypy_function_name', 'container.__ipow__')
        container.__ipow__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__ipow__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__ipow__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__ipow__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__ipow__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__ipow__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__ipow__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__ipow__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ipow__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ipow__(...)' code ##################

        
        # Call to power(...): (line 132)
        # Processing the call arguments (line 132)
        # Getting the type of 'self' (line 132)
        self_128367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 14), 'self', False)
        # Obtaining the member 'array' of a type (line 132)
        array_128368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 14), self_128367, 'array')
        # Getting the type of 'other' (line 132)
        other_128369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 26), 'other', False)
        # Getting the type of 'self' (line 132)
        self_128370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 33), 'self', False)
        # Obtaining the member 'array' of a type (line 132)
        array_128371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 33), self_128370, 'array')
        # Processing the call keyword arguments (line 132)
        kwargs_128372 = {}
        # Getting the type of 'power' (line 132)
        power_128366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'power', False)
        # Calling power(args, kwargs) (line 132)
        power_call_result_128373 = invoke(stypy.reporting.localization.Localization(__file__, 132, 8), power_128366, *[array_128368, other_128369, array_128371], **kwargs_128372)
        
        # Getting the type of 'self' (line 133)
        self_128374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'stypy_return_type', self_128374)
        
        # ################# End of '__ipow__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ipow__' in the type store
        # Getting the type of 'stypy_return_type' (line 131)
        stypy_return_type_128375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128375)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ipow__'
        return stypy_return_type_128375


    @norecursion
    def __lshift__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lshift__'
        module_type_store = module_type_store.open_function_context('__lshift__', 135, 4, False)
        # Assigning a type to the variable 'self' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__lshift__.__dict__.__setitem__('stypy_localization', localization)
        container.__lshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__lshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__lshift__.__dict__.__setitem__('stypy_function_name', 'container.__lshift__')
        container.__lshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__lshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__lshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__lshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__lshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__lshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__lshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__lshift__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__lshift__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__lshift__(...)' code ##################

        
        # Call to _rc(...): (line 136)
        # Processing the call arguments (line 136)
        
        # Call to left_shift(...): (line 136)
        # Processing the call arguments (line 136)
        # Getting the type of 'self' (line 136)
        self_128379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 35), 'self', False)
        # Obtaining the member 'array' of a type (line 136)
        array_128380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 35), self_128379, 'array')
        # Getting the type of 'other' (line 136)
        other_128381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 47), 'other', False)
        # Processing the call keyword arguments (line 136)
        kwargs_128382 = {}
        # Getting the type of 'left_shift' (line 136)
        left_shift_128378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 24), 'left_shift', False)
        # Calling left_shift(args, kwargs) (line 136)
        left_shift_call_result_128383 = invoke(stypy.reporting.localization.Localization(__file__, 136, 24), left_shift_128378, *[array_128380, other_128381], **kwargs_128382)
        
        # Processing the call keyword arguments (line 136)
        kwargs_128384 = {}
        # Getting the type of 'self' (line 136)
        self_128376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 136)
        _rc_128377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 15), self_128376, '_rc')
        # Calling _rc(args, kwargs) (line 136)
        _rc_call_result_128385 = invoke(stypy.reporting.localization.Localization(__file__, 136, 15), _rc_128377, *[left_shift_call_result_128383], **kwargs_128384)
        
        # Assigning a type to the variable 'stypy_return_type' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 8), 'stypy_return_type', _rc_call_result_128385)
        
        # ################# End of '__lshift__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lshift__' in the type store
        # Getting the type of 'stypy_return_type' (line 135)
        stypy_return_type_128386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128386)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lshift__'
        return stypy_return_type_128386


    @norecursion
    def __rshift__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rshift__'
        module_type_store = module_type_store.open_function_context('__rshift__', 138, 4, False)
        # Assigning a type to the variable 'self' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rshift__.__dict__.__setitem__('stypy_localization', localization)
        container.__rshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rshift__.__dict__.__setitem__('stypy_function_name', 'container.__rshift__')
        container.__rshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rshift__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rshift__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rshift__(...)' code ##################

        
        # Call to _rc(...): (line 139)
        # Processing the call arguments (line 139)
        
        # Call to right_shift(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'self' (line 139)
        self_128390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 36), 'self', False)
        # Obtaining the member 'array' of a type (line 139)
        array_128391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 36), self_128390, 'array')
        # Getting the type of 'other' (line 139)
        other_128392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 48), 'other', False)
        # Processing the call keyword arguments (line 139)
        kwargs_128393 = {}
        # Getting the type of 'right_shift' (line 139)
        right_shift_128389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'right_shift', False)
        # Calling right_shift(args, kwargs) (line 139)
        right_shift_call_result_128394 = invoke(stypy.reporting.localization.Localization(__file__, 139, 24), right_shift_128389, *[array_128391, other_128392], **kwargs_128393)
        
        # Processing the call keyword arguments (line 139)
        kwargs_128395 = {}
        # Getting the type of 'self' (line 139)
        self_128387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 139)
        _rc_128388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), self_128387, '_rc')
        # Calling _rc(args, kwargs) (line 139)
        _rc_call_result_128396 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), _rc_128388, *[right_shift_call_result_128394], **kwargs_128395)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', _rc_call_result_128396)
        
        # ################# End of '__rshift__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rshift__' in the type store
        # Getting the type of 'stypy_return_type' (line 138)
        stypy_return_type_128397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128397)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rshift__'
        return stypy_return_type_128397


    @norecursion
    def __rlshift__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rlshift__'
        module_type_store = module_type_store.open_function_context('__rlshift__', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rlshift__.__dict__.__setitem__('stypy_localization', localization)
        container.__rlshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rlshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rlshift__.__dict__.__setitem__('stypy_function_name', 'container.__rlshift__')
        container.__rlshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rlshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rlshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rlshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rlshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rlshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rlshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rlshift__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rlshift__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rlshift__(...)' code ##################

        
        # Call to _rc(...): (line 142)
        # Processing the call arguments (line 142)
        
        # Call to left_shift(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'other' (line 142)
        other_128401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 35), 'other', False)
        # Getting the type of 'self' (line 142)
        self_128402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 42), 'self', False)
        # Obtaining the member 'array' of a type (line 142)
        array_128403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 42), self_128402, 'array')
        # Processing the call keyword arguments (line 142)
        kwargs_128404 = {}
        # Getting the type of 'left_shift' (line 142)
        left_shift_128400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'left_shift', False)
        # Calling left_shift(args, kwargs) (line 142)
        left_shift_call_result_128405 = invoke(stypy.reporting.localization.Localization(__file__, 142, 24), left_shift_128400, *[other_128401, array_128403], **kwargs_128404)
        
        # Processing the call keyword arguments (line 142)
        kwargs_128406 = {}
        # Getting the type of 'self' (line 142)
        self_128398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 142)
        _rc_128399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 15), self_128398, '_rc')
        # Calling _rc(args, kwargs) (line 142)
        _rc_call_result_128407 = invoke(stypy.reporting.localization.Localization(__file__, 142, 15), _rc_128399, *[left_shift_call_result_128405], **kwargs_128406)
        
        # Assigning a type to the variable 'stypy_return_type' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'stypy_return_type', _rc_call_result_128407)
        
        # ################# End of '__rlshift__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rlshift__' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_128408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128408)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rlshift__'
        return stypy_return_type_128408


    @norecursion
    def __rrshift__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rrshift__'
        module_type_store = module_type_store.open_function_context('__rrshift__', 144, 4, False)
        # Assigning a type to the variable 'self' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rrshift__.__dict__.__setitem__('stypy_localization', localization)
        container.__rrshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rrshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rrshift__.__dict__.__setitem__('stypy_function_name', 'container.__rrshift__')
        container.__rrshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rrshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rrshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rrshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rrshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rrshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rrshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rrshift__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rrshift__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rrshift__(...)' code ##################

        
        # Call to _rc(...): (line 145)
        # Processing the call arguments (line 145)
        
        # Call to right_shift(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'other' (line 145)
        other_128412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 36), 'other', False)
        # Getting the type of 'self' (line 145)
        self_128413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 43), 'self', False)
        # Obtaining the member 'array' of a type (line 145)
        array_128414 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 43), self_128413, 'array')
        # Processing the call keyword arguments (line 145)
        kwargs_128415 = {}
        # Getting the type of 'right_shift' (line 145)
        right_shift_128411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 24), 'right_shift', False)
        # Calling right_shift(args, kwargs) (line 145)
        right_shift_call_result_128416 = invoke(stypy.reporting.localization.Localization(__file__, 145, 24), right_shift_128411, *[other_128412, array_128414], **kwargs_128415)
        
        # Processing the call keyword arguments (line 145)
        kwargs_128417 = {}
        # Getting the type of 'self' (line 145)
        self_128409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 145)
        _rc_128410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 15), self_128409, '_rc')
        # Calling _rc(args, kwargs) (line 145)
        _rc_call_result_128418 = invoke(stypy.reporting.localization.Localization(__file__, 145, 15), _rc_128410, *[right_shift_call_result_128416], **kwargs_128417)
        
        # Assigning a type to the variable 'stypy_return_type' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'stypy_return_type', _rc_call_result_128418)
        
        # ################# End of '__rrshift__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rrshift__' in the type store
        # Getting the type of 'stypy_return_type' (line 144)
        stypy_return_type_128419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128419)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rrshift__'
        return stypy_return_type_128419


    @norecursion
    def __ilshift__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ilshift__'
        module_type_store = module_type_store.open_function_context('__ilshift__', 147, 4, False)
        # Assigning a type to the variable 'self' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__ilshift__.__dict__.__setitem__('stypy_localization', localization)
        container.__ilshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__ilshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__ilshift__.__dict__.__setitem__('stypy_function_name', 'container.__ilshift__')
        container.__ilshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__ilshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__ilshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__ilshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__ilshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__ilshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__ilshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__ilshift__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ilshift__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ilshift__(...)' code ##################

        
        # Call to left_shift(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'self' (line 148)
        self_128421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'self', False)
        # Obtaining the member 'array' of a type (line 148)
        array_128422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), self_128421, 'array')
        # Getting the type of 'other' (line 148)
        other_128423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 31), 'other', False)
        # Getting the type of 'self' (line 148)
        self_128424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 38), 'self', False)
        # Obtaining the member 'array' of a type (line 148)
        array_128425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 38), self_128424, 'array')
        # Processing the call keyword arguments (line 148)
        kwargs_128426 = {}
        # Getting the type of 'left_shift' (line 148)
        left_shift_128420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'left_shift', False)
        # Calling left_shift(args, kwargs) (line 148)
        left_shift_call_result_128427 = invoke(stypy.reporting.localization.Localization(__file__, 148, 8), left_shift_128420, *[array_128422, other_128423, array_128425], **kwargs_128426)
        
        # Getting the type of 'self' (line 149)
        self_128428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', self_128428)
        
        # ################# End of '__ilshift__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ilshift__' in the type store
        # Getting the type of 'stypy_return_type' (line 147)
        stypy_return_type_128429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128429)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ilshift__'
        return stypy_return_type_128429


    @norecursion
    def __irshift__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__irshift__'
        module_type_store = module_type_store.open_function_context('__irshift__', 151, 4, False)
        # Assigning a type to the variable 'self' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__irshift__.__dict__.__setitem__('stypy_localization', localization)
        container.__irshift__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__irshift__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__irshift__.__dict__.__setitem__('stypy_function_name', 'container.__irshift__')
        container.__irshift__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__irshift__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__irshift__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__irshift__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__irshift__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__irshift__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__irshift__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__irshift__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__irshift__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__irshift__(...)' code ##################

        
        # Call to right_shift(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'self' (line 152)
        self_128431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'self', False)
        # Obtaining the member 'array' of a type (line 152)
        array_128432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 20), self_128431, 'array')
        # Getting the type of 'other' (line 152)
        other_128433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 32), 'other', False)
        # Getting the type of 'self' (line 152)
        self_128434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 39), 'self', False)
        # Obtaining the member 'array' of a type (line 152)
        array_128435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 39), self_128434, 'array')
        # Processing the call keyword arguments (line 152)
        kwargs_128436 = {}
        # Getting the type of 'right_shift' (line 152)
        right_shift_128430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'right_shift', False)
        # Calling right_shift(args, kwargs) (line 152)
        right_shift_call_result_128437 = invoke(stypy.reporting.localization.Localization(__file__, 152, 8), right_shift_128430, *[array_128432, other_128433, array_128435], **kwargs_128436)
        
        # Getting the type of 'self' (line 153)
        self_128438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', self_128438)
        
        # ################# End of '__irshift__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__irshift__' in the type store
        # Getting the type of 'stypy_return_type' (line 151)
        stypy_return_type_128439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128439)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__irshift__'
        return stypy_return_type_128439


    @norecursion
    def __and__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__and__'
        module_type_store = module_type_store.open_function_context('__and__', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__and__.__dict__.__setitem__('stypy_localization', localization)
        container.__and__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__and__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__and__.__dict__.__setitem__('stypy_function_name', 'container.__and__')
        container.__and__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__and__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__and__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__and__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__and__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__and__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__and__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__and__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__and__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__and__(...)' code ##################

        
        # Call to _rc(...): (line 156)
        # Processing the call arguments (line 156)
        
        # Call to bitwise_and(...): (line 156)
        # Processing the call arguments (line 156)
        # Getting the type of 'self' (line 156)
        self_128443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 36), 'self', False)
        # Obtaining the member 'array' of a type (line 156)
        array_128444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 36), self_128443, 'array')
        # Getting the type of 'other' (line 156)
        other_128445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 48), 'other', False)
        # Processing the call keyword arguments (line 156)
        kwargs_128446 = {}
        # Getting the type of 'bitwise_and' (line 156)
        bitwise_and_128442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'bitwise_and', False)
        # Calling bitwise_and(args, kwargs) (line 156)
        bitwise_and_call_result_128447 = invoke(stypy.reporting.localization.Localization(__file__, 156, 24), bitwise_and_128442, *[array_128444, other_128445], **kwargs_128446)
        
        # Processing the call keyword arguments (line 156)
        kwargs_128448 = {}
        # Getting the type of 'self' (line 156)
        self_128440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 156)
        _rc_128441 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 15), self_128440, '_rc')
        # Calling _rc(args, kwargs) (line 156)
        _rc_call_result_128449 = invoke(stypy.reporting.localization.Localization(__file__, 156, 15), _rc_128441, *[bitwise_and_call_result_128447], **kwargs_128448)
        
        # Assigning a type to the variable 'stypy_return_type' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 8), 'stypy_return_type', _rc_call_result_128449)
        
        # ################# End of '__and__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__and__' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_128450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128450)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__and__'
        return stypy_return_type_128450


    @norecursion
    def __rand__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rand__'
        module_type_store = module_type_store.open_function_context('__rand__', 158, 4, False)
        # Assigning a type to the variable 'self' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rand__.__dict__.__setitem__('stypy_localization', localization)
        container.__rand__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rand__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rand__.__dict__.__setitem__('stypy_function_name', 'container.__rand__')
        container.__rand__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rand__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rand__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rand__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rand__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rand__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rand__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rand__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rand__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rand__(...)' code ##################

        
        # Call to _rc(...): (line 159)
        # Processing the call arguments (line 159)
        
        # Call to bitwise_and(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'other' (line 159)
        other_128454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 36), 'other', False)
        # Getting the type of 'self' (line 159)
        self_128455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 43), 'self', False)
        # Obtaining the member 'array' of a type (line 159)
        array_128456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 43), self_128455, 'array')
        # Processing the call keyword arguments (line 159)
        kwargs_128457 = {}
        # Getting the type of 'bitwise_and' (line 159)
        bitwise_and_128453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 24), 'bitwise_and', False)
        # Calling bitwise_and(args, kwargs) (line 159)
        bitwise_and_call_result_128458 = invoke(stypy.reporting.localization.Localization(__file__, 159, 24), bitwise_and_128453, *[other_128454, array_128456], **kwargs_128457)
        
        # Processing the call keyword arguments (line 159)
        kwargs_128459 = {}
        # Getting the type of 'self' (line 159)
        self_128451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 159)
        _rc_128452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), self_128451, '_rc')
        # Calling _rc(args, kwargs) (line 159)
        _rc_call_result_128460 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), _rc_128452, *[bitwise_and_call_result_128458], **kwargs_128459)
        
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', _rc_call_result_128460)
        
        # ################# End of '__rand__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rand__' in the type store
        # Getting the type of 'stypy_return_type' (line 158)
        stypy_return_type_128461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128461)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rand__'
        return stypy_return_type_128461


    @norecursion
    def __iand__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iand__'
        module_type_store = module_type_store.open_function_context('__iand__', 161, 4, False)
        # Assigning a type to the variable 'self' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__iand__.__dict__.__setitem__('stypy_localization', localization)
        container.__iand__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__iand__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__iand__.__dict__.__setitem__('stypy_function_name', 'container.__iand__')
        container.__iand__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__iand__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__iand__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__iand__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__iand__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__iand__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__iand__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__iand__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iand__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iand__(...)' code ##################

        
        # Call to bitwise_and(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'self' (line 162)
        self_128463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 20), 'self', False)
        # Obtaining the member 'array' of a type (line 162)
        array_128464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 20), self_128463, 'array')
        # Getting the type of 'other' (line 162)
        other_128465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 32), 'other', False)
        # Getting the type of 'self' (line 162)
        self_128466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 39), 'self', False)
        # Obtaining the member 'array' of a type (line 162)
        array_128467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 39), self_128466, 'array')
        # Processing the call keyword arguments (line 162)
        kwargs_128468 = {}
        # Getting the type of 'bitwise_and' (line 162)
        bitwise_and_128462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'bitwise_and', False)
        # Calling bitwise_and(args, kwargs) (line 162)
        bitwise_and_call_result_128469 = invoke(stypy.reporting.localization.Localization(__file__, 162, 8), bitwise_and_128462, *[array_128464, other_128465, array_128467], **kwargs_128468)
        
        # Getting the type of 'self' (line 163)
        self_128470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 163)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 8), 'stypy_return_type', self_128470)
        
        # ################# End of '__iand__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iand__' in the type store
        # Getting the type of 'stypy_return_type' (line 161)
        stypy_return_type_128471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128471)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iand__'
        return stypy_return_type_128471


    @norecursion
    def __xor__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__xor__'
        module_type_store = module_type_store.open_function_context('__xor__', 165, 4, False)
        # Assigning a type to the variable 'self' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__xor__.__dict__.__setitem__('stypy_localization', localization)
        container.__xor__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__xor__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__xor__.__dict__.__setitem__('stypy_function_name', 'container.__xor__')
        container.__xor__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__xor__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__xor__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__xor__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__xor__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__xor__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__xor__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__xor__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__xor__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__xor__(...)' code ##################

        
        # Call to _rc(...): (line 166)
        # Processing the call arguments (line 166)
        
        # Call to bitwise_xor(...): (line 166)
        # Processing the call arguments (line 166)
        # Getting the type of 'self' (line 166)
        self_128475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 36), 'self', False)
        # Obtaining the member 'array' of a type (line 166)
        array_128476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 36), self_128475, 'array')
        # Getting the type of 'other' (line 166)
        other_128477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 48), 'other', False)
        # Processing the call keyword arguments (line 166)
        kwargs_128478 = {}
        # Getting the type of 'bitwise_xor' (line 166)
        bitwise_xor_128474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 24), 'bitwise_xor', False)
        # Calling bitwise_xor(args, kwargs) (line 166)
        bitwise_xor_call_result_128479 = invoke(stypy.reporting.localization.Localization(__file__, 166, 24), bitwise_xor_128474, *[array_128476, other_128477], **kwargs_128478)
        
        # Processing the call keyword arguments (line 166)
        kwargs_128480 = {}
        # Getting the type of 'self' (line 166)
        self_128472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 166)
        _rc_128473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 166, 15), self_128472, '_rc')
        # Calling _rc(args, kwargs) (line 166)
        _rc_call_result_128481 = invoke(stypy.reporting.localization.Localization(__file__, 166, 15), _rc_128473, *[bitwise_xor_call_result_128479], **kwargs_128480)
        
        # Assigning a type to the variable 'stypy_return_type' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'stypy_return_type', _rc_call_result_128481)
        
        # ################# End of '__xor__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__xor__' in the type store
        # Getting the type of 'stypy_return_type' (line 165)
        stypy_return_type_128482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128482)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__xor__'
        return stypy_return_type_128482


    @norecursion
    def __rxor__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__rxor__'
        module_type_store = module_type_store.open_function_context('__rxor__', 168, 4, False)
        # Assigning a type to the variable 'self' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__rxor__.__dict__.__setitem__('stypy_localization', localization)
        container.__rxor__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__rxor__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__rxor__.__dict__.__setitem__('stypy_function_name', 'container.__rxor__')
        container.__rxor__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__rxor__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__rxor__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__rxor__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__rxor__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__rxor__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__rxor__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__rxor__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__rxor__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__rxor__(...)' code ##################

        
        # Call to _rc(...): (line 169)
        # Processing the call arguments (line 169)
        
        # Call to bitwise_xor(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'other' (line 169)
        other_128486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 36), 'other', False)
        # Getting the type of 'self' (line 169)
        self_128487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), 'self', False)
        # Obtaining the member 'array' of a type (line 169)
        array_128488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 43), self_128487, 'array')
        # Processing the call keyword arguments (line 169)
        kwargs_128489 = {}
        # Getting the type of 'bitwise_xor' (line 169)
        bitwise_xor_128485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'bitwise_xor', False)
        # Calling bitwise_xor(args, kwargs) (line 169)
        bitwise_xor_call_result_128490 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), bitwise_xor_128485, *[other_128486, array_128488], **kwargs_128489)
        
        # Processing the call keyword arguments (line 169)
        kwargs_128491 = {}
        # Getting the type of 'self' (line 169)
        self_128483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 169)
        _rc_128484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 15), self_128483, '_rc')
        # Calling _rc(args, kwargs) (line 169)
        _rc_call_result_128492 = invoke(stypy.reporting.localization.Localization(__file__, 169, 15), _rc_128484, *[bitwise_xor_call_result_128490], **kwargs_128491)
        
        # Assigning a type to the variable 'stypy_return_type' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'stypy_return_type', _rc_call_result_128492)
        
        # ################# End of '__rxor__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__rxor__' in the type store
        # Getting the type of 'stypy_return_type' (line 168)
        stypy_return_type_128493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128493)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__rxor__'
        return stypy_return_type_128493


    @norecursion
    def __ixor__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ixor__'
        module_type_store = module_type_store.open_function_context('__ixor__', 171, 4, False)
        # Assigning a type to the variable 'self' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__ixor__.__dict__.__setitem__('stypy_localization', localization)
        container.__ixor__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__ixor__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__ixor__.__dict__.__setitem__('stypy_function_name', 'container.__ixor__')
        container.__ixor__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__ixor__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__ixor__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__ixor__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__ixor__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__ixor__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__ixor__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__ixor__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ixor__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ixor__(...)' code ##################

        
        # Call to bitwise_xor(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'self' (line 172)
        self_128495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), 'self', False)
        # Obtaining the member 'array' of a type (line 172)
        array_128496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 20), self_128495, 'array')
        # Getting the type of 'other' (line 172)
        other_128497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 32), 'other', False)
        # Getting the type of 'self' (line 172)
        self_128498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 39), 'self', False)
        # Obtaining the member 'array' of a type (line 172)
        array_128499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 39), self_128498, 'array')
        # Processing the call keyword arguments (line 172)
        kwargs_128500 = {}
        # Getting the type of 'bitwise_xor' (line 172)
        bitwise_xor_128494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'bitwise_xor', False)
        # Calling bitwise_xor(args, kwargs) (line 172)
        bitwise_xor_call_result_128501 = invoke(stypy.reporting.localization.Localization(__file__, 172, 8), bitwise_xor_128494, *[array_128496, other_128497, array_128499], **kwargs_128500)
        
        # Getting the type of 'self' (line 173)
        self_128502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'stypy_return_type', self_128502)
        
        # ################# End of '__ixor__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ixor__' in the type store
        # Getting the type of 'stypy_return_type' (line 171)
        stypy_return_type_128503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ixor__'
        return stypy_return_type_128503


    @norecursion
    def __or__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__or__'
        module_type_store = module_type_store.open_function_context('__or__', 175, 4, False)
        # Assigning a type to the variable 'self' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__or__.__dict__.__setitem__('stypy_localization', localization)
        container.__or__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__or__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__or__.__dict__.__setitem__('stypy_function_name', 'container.__or__')
        container.__or__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__or__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__or__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__or__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__or__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__or__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__or__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__or__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__or__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__or__(...)' code ##################

        
        # Call to _rc(...): (line 176)
        # Processing the call arguments (line 176)
        
        # Call to bitwise_or(...): (line 176)
        # Processing the call arguments (line 176)
        # Getting the type of 'self' (line 176)
        self_128507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 35), 'self', False)
        # Obtaining the member 'array' of a type (line 176)
        array_128508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 35), self_128507, 'array')
        # Getting the type of 'other' (line 176)
        other_128509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 47), 'other', False)
        # Processing the call keyword arguments (line 176)
        kwargs_128510 = {}
        # Getting the type of 'bitwise_or' (line 176)
        bitwise_or_128506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 24), 'bitwise_or', False)
        # Calling bitwise_or(args, kwargs) (line 176)
        bitwise_or_call_result_128511 = invoke(stypy.reporting.localization.Localization(__file__, 176, 24), bitwise_or_128506, *[array_128508, other_128509], **kwargs_128510)
        
        # Processing the call keyword arguments (line 176)
        kwargs_128512 = {}
        # Getting the type of 'self' (line 176)
        self_128504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 176)
        _rc_128505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), self_128504, '_rc')
        # Calling _rc(args, kwargs) (line 176)
        _rc_call_result_128513 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), _rc_128505, *[bitwise_or_call_result_128511], **kwargs_128512)
        
        # Assigning a type to the variable 'stypy_return_type' (line 176)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'stypy_return_type', _rc_call_result_128513)
        
        # ################# End of '__or__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__or__' in the type store
        # Getting the type of 'stypy_return_type' (line 175)
        stypy_return_type_128514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128514)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__or__'
        return stypy_return_type_128514


    @norecursion
    def __ror__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ror__'
        module_type_store = module_type_store.open_function_context('__ror__', 178, 4, False)
        # Assigning a type to the variable 'self' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__ror__.__dict__.__setitem__('stypy_localization', localization)
        container.__ror__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__ror__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__ror__.__dict__.__setitem__('stypy_function_name', 'container.__ror__')
        container.__ror__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__ror__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__ror__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__ror__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__ror__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__ror__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__ror__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__ror__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ror__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ror__(...)' code ##################

        
        # Call to _rc(...): (line 179)
        # Processing the call arguments (line 179)
        
        # Call to bitwise_or(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'other' (line 179)
        other_128518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 35), 'other', False)
        # Getting the type of 'self' (line 179)
        self_128519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 42), 'self', False)
        # Obtaining the member 'array' of a type (line 179)
        array_128520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 42), self_128519, 'array')
        # Processing the call keyword arguments (line 179)
        kwargs_128521 = {}
        # Getting the type of 'bitwise_or' (line 179)
        bitwise_or_128517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 24), 'bitwise_or', False)
        # Calling bitwise_or(args, kwargs) (line 179)
        bitwise_or_call_result_128522 = invoke(stypy.reporting.localization.Localization(__file__, 179, 24), bitwise_or_128517, *[other_128518, array_128520], **kwargs_128521)
        
        # Processing the call keyword arguments (line 179)
        kwargs_128523 = {}
        # Getting the type of 'self' (line 179)
        self_128515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 179)
        _rc_128516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 15), self_128515, '_rc')
        # Calling _rc(args, kwargs) (line 179)
        _rc_call_result_128524 = invoke(stypy.reporting.localization.Localization(__file__, 179, 15), _rc_128516, *[bitwise_or_call_result_128522], **kwargs_128523)
        
        # Assigning a type to the variable 'stypy_return_type' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'stypy_return_type', _rc_call_result_128524)
        
        # ################# End of '__ror__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ror__' in the type store
        # Getting the type of 'stypy_return_type' (line 178)
        stypy_return_type_128525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128525)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ror__'
        return stypy_return_type_128525


    @norecursion
    def __ior__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ior__'
        module_type_store = module_type_store.open_function_context('__ior__', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__ior__.__dict__.__setitem__('stypy_localization', localization)
        container.__ior__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__ior__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__ior__.__dict__.__setitem__('stypy_function_name', 'container.__ior__')
        container.__ior__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__ior__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__ior__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__ior__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__ior__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__ior__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__ior__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__ior__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ior__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ior__(...)' code ##################

        
        # Call to bitwise_or(...): (line 182)
        # Processing the call arguments (line 182)
        # Getting the type of 'self' (line 182)
        self_128527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 19), 'self', False)
        # Obtaining the member 'array' of a type (line 182)
        array_128528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 19), self_128527, 'array')
        # Getting the type of 'other' (line 182)
        other_128529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 31), 'other', False)
        # Getting the type of 'self' (line 182)
        self_128530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 38), 'self', False)
        # Obtaining the member 'array' of a type (line 182)
        array_128531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 38), self_128530, 'array')
        # Processing the call keyword arguments (line 182)
        kwargs_128532 = {}
        # Getting the type of 'bitwise_or' (line 182)
        bitwise_or_128526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'bitwise_or', False)
        # Calling bitwise_or(args, kwargs) (line 182)
        bitwise_or_call_result_128533 = invoke(stypy.reporting.localization.Localization(__file__, 182, 8), bitwise_or_128526, *[array_128528, other_128529, array_128531], **kwargs_128532)
        
        # Getting the type of 'self' (line 183)
        self_128534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'stypy_return_type', self_128534)
        
        # ################# End of '__ior__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ior__' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_128535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128535)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ior__'
        return stypy_return_type_128535


    @norecursion
    def __pos__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__pos__'
        module_type_store = module_type_store.open_function_context('__pos__', 185, 4, False)
        # Assigning a type to the variable 'self' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__pos__.__dict__.__setitem__('stypy_localization', localization)
        container.__pos__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__pos__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__pos__.__dict__.__setitem__('stypy_function_name', 'container.__pos__')
        container.__pos__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__pos__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__pos__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__pos__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__pos__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__pos__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__pos__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__pos__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__pos__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__pos__(...)' code ##################

        
        # Call to _rc(...): (line 186)
        # Processing the call arguments (line 186)
        # Getting the type of 'self' (line 186)
        self_128538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 24), 'self', False)
        # Obtaining the member 'array' of a type (line 186)
        array_128539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 24), self_128538, 'array')
        # Processing the call keyword arguments (line 186)
        kwargs_128540 = {}
        # Getting the type of 'self' (line 186)
        self_128536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 186)
        _rc_128537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 15), self_128536, '_rc')
        # Calling _rc(args, kwargs) (line 186)
        _rc_call_result_128541 = invoke(stypy.reporting.localization.Localization(__file__, 186, 15), _rc_128537, *[array_128539], **kwargs_128540)
        
        # Assigning a type to the variable 'stypy_return_type' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'stypy_return_type', _rc_call_result_128541)
        
        # ################# End of '__pos__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__pos__' in the type store
        # Getting the type of 'stypy_return_type' (line 185)
        stypy_return_type_128542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128542)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__pos__'
        return stypy_return_type_128542


    @norecursion
    def __invert__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__invert__'
        module_type_store = module_type_store.open_function_context('__invert__', 188, 4, False)
        # Assigning a type to the variable 'self' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__invert__.__dict__.__setitem__('stypy_localization', localization)
        container.__invert__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__invert__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__invert__.__dict__.__setitem__('stypy_function_name', 'container.__invert__')
        container.__invert__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__invert__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__invert__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__invert__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__invert__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__invert__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__invert__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__invert__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__invert__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__invert__(...)' code ##################

        
        # Call to _rc(...): (line 189)
        # Processing the call arguments (line 189)
        
        # Call to invert(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'self' (line 189)
        self_128546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 31), 'self', False)
        # Obtaining the member 'array' of a type (line 189)
        array_128547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 31), self_128546, 'array')
        # Processing the call keyword arguments (line 189)
        kwargs_128548 = {}
        # Getting the type of 'invert' (line 189)
        invert_128545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'invert', False)
        # Calling invert(args, kwargs) (line 189)
        invert_call_result_128549 = invoke(stypy.reporting.localization.Localization(__file__, 189, 24), invert_128545, *[array_128547], **kwargs_128548)
        
        # Processing the call keyword arguments (line 189)
        kwargs_128550 = {}
        # Getting the type of 'self' (line 189)
        self_128543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 189)
        _rc_128544 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 15), self_128543, '_rc')
        # Calling _rc(args, kwargs) (line 189)
        _rc_call_result_128551 = invoke(stypy.reporting.localization.Localization(__file__, 189, 15), _rc_128544, *[invert_call_result_128549], **kwargs_128550)
        
        # Assigning a type to the variable 'stypy_return_type' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'stypy_return_type', _rc_call_result_128551)
        
        # ################# End of '__invert__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__invert__' in the type store
        # Getting the type of 'stypy_return_type' (line 188)
        stypy_return_type_128552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__invert__'
        return stypy_return_type_128552


    @norecursion
    def _scalarfunc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_scalarfunc'
        module_type_store = module_type_store.open_function_context('_scalarfunc', 191, 4, False)
        # Assigning a type to the variable 'self' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container._scalarfunc.__dict__.__setitem__('stypy_localization', localization)
        container._scalarfunc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container._scalarfunc.__dict__.__setitem__('stypy_type_store', module_type_store)
        container._scalarfunc.__dict__.__setitem__('stypy_function_name', 'container._scalarfunc')
        container._scalarfunc.__dict__.__setitem__('stypy_param_names_list', ['func'])
        container._scalarfunc.__dict__.__setitem__('stypy_varargs_param_name', None)
        container._scalarfunc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container._scalarfunc.__dict__.__setitem__('stypy_call_defaults', defaults)
        container._scalarfunc.__dict__.__setitem__('stypy_call_varargs', varargs)
        container._scalarfunc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container._scalarfunc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container._scalarfunc', ['func'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_scalarfunc', localization, ['func'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_scalarfunc(...)' code ##################

        
        
        
        # Call to len(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'self' (line 192)
        self_128554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 15), 'self', False)
        # Obtaining the member 'shape' of a type (line 192)
        shape_128555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 15), self_128554, 'shape')
        # Processing the call keyword arguments (line 192)
        kwargs_128556 = {}
        # Getting the type of 'len' (line 192)
        len_128553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 11), 'len', False)
        # Calling len(args, kwargs) (line 192)
        len_call_result_128557 = invoke(stypy.reporting.localization.Localization(__file__, 192, 11), len_128553, *[shape_128555], **kwargs_128556)
        
        int_128558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 30), 'int')
        # Applying the binary operator '==' (line 192)
        result_eq_128559 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 11), '==', len_call_result_128557, int_128558)
        
        # Testing the type of an if condition (line 192)
        if_condition_128560 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 8), result_eq_128559)
        # Assigning a type to the variable 'if_condition_128560' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 8), 'if_condition_128560', if_condition_128560)
        # SSA begins for if statement (line 192)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to func(...): (line 193)
        # Processing the call arguments (line 193)
        
        # Obtaining the type of the subscript
        int_128562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 29), 'int')
        # Getting the type of 'self' (line 193)
        self_128563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 24), 'self', False)
        # Obtaining the member '__getitem__' of a type (line 193)
        getitem___128564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 24), self_128563, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 193)
        subscript_call_result_128565 = invoke(stypy.reporting.localization.Localization(__file__, 193, 24), getitem___128564, int_128562)
        
        # Processing the call keyword arguments (line 193)
        kwargs_128566 = {}
        # Getting the type of 'func' (line 193)
        func_128561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 19), 'func', False)
        # Calling func(args, kwargs) (line 193)
        func_call_result_128567 = invoke(stypy.reporting.localization.Localization(__file__, 193, 19), func_128561, *[subscript_call_result_128565], **kwargs_128566)
        
        # Assigning a type to the variable 'stypy_return_type' (line 193)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 12), 'stypy_return_type', func_call_result_128567)
        # SSA branch for the else part of an if statement (line 192)
        module_type_store.open_ssa_branch('else')
        
        # Call to TypeError(...): (line 195)
        # Processing the call arguments (line 195)
        str_128569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 16), 'str', 'only rank-0 arrays can be converted to Python scalars.')
        # Processing the call keyword arguments (line 195)
        kwargs_128570 = {}
        # Getting the type of 'TypeError' (line 195)
        TypeError_128568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 18), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 195)
        TypeError_call_result_128571 = invoke(stypy.reporting.localization.Localization(__file__, 195, 18), TypeError_128568, *[str_128569], **kwargs_128570)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 195, 12), TypeError_call_result_128571, 'raise parameter', BaseException)
        # SSA join for if statement (line 192)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_scalarfunc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_scalarfunc' in the type store
        # Getting the type of 'stypy_return_type' (line 191)
        stypy_return_type_128572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128572)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_scalarfunc'
        return stypy_return_type_128572


    @norecursion
    def __complex__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__complex__'
        module_type_store = module_type_store.open_function_context('__complex__', 198, 4, False)
        # Assigning a type to the variable 'self' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__complex__.__dict__.__setitem__('stypy_localization', localization)
        container.__complex__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__complex__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__complex__.__dict__.__setitem__('stypy_function_name', 'container.__complex__')
        container.__complex__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__complex__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__complex__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__complex__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__complex__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__complex__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__complex__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__complex__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__complex__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__complex__(...)' code ##################

        
        # Call to _scalarfunc(...): (line 199)
        # Processing the call arguments (line 199)
        # Getting the type of 'complex' (line 199)
        complex_128575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 32), 'complex', False)
        # Processing the call keyword arguments (line 199)
        kwargs_128576 = {}
        # Getting the type of 'self' (line 199)
        self_128573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 15), 'self', False)
        # Obtaining the member '_scalarfunc' of a type (line 199)
        _scalarfunc_128574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 15), self_128573, '_scalarfunc')
        # Calling _scalarfunc(args, kwargs) (line 199)
        _scalarfunc_call_result_128577 = invoke(stypy.reporting.localization.Localization(__file__, 199, 15), _scalarfunc_128574, *[complex_128575], **kwargs_128576)
        
        # Assigning a type to the variable 'stypy_return_type' (line 199)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 8), 'stypy_return_type', _scalarfunc_call_result_128577)
        
        # ################# End of '__complex__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__complex__' in the type store
        # Getting the type of 'stypy_return_type' (line 198)
        stypy_return_type_128578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128578)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__complex__'
        return stypy_return_type_128578


    @norecursion
    def __float__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__float__'
        module_type_store = module_type_store.open_function_context('__float__', 201, 4, False)
        # Assigning a type to the variable 'self' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__float__.__dict__.__setitem__('stypy_localization', localization)
        container.__float__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__float__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__float__.__dict__.__setitem__('stypy_function_name', 'container.__float__')
        container.__float__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__float__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__float__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__float__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__float__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__float__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__float__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__float__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__float__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__float__(...)' code ##################

        
        # Call to _scalarfunc(...): (line 202)
        # Processing the call arguments (line 202)
        # Getting the type of 'float' (line 202)
        float_128581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 32), 'float', False)
        # Processing the call keyword arguments (line 202)
        kwargs_128582 = {}
        # Getting the type of 'self' (line 202)
        self_128579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'self', False)
        # Obtaining the member '_scalarfunc' of a type (line 202)
        _scalarfunc_128580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 15), self_128579, '_scalarfunc')
        # Calling _scalarfunc(args, kwargs) (line 202)
        _scalarfunc_call_result_128583 = invoke(stypy.reporting.localization.Localization(__file__, 202, 15), _scalarfunc_128580, *[float_128581], **kwargs_128582)
        
        # Assigning a type to the variable 'stypy_return_type' (line 202)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 8), 'stypy_return_type', _scalarfunc_call_result_128583)
        
        # ################# End of '__float__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__float__' in the type store
        # Getting the type of 'stypy_return_type' (line 201)
        stypy_return_type_128584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128584)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__float__'
        return stypy_return_type_128584


    @norecursion
    def __int__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__int__'
        module_type_store = module_type_store.open_function_context('__int__', 204, 4, False)
        # Assigning a type to the variable 'self' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__int__.__dict__.__setitem__('stypy_localization', localization)
        container.__int__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__int__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__int__.__dict__.__setitem__('stypy_function_name', 'container.__int__')
        container.__int__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__int__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__int__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__int__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__int__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__int__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__int__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__int__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__int__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__int__(...)' code ##################

        
        # Call to _scalarfunc(...): (line 205)
        # Processing the call arguments (line 205)
        # Getting the type of 'int' (line 205)
        int_128587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 32), 'int', False)
        # Processing the call keyword arguments (line 205)
        kwargs_128588 = {}
        # Getting the type of 'self' (line 205)
        self_128585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 15), 'self', False)
        # Obtaining the member '_scalarfunc' of a type (line 205)
        _scalarfunc_128586 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 15), self_128585, '_scalarfunc')
        # Calling _scalarfunc(args, kwargs) (line 205)
        _scalarfunc_call_result_128589 = invoke(stypy.reporting.localization.Localization(__file__, 205, 15), _scalarfunc_128586, *[int_128587], **kwargs_128588)
        
        # Assigning a type to the variable 'stypy_return_type' (line 205)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 8), 'stypy_return_type', _scalarfunc_call_result_128589)
        
        # ################# End of '__int__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__int__' in the type store
        # Getting the type of 'stypy_return_type' (line 204)
        stypy_return_type_128590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128590)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__int__'
        return stypy_return_type_128590


    @norecursion
    def __long__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__long__'
        module_type_store = module_type_store.open_function_context('__long__', 207, 4, False)
        # Assigning a type to the variable 'self' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__long__.__dict__.__setitem__('stypy_localization', localization)
        container.__long__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__long__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__long__.__dict__.__setitem__('stypy_function_name', 'container.__long__')
        container.__long__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__long__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__long__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__long__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__long__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__long__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__long__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__long__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__long__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__long__(...)' code ##################

        
        # Call to _scalarfunc(...): (line 208)
        # Processing the call arguments (line 208)
        # Getting the type of 'long' (line 208)
        long_128593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 32), 'long', False)
        # Processing the call keyword arguments (line 208)
        kwargs_128594 = {}
        # Getting the type of 'self' (line 208)
        self_128591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 15), 'self', False)
        # Obtaining the member '_scalarfunc' of a type (line 208)
        _scalarfunc_128592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 15), self_128591, '_scalarfunc')
        # Calling _scalarfunc(args, kwargs) (line 208)
        _scalarfunc_call_result_128595 = invoke(stypy.reporting.localization.Localization(__file__, 208, 15), _scalarfunc_128592, *[long_128593], **kwargs_128594)
        
        # Assigning a type to the variable 'stypy_return_type' (line 208)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'stypy_return_type', _scalarfunc_call_result_128595)
        
        # ################# End of '__long__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__long__' in the type store
        # Getting the type of 'stypy_return_type' (line 207)
        stypy_return_type_128596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128596)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__long__'
        return stypy_return_type_128596


    @norecursion
    def __hex__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__hex__'
        module_type_store = module_type_store.open_function_context('__hex__', 210, 4, False)
        # Assigning a type to the variable 'self' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__hex__.__dict__.__setitem__('stypy_localization', localization)
        container.__hex__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__hex__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__hex__.__dict__.__setitem__('stypy_function_name', 'container.__hex__')
        container.__hex__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__hex__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__hex__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__hex__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__hex__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__hex__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__hex__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__hex__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__hex__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__hex__(...)' code ##################

        
        # Call to _scalarfunc(...): (line 211)
        # Processing the call arguments (line 211)
        # Getting the type of 'hex' (line 211)
        hex_128599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 32), 'hex', False)
        # Processing the call keyword arguments (line 211)
        kwargs_128600 = {}
        # Getting the type of 'self' (line 211)
        self_128597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'self', False)
        # Obtaining the member '_scalarfunc' of a type (line 211)
        _scalarfunc_128598 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 15), self_128597, '_scalarfunc')
        # Calling _scalarfunc(args, kwargs) (line 211)
        _scalarfunc_call_result_128601 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), _scalarfunc_128598, *[hex_128599], **kwargs_128600)
        
        # Assigning a type to the variable 'stypy_return_type' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'stypy_return_type', _scalarfunc_call_result_128601)
        
        # ################# End of '__hex__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__hex__' in the type store
        # Getting the type of 'stypy_return_type' (line 210)
        stypy_return_type_128602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128602)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__hex__'
        return stypy_return_type_128602


    @norecursion
    def __oct__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__oct__'
        module_type_store = module_type_store.open_function_context('__oct__', 213, 4, False)
        # Assigning a type to the variable 'self' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__oct__.__dict__.__setitem__('stypy_localization', localization)
        container.__oct__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__oct__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__oct__.__dict__.__setitem__('stypy_function_name', 'container.__oct__')
        container.__oct__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__oct__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__oct__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__oct__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__oct__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__oct__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__oct__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__oct__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__oct__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__oct__(...)' code ##################

        
        # Call to _scalarfunc(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'oct' (line 214)
        oct_128605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 32), 'oct', False)
        # Processing the call keyword arguments (line 214)
        kwargs_128606 = {}
        # Getting the type of 'self' (line 214)
        self_128603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 15), 'self', False)
        # Obtaining the member '_scalarfunc' of a type (line 214)
        _scalarfunc_128604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 15), self_128603, '_scalarfunc')
        # Calling _scalarfunc(args, kwargs) (line 214)
        _scalarfunc_call_result_128607 = invoke(stypy.reporting.localization.Localization(__file__, 214, 15), _scalarfunc_128604, *[oct_128605], **kwargs_128606)
        
        # Assigning a type to the variable 'stypy_return_type' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'stypy_return_type', _scalarfunc_call_result_128607)
        
        # ################# End of '__oct__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__oct__' in the type store
        # Getting the type of 'stypy_return_type' (line 213)
        stypy_return_type_128608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128608)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__oct__'
        return stypy_return_type_128608


    @norecursion
    def __lt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__lt__'
        module_type_store = module_type_store.open_function_context('__lt__', 216, 4, False)
        # Assigning a type to the variable 'self' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__lt__.__dict__.__setitem__('stypy_localization', localization)
        container.__lt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__lt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__lt__.__dict__.__setitem__('stypy_function_name', 'container.__lt__')
        container.__lt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__lt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__lt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__lt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__lt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__lt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__lt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__lt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__lt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__lt__(...)' code ##################

        
        # Call to _rc(...): (line 217)
        # Processing the call arguments (line 217)
        
        # Call to less(...): (line 217)
        # Processing the call arguments (line 217)
        # Getting the type of 'self' (line 217)
        self_128612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 29), 'self', False)
        # Obtaining the member 'array' of a type (line 217)
        array_128613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 29), self_128612, 'array')
        # Getting the type of 'other' (line 217)
        other_128614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 41), 'other', False)
        # Processing the call keyword arguments (line 217)
        kwargs_128615 = {}
        # Getting the type of 'less' (line 217)
        less_128611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 24), 'less', False)
        # Calling less(args, kwargs) (line 217)
        less_call_result_128616 = invoke(stypy.reporting.localization.Localization(__file__, 217, 24), less_128611, *[array_128613, other_128614], **kwargs_128615)
        
        # Processing the call keyword arguments (line 217)
        kwargs_128617 = {}
        # Getting the type of 'self' (line 217)
        self_128609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 217)
        _rc_128610 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), self_128609, '_rc')
        # Calling _rc(args, kwargs) (line 217)
        _rc_call_result_128618 = invoke(stypy.reporting.localization.Localization(__file__, 217, 15), _rc_128610, *[less_call_result_128616], **kwargs_128617)
        
        # Assigning a type to the variable 'stypy_return_type' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 8), 'stypy_return_type', _rc_call_result_128618)
        
        # ################# End of '__lt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__lt__' in the type store
        # Getting the type of 'stypy_return_type' (line 216)
        stypy_return_type_128619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128619)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__lt__'
        return stypy_return_type_128619


    @norecursion
    def __le__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__le__'
        module_type_store = module_type_store.open_function_context('__le__', 219, 4, False)
        # Assigning a type to the variable 'self' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__le__.__dict__.__setitem__('stypy_localization', localization)
        container.__le__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__le__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__le__.__dict__.__setitem__('stypy_function_name', 'container.__le__')
        container.__le__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__le__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__le__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__le__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__le__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__le__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__le__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__le__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__le__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__le__(...)' code ##################

        
        # Call to _rc(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to less_equal(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'self' (line 220)
        self_128623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 35), 'self', False)
        # Obtaining the member 'array' of a type (line 220)
        array_128624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 35), self_128623, 'array')
        # Getting the type of 'other' (line 220)
        other_128625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 47), 'other', False)
        # Processing the call keyword arguments (line 220)
        kwargs_128626 = {}
        # Getting the type of 'less_equal' (line 220)
        less_equal_128622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 24), 'less_equal', False)
        # Calling less_equal(args, kwargs) (line 220)
        less_equal_call_result_128627 = invoke(stypy.reporting.localization.Localization(__file__, 220, 24), less_equal_128622, *[array_128624, other_128625], **kwargs_128626)
        
        # Processing the call keyword arguments (line 220)
        kwargs_128628 = {}
        # Getting the type of 'self' (line 220)
        self_128620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 220)
        _rc_128621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 15), self_128620, '_rc')
        # Calling _rc(args, kwargs) (line 220)
        _rc_call_result_128629 = invoke(stypy.reporting.localization.Localization(__file__, 220, 15), _rc_128621, *[less_equal_call_result_128627], **kwargs_128628)
        
        # Assigning a type to the variable 'stypy_return_type' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'stypy_return_type', _rc_call_result_128629)
        
        # ################# End of '__le__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__le__' in the type store
        # Getting the type of 'stypy_return_type' (line 219)
        stypy_return_type_128630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128630)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__le__'
        return stypy_return_type_128630


    @norecursion
    def stypy__eq__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__eq__'
        module_type_store = module_type_store.open_function_context('__eq__', 222, 4, False)
        # Assigning a type to the variable 'self' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.stypy__eq__.__dict__.__setitem__('stypy_localization', localization)
        container.stypy__eq__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.stypy__eq__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.stypy__eq__.__dict__.__setitem__('stypy_function_name', 'container.__eq__')
        container.stypy__eq__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.stypy__eq__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.stypy__eq__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.stypy__eq__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.stypy__eq__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.stypy__eq__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.stypy__eq__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__eq__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__eq__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__eq__(...)' code ##################

        
        # Call to _rc(...): (line 223)
        # Processing the call arguments (line 223)
        
        # Call to equal(...): (line 223)
        # Processing the call arguments (line 223)
        # Getting the type of 'self' (line 223)
        self_128634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 30), 'self', False)
        # Obtaining the member 'array' of a type (line 223)
        array_128635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 30), self_128634, 'array')
        # Getting the type of 'other' (line 223)
        other_128636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 42), 'other', False)
        # Processing the call keyword arguments (line 223)
        kwargs_128637 = {}
        # Getting the type of 'equal' (line 223)
        equal_128633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 24), 'equal', False)
        # Calling equal(args, kwargs) (line 223)
        equal_call_result_128638 = invoke(stypy.reporting.localization.Localization(__file__, 223, 24), equal_128633, *[array_128635, other_128636], **kwargs_128637)
        
        # Processing the call keyword arguments (line 223)
        kwargs_128639 = {}
        # Getting the type of 'self' (line 223)
        self_128631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 223)
        _rc_128632 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 15), self_128631, '_rc')
        # Calling _rc(args, kwargs) (line 223)
        _rc_call_result_128640 = invoke(stypy.reporting.localization.Localization(__file__, 223, 15), _rc_128632, *[equal_call_result_128638], **kwargs_128639)
        
        # Assigning a type to the variable 'stypy_return_type' (line 223)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 8), 'stypy_return_type', _rc_call_result_128640)
        
        # ################# End of '__eq__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__eq__' in the type store
        # Getting the type of 'stypy_return_type' (line 222)
        stypy_return_type_128641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128641)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__eq__'
        return stypy_return_type_128641


    @norecursion
    def __ne__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ne__'
        module_type_store = module_type_store.open_function_context('__ne__', 225, 4, False)
        # Assigning a type to the variable 'self' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__ne__.__dict__.__setitem__('stypy_localization', localization)
        container.__ne__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__ne__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__ne__.__dict__.__setitem__('stypy_function_name', 'container.__ne__')
        container.__ne__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__ne__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__ne__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__ne__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__ne__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__ne__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__ne__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__ne__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ne__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ne__(...)' code ##################

        
        # Call to _rc(...): (line 226)
        # Processing the call arguments (line 226)
        
        # Call to not_equal(...): (line 226)
        # Processing the call arguments (line 226)
        # Getting the type of 'self' (line 226)
        self_128645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'self', False)
        # Obtaining the member 'array' of a type (line 226)
        array_128646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 34), self_128645, 'array')
        # Getting the type of 'other' (line 226)
        other_128647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 46), 'other', False)
        # Processing the call keyword arguments (line 226)
        kwargs_128648 = {}
        # Getting the type of 'not_equal' (line 226)
        not_equal_128644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'not_equal', False)
        # Calling not_equal(args, kwargs) (line 226)
        not_equal_call_result_128649 = invoke(stypy.reporting.localization.Localization(__file__, 226, 24), not_equal_128644, *[array_128646, other_128647], **kwargs_128648)
        
        # Processing the call keyword arguments (line 226)
        kwargs_128650 = {}
        # Getting the type of 'self' (line 226)
        self_128642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 226)
        _rc_128643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 15), self_128642, '_rc')
        # Calling _rc(args, kwargs) (line 226)
        _rc_call_result_128651 = invoke(stypy.reporting.localization.Localization(__file__, 226, 15), _rc_128643, *[not_equal_call_result_128649], **kwargs_128650)
        
        # Assigning a type to the variable 'stypy_return_type' (line 226)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'stypy_return_type', _rc_call_result_128651)
        
        # ################# End of '__ne__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ne__' in the type store
        # Getting the type of 'stypy_return_type' (line 225)
        stypy_return_type_128652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128652)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ne__'
        return stypy_return_type_128652


    @norecursion
    def __gt__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__gt__'
        module_type_store = module_type_store.open_function_context('__gt__', 228, 4, False)
        # Assigning a type to the variable 'self' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__gt__.__dict__.__setitem__('stypy_localization', localization)
        container.__gt__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__gt__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__gt__.__dict__.__setitem__('stypy_function_name', 'container.__gt__')
        container.__gt__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__gt__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__gt__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__gt__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__gt__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__gt__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__gt__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__gt__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__gt__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__gt__(...)' code ##################

        
        # Call to _rc(...): (line 229)
        # Processing the call arguments (line 229)
        
        # Call to greater(...): (line 229)
        # Processing the call arguments (line 229)
        # Getting the type of 'self' (line 229)
        self_128656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 32), 'self', False)
        # Obtaining the member 'array' of a type (line 229)
        array_128657 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 32), self_128656, 'array')
        # Getting the type of 'other' (line 229)
        other_128658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 44), 'other', False)
        # Processing the call keyword arguments (line 229)
        kwargs_128659 = {}
        # Getting the type of 'greater' (line 229)
        greater_128655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 24), 'greater', False)
        # Calling greater(args, kwargs) (line 229)
        greater_call_result_128660 = invoke(stypy.reporting.localization.Localization(__file__, 229, 24), greater_128655, *[array_128657, other_128658], **kwargs_128659)
        
        # Processing the call keyword arguments (line 229)
        kwargs_128661 = {}
        # Getting the type of 'self' (line 229)
        self_128653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 229)
        _rc_128654 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 15), self_128653, '_rc')
        # Calling _rc(args, kwargs) (line 229)
        _rc_call_result_128662 = invoke(stypy.reporting.localization.Localization(__file__, 229, 15), _rc_128654, *[greater_call_result_128660], **kwargs_128661)
        
        # Assigning a type to the variable 'stypy_return_type' (line 229)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 8), 'stypy_return_type', _rc_call_result_128662)
        
        # ################# End of '__gt__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__gt__' in the type store
        # Getting the type of 'stypy_return_type' (line 228)
        stypy_return_type_128663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128663)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__gt__'
        return stypy_return_type_128663


    @norecursion
    def __ge__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__ge__'
        module_type_store = module_type_store.open_function_context('__ge__', 231, 4, False)
        # Assigning a type to the variable 'self' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__ge__.__dict__.__setitem__('stypy_localization', localization)
        container.__ge__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__ge__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__ge__.__dict__.__setitem__('stypy_function_name', 'container.__ge__')
        container.__ge__.__dict__.__setitem__('stypy_param_names_list', ['other'])
        container.__ge__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__ge__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__ge__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__ge__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__ge__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__ge__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__ge__', ['other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__ge__', localization, ['other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__ge__(...)' code ##################

        
        # Call to _rc(...): (line 232)
        # Processing the call arguments (line 232)
        
        # Call to greater_equal(...): (line 232)
        # Processing the call arguments (line 232)
        # Getting the type of 'self' (line 232)
        self_128667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 38), 'self', False)
        # Obtaining the member 'array' of a type (line 232)
        array_128668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 38), self_128667, 'array')
        # Getting the type of 'other' (line 232)
        other_128669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 50), 'other', False)
        # Processing the call keyword arguments (line 232)
        kwargs_128670 = {}
        # Getting the type of 'greater_equal' (line 232)
        greater_equal_128666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 24), 'greater_equal', False)
        # Calling greater_equal(args, kwargs) (line 232)
        greater_equal_call_result_128671 = invoke(stypy.reporting.localization.Localization(__file__, 232, 24), greater_equal_128666, *[array_128668, other_128669], **kwargs_128670)
        
        # Processing the call keyword arguments (line 232)
        kwargs_128672 = {}
        # Getting the type of 'self' (line 232)
        self_128664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 232)
        _rc_128665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 232, 15), self_128664, '_rc')
        # Calling _rc(args, kwargs) (line 232)
        _rc_call_result_128673 = invoke(stypy.reporting.localization.Localization(__file__, 232, 15), _rc_128665, *[greater_equal_call_result_128671], **kwargs_128672)
        
        # Assigning a type to the variable 'stypy_return_type' (line 232)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 8), 'stypy_return_type', _rc_call_result_128673)
        
        # ################# End of '__ge__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__ge__' in the type store
        # Getting the type of 'stypy_return_type' (line 231)
        stypy_return_type_128674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128674)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__ge__'
        return stypy_return_type_128674


    @norecursion
    def copy(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'copy'
        module_type_store = module_type_store.open_function_context('copy', 234, 4, False)
        # Assigning a type to the variable 'self' (line 235)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 235, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.copy.__dict__.__setitem__('stypy_localization', localization)
        container.copy.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.copy.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.copy.__dict__.__setitem__('stypy_function_name', 'container.copy')
        container.copy.__dict__.__setitem__('stypy_param_names_list', [])
        container.copy.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.copy.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.copy.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.copy.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.copy.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.copy.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.copy', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'copy', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'copy(...)' code ##################

        str_128675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 235, 8), 'str', '')
        
        # Call to _rc(...): (line 236)
        # Processing the call arguments (line 236)
        
        # Call to copy(...): (line 236)
        # Processing the call keyword arguments (line 236)
        kwargs_128681 = {}
        # Getting the type of 'self' (line 236)
        self_128678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 24), 'self', False)
        # Obtaining the member 'array' of a type (line 236)
        array_128679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 24), self_128678, 'array')
        # Obtaining the member 'copy' of a type (line 236)
        copy_128680 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 24), array_128679, 'copy')
        # Calling copy(args, kwargs) (line 236)
        copy_call_result_128682 = invoke(stypy.reporting.localization.Localization(__file__, 236, 24), copy_128680, *[], **kwargs_128681)
        
        # Processing the call keyword arguments (line 236)
        kwargs_128683 = {}
        # Getting the type of 'self' (line 236)
        self_128676 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 236)
        _rc_128677 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 15), self_128676, '_rc')
        # Calling _rc(args, kwargs) (line 236)
        _rc_call_result_128684 = invoke(stypy.reporting.localization.Localization(__file__, 236, 15), _rc_128677, *[copy_call_result_128682], **kwargs_128683)
        
        # Assigning a type to the variable 'stypy_return_type' (line 236)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'stypy_return_type', _rc_call_result_128684)
        
        # ################# End of 'copy(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'copy' in the type store
        # Getting the type of 'stypy_return_type' (line 234)
        stypy_return_type_128685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128685)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'copy'
        return stypy_return_type_128685


    @norecursion
    def tostring(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tostring'
        module_type_store = module_type_store.open_function_context('tostring', 238, 4, False)
        # Assigning a type to the variable 'self' (line 239)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.tostring.__dict__.__setitem__('stypy_localization', localization)
        container.tostring.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.tostring.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.tostring.__dict__.__setitem__('stypy_function_name', 'container.tostring')
        container.tostring.__dict__.__setitem__('stypy_param_names_list', [])
        container.tostring.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.tostring.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.tostring.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.tostring.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.tostring.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.tostring.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.tostring', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tostring', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tostring(...)' code ##################

        str_128686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 239, 8), 'str', '')
        
        # Call to tostring(...): (line 240)
        # Processing the call keyword arguments (line 240)
        kwargs_128690 = {}
        # Getting the type of 'self' (line 240)
        self_128687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 15), 'self', False)
        # Obtaining the member 'array' of a type (line 240)
        array_128688 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), self_128687, 'array')
        # Obtaining the member 'tostring' of a type (line 240)
        tostring_128689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 15), array_128688, 'tostring')
        # Calling tostring(args, kwargs) (line 240)
        tostring_call_result_128691 = invoke(stypy.reporting.localization.Localization(__file__, 240, 15), tostring_128689, *[], **kwargs_128690)
        
        # Assigning a type to the variable 'stypy_return_type' (line 240)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 240, 8), 'stypy_return_type', tostring_call_result_128691)
        
        # ################# End of 'tostring(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tostring' in the type store
        # Getting the type of 'stypy_return_type' (line 238)
        stypy_return_type_128692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128692)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tostring'
        return stypy_return_type_128692


    @norecursion
    def byteswap(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'byteswap'
        module_type_store = module_type_store.open_function_context('byteswap', 242, 4, False)
        # Assigning a type to the variable 'self' (line 243)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.byteswap.__dict__.__setitem__('stypy_localization', localization)
        container.byteswap.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.byteswap.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.byteswap.__dict__.__setitem__('stypy_function_name', 'container.byteswap')
        container.byteswap.__dict__.__setitem__('stypy_param_names_list', [])
        container.byteswap.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.byteswap.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.byteswap.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.byteswap.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.byteswap.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.byteswap.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.byteswap', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'byteswap', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'byteswap(...)' code ##################

        str_128693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 243, 8), 'str', '')
        
        # Call to _rc(...): (line 244)
        # Processing the call arguments (line 244)
        
        # Call to byteswap(...): (line 244)
        # Processing the call keyword arguments (line 244)
        kwargs_128699 = {}
        # Getting the type of 'self' (line 244)
        self_128696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 24), 'self', False)
        # Obtaining the member 'array' of a type (line 244)
        array_128697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), self_128696, 'array')
        # Obtaining the member 'byteswap' of a type (line 244)
        byteswap_128698 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 24), array_128697, 'byteswap')
        # Calling byteswap(args, kwargs) (line 244)
        byteswap_call_result_128700 = invoke(stypy.reporting.localization.Localization(__file__, 244, 24), byteswap_128698, *[], **kwargs_128699)
        
        # Processing the call keyword arguments (line 244)
        kwargs_128701 = {}
        # Getting the type of 'self' (line 244)
        self_128694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 244, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 244)
        _rc_128695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 244, 15), self_128694, '_rc')
        # Calling _rc(args, kwargs) (line 244)
        _rc_call_result_128702 = invoke(stypy.reporting.localization.Localization(__file__, 244, 15), _rc_128695, *[byteswap_call_result_128700], **kwargs_128701)
        
        # Assigning a type to the variable 'stypy_return_type' (line 244)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 244, 8), 'stypy_return_type', _rc_call_result_128702)
        
        # ################# End of 'byteswap(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'byteswap' in the type store
        # Getting the type of 'stypy_return_type' (line 242)
        stypy_return_type_128703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128703)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'byteswap'
        return stypy_return_type_128703


    @norecursion
    def astype(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'astype'
        module_type_store = module_type_store.open_function_context('astype', 246, 4, False)
        # Assigning a type to the variable 'self' (line 247)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 247, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.astype.__dict__.__setitem__('stypy_localization', localization)
        container.astype.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.astype.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.astype.__dict__.__setitem__('stypy_function_name', 'container.astype')
        container.astype.__dict__.__setitem__('stypy_param_names_list', ['typecode'])
        container.astype.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.astype.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.astype.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.astype.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.astype.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.astype.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.astype', ['typecode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'astype', localization, ['typecode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'astype(...)' code ##################

        str_128704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 247, 8), 'str', '')
        
        # Call to _rc(...): (line 248)
        # Processing the call arguments (line 248)
        
        # Call to astype(...): (line 248)
        # Processing the call arguments (line 248)
        # Getting the type of 'typecode' (line 248)
        typecode_128710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 42), 'typecode', False)
        # Processing the call keyword arguments (line 248)
        kwargs_128711 = {}
        # Getting the type of 'self' (line 248)
        self_128707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 24), 'self', False)
        # Obtaining the member 'array' of a type (line 248)
        array_128708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 24), self_128707, 'array')
        # Obtaining the member 'astype' of a type (line 248)
        astype_128709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 24), array_128708, 'astype')
        # Calling astype(args, kwargs) (line 248)
        astype_call_result_128712 = invoke(stypy.reporting.localization.Localization(__file__, 248, 24), astype_128709, *[typecode_128710], **kwargs_128711)
        
        # Processing the call keyword arguments (line 248)
        kwargs_128713 = {}
        # Getting the type of 'self' (line 248)
        self_128705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 248, 15), 'self', False)
        # Obtaining the member '_rc' of a type (line 248)
        _rc_128706 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 248, 15), self_128705, '_rc')
        # Calling _rc(args, kwargs) (line 248)
        _rc_call_result_128714 = invoke(stypy.reporting.localization.Localization(__file__, 248, 15), _rc_128706, *[astype_call_result_128712], **kwargs_128713)
        
        # Assigning a type to the variable 'stypy_return_type' (line 248)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 248, 8), 'stypy_return_type', _rc_call_result_128714)
        
        # ################# End of 'astype(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'astype' in the type store
        # Getting the type of 'stypy_return_type' (line 246)
        stypy_return_type_128715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 246, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'astype'
        return stypy_return_type_128715


    @norecursion
    def _rc(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_rc'
        module_type_store = module_type_store.open_function_context('_rc', 250, 4, False)
        # Assigning a type to the variable 'self' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container._rc.__dict__.__setitem__('stypy_localization', localization)
        container._rc.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container._rc.__dict__.__setitem__('stypy_type_store', module_type_store)
        container._rc.__dict__.__setitem__('stypy_function_name', 'container._rc')
        container._rc.__dict__.__setitem__('stypy_param_names_list', ['a'])
        container._rc.__dict__.__setitem__('stypy_varargs_param_name', None)
        container._rc.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container._rc.__dict__.__setitem__('stypy_call_defaults', defaults)
        container._rc.__dict__.__setitem__('stypy_call_varargs', varargs)
        container._rc.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container._rc.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container._rc', ['a'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_rc', localization, ['a'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_rc(...)' code ##################

        
        
        
        # Call to len(...): (line 251)
        # Processing the call arguments (line 251)
        
        # Call to shape(...): (line 251)
        # Processing the call arguments (line 251)
        # Getting the type of 'a' (line 251)
        a_128718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 21), 'a', False)
        # Processing the call keyword arguments (line 251)
        kwargs_128719 = {}
        # Getting the type of 'shape' (line 251)
        shape_128717 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 15), 'shape', False)
        # Calling shape(args, kwargs) (line 251)
        shape_call_result_128720 = invoke(stypy.reporting.localization.Localization(__file__, 251, 15), shape_128717, *[a_128718], **kwargs_128719)
        
        # Processing the call keyword arguments (line 251)
        kwargs_128721 = {}
        # Getting the type of 'len' (line 251)
        len_128716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 251, 11), 'len', False)
        # Calling len(args, kwargs) (line 251)
        len_call_result_128722 = invoke(stypy.reporting.localization.Localization(__file__, 251, 11), len_128716, *[shape_call_result_128720], **kwargs_128721)
        
        int_128723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 251, 28), 'int')
        # Applying the binary operator '==' (line 251)
        result_eq_128724 = python_operator(stypy.reporting.localization.Localization(__file__, 251, 11), '==', len_call_result_128722, int_128723)
        
        # Testing the type of an if condition (line 251)
        if_condition_128725 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 251, 8), result_eq_128724)
        # Assigning a type to the variable 'if_condition_128725' (line 251)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 251, 8), 'if_condition_128725', if_condition_128725)
        # SSA begins for if statement (line 251)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'a' (line 252)
        a_128726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 252, 19), 'a')
        # Assigning a type to the variable 'stypy_return_type' (line 252)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 252, 12), 'stypy_return_type', a_128726)
        # SSA branch for the else part of an if statement (line 251)
        module_type_store.open_ssa_branch('else')
        
        # Call to __class__(...): (line 254)
        # Processing the call arguments (line 254)
        # Getting the type of 'a' (line 254)
        a_128729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 34), 'a', False)
        # Processing the call keyword arguments (line 254)
        kwargs_128730 = {}
        # Getting the type of 'self' (line 254)
        self_128727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 254, 19), 'self', False)
        # Obtaining the member '__class__' of a type (line 254)
        class___128728 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 254, 19), self_128727, '__class__')
        # Calling __class__(args, kwargs) (line 254)
        class___call_result_128731 = invoke(stypy.reporting.localization.Localization(__file__, 254, 19), class___128728, *[a_128729], **kwargs_128730)
        
        # Assigning a type to the variable 'stypy_return_type' (line 254)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 254, 12), 'stypy_return_type', class___call_result_128731)
        # SSA join for if statement (line 251)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '_rc(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_rc' in the type store
        # Getting the type of 'stypy_return_type' (line 250)
        stypy_return_type_128732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 250, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128732)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_rc'
        return stypy_return_type_128732


    @norecursion
    def __array_wrap__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array_wrap__'
        module_type_store = module_type_store.open_function_context('__array_wrap__', 256, 4, False)
        # Assigning a type to the variable 'self' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__array_wrap__.__dict__.__setitem__('stypy_localization', localization)
        container.__array_wrap__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__array_wrap__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__array_wrap__.__dict__.__setitem__('stypy_function_name', 'container.__array_wrap__')
        container.__array_wrap__.__dict__.__setitem__('stypy_param_names_list', [])
        container.__array_wrap__.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        container.__array_wrap__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__array_wrap__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__array_wrap__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__array_wrap__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__array_wrap__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__array_wrap__', [], 'args', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array_wrap__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array_wrap__(...)' code ##################

        
        # Call to __class__(...): (line 257)
        # Processing the call arguments (line 257)
        
        # Obtaining the type of the subscript
        int_128735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 257, 35), 'int')
        # Getting the type of 'args' (line 257)
        args_128736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 30), 'args', False)
        # Obtaining the member '__getitem__' of a type (line 257)
        getitem___128737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 30), args_128736, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 257)
        subscript_call_result_128738 = invoke(stypy.reporting.localization.Localization(__file__, 257, 30), getitem___128737, int_128735)
        
        # Processing the call keyword arguments (line 257)
        kwargs_128739 = {}
        # Getting the type of 'self' (line 257)
        self_128733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 257, 15), 'self', False)
        # Obtaining the member '__class__' of a type (line 257)
        class___128734 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 257, 15), self_128733, '__class__')
        # Calling __class__(args, kwargs) (line 257)
        class___call_result_128740 = invoke(stypy.reporting.localization.Localization(__file__, 257, 15), class___128734, *[subscript_call_result_128738], **kwargs_128739)
        
        # Assigning a type to the variable 'stypy_return_type' (line 257)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 257, 8), 'stypy_return_type', class___call_result_128740)
        
        # ################# End of '__array_wrap__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array_wrap__' in the type store
        # Getting the type of 'stypy_return_type' (line 256)
        stypy_return_type_128741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 256, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128741)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array_wrap__'
        return stypy_return_type_128741


    @norecursion
    def __setattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__setattr__'
        module_type_store = module_type_store.open_function_context('__setattr__', 259, 4, False)
        # Assigning a type to the variable 'self' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__setattr__.__dict__.__setitem__('stypy_localization', localization)
        container.__setattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__setattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__setattr__.__dict__.__setitem__('stypy_function_name', 'container.__setattr__')
        container.__setattr__.__dict__.__setitem__('stypy_param_names_list', ['attr', 'value'])
        container.__setattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__setattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__setattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__setattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__setattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__setattr__.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__setattr__', ['attr', 'value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__setattr__', localization, ['attr', 'value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__setattr__(...)' code ##################

        
        
        # Getting the type of 'attr' (line 260)
        attr_128742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 260, 11), 'attr')
        str_128743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 260, 19), 'str', 'array')
        # Applying the binary operator '==' (line 260)
        result_eq_128744 = python_operator(stypy.reporting.localization.Localization(__file__, 260, 11), '==', attr_128742, str_128743)
        
        # Testing the type of an if condition (line 260)
        if_condition_128745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 260, 8), result_eq_128744)
        # Assigning a type to the variable 'if_condition_128745' (line 260)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 260, 8), 'if_condition_128745', if_condition_128745)
        # SSA begins for if statement (line 260)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __setattr__(...): (line 261)
        # Processing the call arguments (line 261)
        # Getting the type of 'self' (line 261)
        self_128748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 31), 'self', False)
        # Getting the type of 'attr' (line 261)
        attr_128749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 37), 'attr', False)
        # Getting the type of 'value' (line 261)
        value_128750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 43), 'value', False)
        # Processing the call keyword arguments (line 261)
        kwargs_128751 = {}
        # Getting the type of 'object' (line 261)
        object_128746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 261, 12), 'object', False)
        # Obtaining the member '__setattr__' of a type (line 261)
        setattr___128747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 261, 12), object_128746, '__setattr__')
        # Calling __setattr__(args, kwargs) (line 261)
        setattr___call_result_128752 = invoke(stypy.reporting.localization.Localization(__file__, 261, 12), setattr___128747, *[self_128748, attr_128749, value_128750], **kwargs_128751)
        
        # Assigning a type to the variable 'stypy_return_type' (line 262)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 262, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 260)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 263)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to __setattr__(...): (line 264)
        # Processing the call arguments (line 264)
        # Getting the type of 'attr' (line 264)
        attr_128756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 35), 'attr', False)
        # Getting the type of 'value' (line 264)
        value_128757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 41), 'value', False)
        # Processing the call keyword arguments (line 264)
        kwargs_128758 = {}
        # Getting the type of 'self' (line 264)
        self_128753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 264, 12), 'self', False)
        # Obtaining the member 'array' of a type (line 264)
        array_128754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), self_128753, 'array')
        # Obtaining the member '__setattr__' of a type (line 264)
        setattr___128755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 264, 12), array_128754, '__setattr__')
        # Calling __setattr__(args, kwargs) (line 264)
        setattr___call_result_128759 = invoke(stypy.reporting.localization.Localization(__file__, 264, 12), setattr___128755, *[attr_128756, value_128757], **kwargs_128758)
        
        # SSA branch for the except part of a try statement (line 263)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 263)
        module_type_store.open_ssa_branch('except')
        
        # Call to __setattr__(...): (line 266)
        # Processing the call arguments (line 266)
        # Getting the type of 'self' (line 266)
        self_128762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 31), 'self', False)
        # Getting the type of 'attr' (line 266)
        attr_128763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 37), 'attr', False)
        # Getting the type of 'value' (line 266)
        value_128764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 43), 'value', False)
        # Processing the call keyword arguments (line 266)
        kwargs_128765 = {}
        # Getting the type of 'object' (line 266)
        object_128760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 266, 12), 'object', False)
        # Obtaining the member '__setattr__' of a type (line 266)
        setattr___128761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 266, 12), object_128760, '__setattr__')
        # Calling __setattr__(args, kwargs) (line 266)
        setattr___call_result_128766 = invoke(stypy.reporting.localization.Localization(__file__, 266, 12), setattr___128761, *[self_128762, attr_128763, value_128764], **kwargs_128765)
        
        # SSA join for try-except statement (line 263)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__setattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__setattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 259)
        stypy_return_type_128767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 259, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128767)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__setattr__'
        return stypy_return_type_128767


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 269, 4, False)
        # Assigning a type to the variable 'self' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        container.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        container.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        container.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        container.__getattr__.__dict__.__setitem__('stypy_function_name', 'container.__getattr__')
        container.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        container.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        container.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        container.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        container.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        container.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        container.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'container.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getattr__', localization, ['attr'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getattr__(...)' code ##################

        
        
        # Getting the type of 'attr' (line 270)
        attr_128768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 270, 12), 'attr')
        str_128769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 270, 20), 'str', 'array')
        # Applying the binary operator '==' (line 270)
        result_eq_128770 = python_operator(stypy.reporting.localization.Localization(__file__, 270, 12), '==', attr_128768, str_128769)
        
        # Testing the type of an if condition (line 270)
        if_condition_128771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 270, 8), result_eq_128770)
        # Assigning a type to the variable 'if_condition_128771' (line 270)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 270, 8), 'if_condition_128771', if_condition_128771)
        # SSA begins for if statement (line 270)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __getattribute__(...): (line 271)
        # Processing the call arguments (line 271)
        # Getting the type of 'self' (line 271)
        self_128774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 43), 'self', False)
        # Getting the type of 'attr' (line 271)
        attr_128775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 49), 'attr', False)
        # Processing the call keyword arguments (line 271)
        kwargs_128776 = {}
        # Getting the type of 'object' (line 271)
        object_128772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 271, 19), 'object', False)
        # Obtaining the member '__getattribute__' of a type (line 271)
        getattribute___128773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 271, 19), object_128772, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 271)
        getattribute___call_result_128777 = invoke(stypy.reporting.localization.Localization(__file__, 271, 19), getattribute___128773, *[self_128774, attr_128775], **kwargs_128776)
        
        # Assigning a type to the variable 'stypy_return_type' (line 271)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 271, 12), 'stypy_return_type', getattribute___call_result_128777)
        # SSA join for if statement (line 270)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to __getattribute__(...): (line 272)
        # Processing the call arguments (line 272)
        # Getting the type of 'attr' (line 272)
        attr_128781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 43), 'attr', False)
        # Processing the call keyword arguments (line 272)
        kwargs_128782 = {}
        # Getting the type of 'self' (line 272)
        self_128778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 272, 15), 'self', False)
        # Obtaining the member 'array' of a type (line 272)
        array_128779 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), self_128778, 'array')
        # Obtaining the member '__getattribute__' of a type (line 272)
        getattribute___128780 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 272, 15), array_128779, '__getattribute__')
        # Calling __getattribute__(args, kwargs) (line 272)
        getattribute___call_result_128783 = invoke(stypy.reporting.localization.Localization(__file__, 272, 15), getattribute___128780, *[attr_128781], **kwargs_128782)
        
        # Assigning a type to the variable 'stypy_return_type' (line 272)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 272, 8), 'stypy_return_type', getattribute___call_result_128783)
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 269)
        stypy_return_type_128784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 269, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_128784)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_128784


# Assigning a type to the variable 'container' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'container', container)

# Assigning a Name to a Name (line 72):
# Getting the type of 'container'
container_128785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'container')
# Obtaining the member '__add__' of a type
add___128786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), container_128785, '__add__')
# Getting the type of 'container'
container_128787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'container')
# Setting the type of the member '__radd__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), container_128787, '__radd__', add___128786)

# Assigning a Name to a Name (line 91):
# Getting the type of 'container'
container_128788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'container')
# Obtaining the member '__mul__' of a type
mul___128789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), container_128788, '__mul__')
# Getting the type of 'container'
container_128790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'container')
# Setting the type of the member '__rmul__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), container_128790, '__rmul__', mul___128789)

if (__name__ == '__main__'):
    
    # Assigning a Call to a Name (line 278):
    
    # Call to reshape(...): (line 278)
    # Processing the call arguments (line 278)
    
    # Call to arange(...): (line 278)
    # Processing the call arguments (line 278)
    int_128793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 26), 'int')
    # Processing the call keyword arguments (line 278)
    kwargs_128794 = {}
    # Getting the type of 'arange' (line 278)
    arange_128792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 19), 'arange', False)
    # Calling arange(args, kwargs) (line 278)
    arange_call_result_128795 = invoke(stypy.reporting.localization.Localization(__file__, 278, 19), arange_128792, *[int_128793], **kwargs_128794)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 278)
    tuple_128796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 278)
    # Adding element type (line 278)
    int_128797 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 35), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 35), tuple_128796, int_128797)
    # Adding element type (line 278)
    int_128798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 278, 40), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 278, 35), tuple_128796, int_128798)
    
    # Processing the call keyword arguments (line 278)
    kwargs_128799 = {}
    # Getting the type of 'reshape' (line 278)
    reshape_128791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 278, 11), 'reshape', False)
    # Calling reshape(args, kwargs) (line 278)
    reshape_call_result_128800 = invoke(stypy.reporting.localization.Localization(__file__, 278, 11), reshape_128791, *[arange_call_result_128795, tuple_128796], **kwargs_128799)
    
    # Assigning a type to the variable 'temp' (line 278)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 278, 4), 'temp', reshape_call_result_128800)
    
    # Assigning a Call to a Name (line 280):
    
    # Call to container(...): (line 280)
    # Processing the call arguments (line 280)
    # Getting the type of 'temp' (line 280)
    temp_128802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 19), 'temp', False)
    # Processing the call keyword arguments (line 280)
    kwargs_128803 = {}
    # Getting the type of 'container' (line 280)
    container_128801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 280, 9), 'container', False)
    # Calling container(args, kwargs) (line 280)
    container_call_result_128804 = invoke(stypy.reporting.localization.Localization(__file__, 280, 9), container_128801, *[temp_128802], **kwargs_128803)
    
    # Assigning a type to the variable 'ua' (line 280)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 280, 4), 'ua', container_call_result_128804)
    
    # Call to print(...): (line 282)
    # Processing the call arguments (line 282)
    
    # Call to dir(...): (line 282)
    # Processing the call arguments (line 282)
    # Getting the type of 'ua' (line 282)
    ua_128807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 14), 'ua', False)
    # Processing the call keyword arguments (line 282)
    kwargs_128808 = {}
    # Getting the type of 'dir' (line 282)
    dir_128806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 10), 'dir', False)
    # Calling dir(args, kwargs) (line 282)
    dir_call_result_128809 = invoke(stypy.reporting.localization.Localization(__file__, 282, 10), dir_128806, *[ua_128807], **kwargs_128808)
    
    # Processing the call keyword arguments (line 282)
    kwargs_128810 = {}
    # Getting the type of 'print' (line 282)
    print_128805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 282, 4), 'print', False)
    # Calling print(args, kwargs) (line 282)
    print_call_result_128811 = invoke(stypy.reporting.localization.Localization(__file__, 282, 4), print_128805, *[dir_call_result_128809], **kwargs_128810)
    
    
    # Call to print(...): (line 283)
    # Processing the call arguments (line 283)
    
    # Call to shape(...): (line 283)
    # Processing the call arguments (line 283)
    # Getting the type of 'ua' (line 283)
    ua_128814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 16), 'ua', False)
    # Processing the call keyword arguments (line 283)
    kwargs_128815 = {}
    # Getting the type of 'shape' (line 283)
    shape_128813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 10), 'shape', False)
    # Calling shape(args, kwargs) (line 283)
    shape_call_result_128816 = invoke(stypy.reporting.localization.Localization(__file__, 283, 10), shape_128813, *[ua_128814], **kwargs_128815)
    
    # Getting the type of 'ua' (line 283)
    ua_128817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 21), 'ua', False)
    # Obtaining the member 'shape' of a type (line 283)
    shape_128818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 283, 21), ua_128817, 'shape')
    # Processing the call keyword arguments (line 283)
    kwargs_128819 = {}
    # Getting the type of 'print' (line 283)
    print_128812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 283, 4), 'print', False)
    # Calling print(args, kwargs) (line 283)
    print_call_result_128820 = invoke(stypy.reporting.localization.Localization(__file__, 283, 4), print_128812, *[shape_call_result_128816, shape_128818], **kwargs_128819)
    
    
    # Assigning a Subscript to a Name (line 285):
    
    # Obtaining the type of the subscript
    int_128821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 19), 'int')
    slice_128822 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 285, 15), None, int_128821, None)
    int_128823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 285, 23), 'int')
    slice_128824 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 285, 15), None, int_128823, None)
    # Getting the type of 'ua' (line 285)
    ua_128825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 285, 15), 'ua')
    # Obtaining the member '__getitem__' of a type (line 285)
    getitem___128826 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 285, 15), ua_128825, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 285)
    subscript_call_result_128827 = invoke(stypy.reporting.localization.Localization(__file__, 285, 15), getitem___128826, (slice_128822, slice_128824))
    
    # Assigning a type to the variable 'ua_small' (line 285)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 285, 4), 'ua_small', subscript_call_result_128827)
    
    # Call to print(...): (line 286)
    # Processing the call arguments (line 286)
    # Getting the type of 'ua_small' (line 286)
    ua_small_128829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 10), 'ua_small', False)
    # Processing the call keyword arguments (line 286)
    kwargs_128830 = {}
    # Getting the type of 'print' (line 286)
    print_128828 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 286, 4), 'print', False)
    # Calling print(args, kwargs) (line 286)
    print_call_result_128831 = invoke(stypy.reporting.localization.Localization(__file__, 286, 4), print_128828, *[ua_small_128829], **kwargs_128830)
    
    
    # Assigning a Num to a Subscript (line 288):
    int_128832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 21), 'int')
    # Getting the type of 'ua_small' (line 288)
    ua_small_128833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 288, 4), 'ua_small')
    
    # Obtaining an instance of the builtin type 'tuple' (line 288)
    tuple_128834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 13), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 288)
    # Adding element type (line 288)
    int_128835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 13), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), tuple_128834, int_128835)
    # Adding element type (line 288)
    int_128836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 288, 16), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 13), tuple_128834, int_128836)
    
    # Storing an element on a container (line 288)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 288, 4), ua_small_128833, (tuple_128834, int_128832))
    
    # Call to print(...): (line 289)
    # Processing the call arguments (line 289)
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 289)
    tuple_128838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 289)
    # Adding element type (line 289)
    int_128839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 19), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 19), tuple_128838, int_128839)
    # Adding element type (line 289)
    int_128840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 22), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 19), tuple_128838, int_128840)
    
    # Getting the type of 'ua_small' (line 289)
    ua_small_128841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 10), 'ua_small', False)
    # Obtaining the member '__getitem__' of a type (line 289)
    getitem___128842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 10), ua_small_128841, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 289)
    subscript_call_result_128843 = invoke(stypy.reporting.localization.Localization(__file__, 289, 10), getitem___128842, tuple_128838)
    
    
    # Obtaining the type of the subscript
    
    # Obtaining an instance of the builtin type 'tuple' (line 289)
    tuple_128844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 289)
    # Adding element type (line 289)
    int_128845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 29), tuple_128844, int_128845)
    # Adding element type (line 289)
    int_128846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 289, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 289, 29), tuple_128844, int_128846)
    
    # Getting the type of 'ua' (line 289)
    ua_128847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 26), 'ua', False)
    # Obtaining the member '__getitem__' of a type (line 289)
    getitem___128848 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 289, 26), ua_128847, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 289)
    subscript_call_result_128849 = invoke(stypy.reporting.localization.Localization(__file__, 289, 26), getitem___128848, tuple_128844)
    
    # Processing the call keyword arguments (line 289)
    kwargs_128850 = {}
    # Getting the type of 'print' (line 289)
    print_128837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 289, 4), 'print', False)
    # Calling print(args, kwargs) (line 289)
    print_call_result_128851 = invoke(stypy.reporting.localization.Localization(__file__, 289, 4), print_128837, *[subscript_call_result_128843, subscript_call_result_128849], **kwargs_128850)
    
    
    # Call to print(...): (line 290)
    # Processing the call arguments (line 290)
    
    # Call to sin(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'ua_small' (line 290)
    ua_small_128854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 14), 'ua_small', False)
    # Processing the call keyword arguments (line 290)
    kwargs_128855 = {}
    # Getting the type of 'sin' (line 290)
    sin_128853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 10), 'sin', False)
    # Calling sin(args, kwargs) (line 290)
    sin_call_result_128856 = invoke(stypy.reporting.localization.Localization(__file__, 290, 10), sin_128853, *[ua_small_128854], **kwargs_128855)
    
    float_128857 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 26), 'float')
    # Applying the binary operator 'div' (line 290)
    result_div_128858 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 10), 'div', sin_call_result_128856, float_128857)
    
    float_128859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 31), 'float')
    # Applying the binary operator '*' (line 290)
    result_mul_128860 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 29), '*', result_div_128858, float_128859)
    
    
    # Call to sqrt(...): (line 290)
    # Processing the call arguments (line 290)
    # Getting the type of 'ua_small' (line 290)
    ua_small_128862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 41), 'ua_small', False)
    int_128863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 290, 53), 'int')
    # Applying the binary operator '**' (line 290)
    result_pow_128864 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 41), '**', ua_small_128862, int_128863)
    
    # Processing the call keyword arguments (line 290)
    kwargs_128865 = {}
    # Getting the type of 'sqrt' (line 290)
    sqrt_128861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 36), 'sqrt', False)
    # Calling sqrt(args, kwargs) (line 290)
    sqrt_call_result_128866 = invoke(stypy.reporting.localization.Localization(__file__, 290, 36), sqrt_128861, *[result_pow_128864], **kwargs_128865)
    
    # Applying the binary operator '+' (line 290)
    result_add_128867 = python_operator(stypy.reporting.localization.Localization(__file__, 290, 10), '+', result_mul_128860, sqrt_call_result_128866)
    
    # Processing the call keyword arguments (line 290)
    kwargs_128868 = {}
    # Getting the type of 'print' (line 290)
    print_128852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 290, 4), 'print', False)
    # Calling print(args, kwargs) (line 290)
    print_call_result_128869 = invoke(stypy.reporting.localization.Localization(__file__, 290, 4), print_128852, *[result_add_128867], **kwargs_128868)
    
    
    # Call to print(...): (line 291)
    # Processing the call arguments (line 291)
    
    # Call to less(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'ua_small' (line 291)
    ua_small_128872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 15), 'ua_small', False)
    int_128873 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 25), 'int')
    # Processing the call keyword arguments (line 291)
    kwargs_128874 = {}
    # Getting the type of 'less' (line 291)
    less_128871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 10), 'less', False)
    # Calling less(args, kwargs) (line 291)
    less_call_result_128875 = invoke(stypy.reporting.localization.Localization(__file__, 291, 10), less_128871, *[ua_small_128872, int_128873], **kwargs_128874)
    
    
    # Call to type(...): (line 291)
    # Processing the call arguments (line 291)
    
    # Call to less(...): (line 291)
    # Processing the call arguments (line 291)
    # Getting the type of 'ua_small' (line 291)
    ua_small_128878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 41), 'ua_small', False)
    int_128879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 291, 51), 'int')
    # Processing the call keyword arguments (line 291)
    kwargs_128880 = {}
    # Getting the type of 'less' (line 291)
    less_128877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 36), 'less', False)
    # Calling less(args, kwargs) (line 291)
    less_call_result_128881 = invoke(stypy.reporting.localization.Localization(__file__, 291, 36), less_128877, *[ua_small_128878, int_128879], **kwargs_128880)
    
    # Processing the call keyword arguments (line 291)
    kwargs_128882 = {}
    # Getting the type of 'type' (line 291)
    type_128876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 31), 'type', False)
    # Calling type(args, kwargs) (line 291)
    type_call_result_128883 = invoke(stypy.reporting.localization.Localization(__file__, 291, 31), type_128876, *[less_call_result_128881], **kwargs_128882)
    
    # Processing the call keyword arguments (line 291)
    kwargs_128884 = {}
    # Getting the type of 'print' (line 291)
    print_128870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 291, 4), 'print', False)
    # Calling print(args, kwargs) (line 291)
    print_call_result_128885 = invoke(stypy.reporting.localization.Localization(__file__, 291, 4), print_128870, *[less_call_result_128875, type_call_result_128883], **kwargs_128884)
    
    
    # Call to print(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Call to type(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'ua_small' (line 292)
    ua_small_128888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 15), 'ua_small', False)
    
    # Call to reshape(...): (line 292)
    # Processing the call arguments (line 292)
    
    # Call to arange(...): (line 292)
    # Processing the call arguments (line 292)
    int_128891 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 292, 41), 'int')
    # Processing the call keyword arguments (line 292)
    kwargs_128892 = {}
    # Getting the type of 'arange' (line 292)
    arange_128890 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 34), 'arange', False)
    # Calling arange(args, kwargs) (line 292)
    arange_call_result_128893 = invoke(stypy.reporting.localization.Localization(__file__, 292, 34), arange_128890, *[int_128891], **kwargs_128892)
    
    
    # Call to shape(...): (line 292)
    # Processing the call arguments (line 292)
    # Getting the type of 'ua_small' (line 292)
    ua_small_128895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 52), 'ua_small', False)
    # Processing the call keyword arguments (line 292)
    kwargs_128896 = {}
    # Getting the type of 'shape' (line 292)
    shape_128894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 46), 'shape', False)
    # Calling shape(args, kwargs) (line 292)
    shape_call_result_128897 = invoke(stypy.reporting.localization.Localization(__file__, 292, 46), shape_128894, *[ua_small_128895], **kwargs_128896)
    
    # Processing the call keyword arguments (line 292)
    kwargs_128898 = {}
    # Getting the type of 'reshape' (line 292)
    reshape_128889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 26), 'reshape', False)
    # Calling reshape(args, kwargs) (line 292)
    reshape_call_result_128899 = invoke(stypy.reporting.localization.Localization(__file__, 292, 26), reshape_128889, *[arange_call_result_128893, shape_call_result_128897], **kwargs_128898)
    
    # Applying the binary operator '*' (line 292)
    result_mul_128900 = python_operator(stypy.reporting.localization.Localization(__file__, 292, 15), '*', ua_small_128888, reshape_call_result_128899)
    
    # Processing the call keyword arguments (line 292)
    kwargs_128901 = {}
    # Getting the type of 'type' (line 292)
    type_128887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 10), 'type', False)
    # Calling type(args, kwargs) (line 292)
    type_call_result_128902 = invoke(stypy.reporting.localization.Localization(__file__, 292, 10), type_128887, *[result_mul_128900], **kwargs_128901)
    
    # Processing the call keyword arguments (line 292)
    kwargs_128903 = {}
    # Getting the type of 'print' (line 292)
    print_128886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 292, 4), 'print', False)
    # Calling print(args, kwargs) (line 292)
    print_call_result_128904 = invoke(stypy.reporting.localization.Localization(__file__, 292, 4), print_128886, *[type_call_result_128902], **kwargs_128903)
    
    
    # Call to print(...): (line 293)
    # Processing the call arguments (line 293)
    
    # Call to reshape(...): (line 293)
    # Processing the call arguments (line 293)
    # Getting the type of 'ua_small' (line 293)
    ua_small_128907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 18), 'ua_small', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 293)
    tuple_128908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 29), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 293)
    # Adding element type (line 293)
    int_128909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 29), tuple_128908, int_128909)
    # Adding element type (line 293)
    int_128910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 293, 32), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 293, 29), tuple_128908, int_128910)
    
    # Processing the call keyword arguments (line 293)
    kwargs_128911 = {}
    # Getting the type of 'reshape' (line 293)
    reshape_128906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 10), 'reshape', False)
    # Calling reshape(args, kwargs) (line 293)
    reshape_call_result_128912 = invoke(stypy.reporting.localization.Localization(__file__, 293, 10), reshape_128906, *[ua_small_128907, tuple_128908], **kwargs_128911)
    
    # Processing the call keyword arguments (line 293)
    kwargs_128913 = {}
    # Getting the type of 'print' (line 293)
    print_128905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 293, 4), 'print', False)
    # Calling print(args, kwargs) (line 293)
    print_call_result_128914 = invoke(stypy.reporting.localization.Localization(__file__, 293, 4), print_128905, *[reshape_call_result_128912], **kwargs_128913)
    
    
    # Call to print(...): (line 294)
    # Processing the call arguments (line 294)
    
    # Call to transpose(...): (line 294)
    # Processing the call arguments (line 294)
    # Getting the type of 'ua_small' (line 294)
    ua_small_128917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 20), 'ua_small', False)
    # Processing the call keyword arguments (line 294)
    kwargs_128918 = {}
    # Getting the type of 'transpose' (line 294)
    transpose_128916 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 10), 'transpose', False)
    # Calling transpose(args, kwargs) (line 294)
    transpose_call_result_128919 = invoke(stypy.reporting.localization.Localization(__file__, 294, 10), transpose_128916, *[ua_small_128917], **kwargs_128918)
    
    # Processing the call keyword arguments (line 294)
    kwargs_128920 = {}
    # Getting the type of 'print' (line 294)
    print_128915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 294, 4), 'print', False)
    # Calling print(args, kwargs) (line 294)
    print_call_result_128921 = invoke(stypy.reporting.localization.Localization(__file__, 294, 4), print_128915, *[transpose_call_result_128919], **kwargs_128920)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
