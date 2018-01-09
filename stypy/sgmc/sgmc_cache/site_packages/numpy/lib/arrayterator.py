
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: A buffered iterator for big arrays.
3: 
4: This module solves the problem of iterating over a big file-based array
5: without having to read it into memory. The `Arrayterator` class wraps
6: an array object, and when iterated it will return sub-arrays with at most
7: a user-specified number of elements.
8: 
9: '''
10: from __future__ import division, absolute_import, print_function
11: 
12: from operator import mul
13: from functools import reduce
14: 
15: from numpy.compat import long
16: 
17: __all__ = ['Arrayterator']
18: 
19: 
20: class Arrayterator(object):
21:     '''
22:     Buffered iterator for big arrays.
23: 
24:     `Arrayterator` creates a buffered iterator for reading big arrays in small
25:     contiguous blocks. The class is useful for objects stored in the
26:     file system. It allows iteration over the object *without* reading
27:     everything in memory; instead, small blocks are read and iterated over.
28: 
29:     `Arrayterator` can be used with any object that supports multidimensional
30:     slices. This includes NumPy arrays, but also variables from
31:     Scientific.IO.NetCDF or pynetcdf for example.
32: 
33:     Parameters
34:     ----------
35:     var : array_like
36:         The object to iterate over.
37:     buf_size : int, optional
38:         The buffer size. If `buf_size` is supplied, the maximum amount of
39:         data that will be read into memory is `buf_size` elements.
40:         Default is None, which will read as many element as possible
41:         into memory.
42: 
43:     Attributes
44:     ----------
45:     var
46:     buf_size
47:     start
48:     stop
49:     step
50:     shape
51:     flat
52: 
53:     See Also
54:     --------
55:     ndenumerate : Multidimensional array iterator.
56:     flatiter : Flat array iterator.
57:     memmap : Create a memory-map to an array stored in a binary file on disk.
58: 
59:     Notes
60:     -----
61:     The algorithm works by first finding a "running dimension", along which
62:     the blocks will be extracted. Given an array of dimensions
63:     ``(d1, d2, ..., dn)``, e.g. if `buf_size` is smaller than ``d1``, the
64:     first dimension will be used. If, on the other hand,
65:     ``d1 < buf_size < d1*d2`` the second dimension will be used, and so on.
66:     Blocks are extracted along this dimension, and when the last block is
67:     returned the process continues from the next dimension, until all
68:     elements have been read.
69: 
70:     Examples
71:     --------
72:     >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
73:     >>> a_itor = np.lib.Arrayterator(a, 2)
74:     >>> a_itor.shape
75:     (3, 4, 5, 6)
76: 
77:     Now we can iterate over ``a_itor``, and it will return arrays of size
78:     two. Since `buf_size` was smaller than any dimension, the first
79:     dimension will be iterated over first:
80: 
81:     >>> for subarr in a_itor:
82:     ...     if not subarr.all():
83:     ...         print(subarr, subarr.shape)
84:     ...
85:     [[[[0 1]]]] (1, 1, 1, 2)
86: 
87:     '''
88: 
89:     def __init__(self, var, buf_size=None):
90:         self.var = var
91:         self.buf_size = buf_size
92: 
93:         self.start = [0 for dim in var.shape]
94:         self.stop = [dim for dim in var.shape]
95:         self.step = [1 for dim in var.shape]
96: 
97:     def __getattr__(self, attr):
98:         return getattr(self.var, attr)
99: 
100:     def __getitem__(self, index):
101:         '''
102:         Return a new arrayterator.
103: 
104:         '''
105:         # Fix index, handling ellipsis and incomplete slices.
106:         if not isinstance(index, tuple):
107:             index = (index,)
108:         fixed = []
109:         length, dims = len(index), len(self.shape)
110:         for slice_ in index:
111:             if slice_ is Ellipsis:
112:                 fixed.extend([slice(None)] * (dims-length+1))
113:                 length = len(fixed)
114:             elif isinstance(slice_, (int, long)):
115:                 fixed.append(slice(slice_, slice_+1, 1))
116:             else:
117:                 fixed.append(slice_)
118:         index = tuple(fixed)
119:         if len(index) < dims:
120:             index += (slice(None),) * (dims-len(index))
121: 
122:         # Return a new arrayterator object.
123:         out = self.__class__(self.var, self.buf_size)
124:         for i, (start, stop, step, slice_) in enumerate(
125:                 zip(self.start, self.stop, self.step, index)):
126:             out.start[i] = start + (slice_.start or 0)
127:             out.step[i] = step * (slice_.step or 1)
128:             out.stop[i] = start + (slice_.stop or stop-start)
129:             out.stop[i] = min(stop, out.stop[i])
130:         return out
131: 
132:     def __array__(self):
133:         '''
134:         Return corresponding data.
135: 
136:         '''
137:         slice_ = tuple(slice(*t) for t in zip(
138:                 self.start, self.stop, self.step))
139:         return self.var[slice_]
140: 
141:     @property
142:     def flat(self):
143:         '''
144:         A 1-D flat iterator for Arrayterator objects.
145: 
146:         This iterator returns elements of the array to be iterated over in
147:         `Arrayterator` one by one. It is similar to `flatiter`.
148: 
149:         See Also
150:         --------
151:         Arrayterator
152:         flatiter
153: 
154:         Examples
155:         --------
156:         >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
157:         >>> a_itor = np.lib.Arrayterator(a, 2)
158: 
159:         >>> for subarr in a_itor.flat:
160:         ...     if not subarr:
161:         ...         print(subarr, type(subarr))
162:         ...
163:         0 <type 'numpy.int32'>
164: 
165:         '''
166:         for block in self:
167:             for value in block.flat:
168:                 yield value
169: 
170:     @property
171:     def shape(self):
172:         '''
173:         The shape of the array to be iterated over.
174: 
175:         For an example, see `Arrayterator`.
176: 
177:         '''
178:         return tuple(((stop-start-1)//step+1) for start, stop, step in
179:                 zip(self.start, self.stop, self.step))
180: 
181:     def __iter__(self):
182:         # Skip arrays with degenerate dimensions
183:         if [dim for dim in self.shape if dim <= 0]:
184:             return
185: 
186:         start = self.start[:]
187:         stop = self.stop[:]
188:         step = self.step[:]
189:         ndims = len(self.var.shape)
190: 
191:         while True:
192:             count = self.buf_size or reduce(mul, self.shape)
193: 
194:             # iterate over each dimension, looking for the
195:             # running dimension (ie, the dimension along which
196:             # the blocks will be built from)
197:             rundim = 0
198:             for i in range(ndims-1, -1, -1):
199:                 # if count is zero we ran out of elements to read
200:                 # along higher dimensions, so we read only a single position
201:                 if count == 0:
202:                     stop[i] = start[i]+1
203:                 elif count <= self.shape[i]:
204:                     # limit along this dimension
205:                     stop[i] = start[i] + count*step[i]
206:                     rundim = i
207:                 else:
208:                     # read everything along this dimension
209:                     stop[i] = self.stop[i]
210:                 stop[i] = min(self.stop[i], stop[i])
211:                 count = count//self.shape[i]
212: 
213:             # yield a block
214:             slice_ = tuple(slice(*t) for t in zip(start, stop, step))
215:             yield self.var[slice_]
216: 
217:             # Update start position, taking care of overflow to
218:             # other dimensions
219:             start[rundim] = stop[rundim]  # start where we stopped
220:             for i in range(ndims-1, 0, -1):
221:                 if start[i] >= self.stop[i]:
222:                     start[i] = self.start[i]
223:                     start[i-1] += self.step[i-1]
224:             if start[0] >= self.stop[0]:
225:                 return
226: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_104739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', '\nA buffered iterator for big arrays.\n\nThis module solves the problem of iterating over a big file-based array\nwithout having to read it into memory. The `Arrayterator` class wraps\nan array object, and when iterated it will return sub-arrays with at most\na user-specified number of elements.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from operator import mul' statement (line 12)
from operator import mul

import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'operator', None, module_type_store, ['mul'], [mul])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from functools import reduce' statement (line 13)
from functools import reduce

import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'functools', None, module_type_store, ['reduce'], [reduce])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from numpy.compat import long' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_104740 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.compat')

if (type(import_104740) is not StypyTypeError):

    if (import_104740 != 'pyd_module'):
        __import__(import_104740)
        sys_modules_104741 = sys.modules[import_104740]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.compat', sys_modules_104741.module_type_store, module_type_store, ['long'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_104741, sys_modules_104741.module_type_store, module_type_store)
    else:
        from numpy.compat import long

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.compat', None, module_type_store, ['long'], [long])

else:
    # Assigning a type to the variable 'numpy.compat' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'numpy.compat', import_104740)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 17):

# Assigning a List to a Name (line 17):
__all__ = ['Arrayterator']
module_type_store.set_exportable_members(['Arrayterator'])

# Obtaining an instance of the builtin type 'list' (line 17)
list_104742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_104743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 11), 'str', 'Arrayterator')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 10), list_104742, str_104743)

# Assigning a type to the variable '__all__' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), '__all__', list_104742)
# Declaration of the 'Arrayterator' class

class Arrayterator(object, ):
    str_104744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, (-1)), 'str', '\n    Buffered iterator for big arrays.\n\n    `Arrayterator` creates a buffered iterator for reading big arrays in small\n    contiguous blocks. The class is useful for objects stored in the\n    file system. It allows iteration over the object *without* reading\n    everything in memory; instead, small blocks are read and iterated over.\n\n    `Arrayterator` can be used with any object that supports multidimensional\n    slices. This includes NumPy arrays, but also variables from\n    Scientific.IO.NetCDF or pynetcdf for example.\n\n    Parameters\n    ----------\n    var : array_like\n        The object to iterate over.\n    buf_size : int, optional\n        The buffer size. If `buf_size` is supplied, the maximum amount of\n        data that will be read into memory is `buf_size` elements.\n        Default is None, which will read as many element as possible\n        into memory.\n\n    Attributes\n    ----------\n    var\n    buf_size\n    start\n    stop\n    step\n    shape\n    flat\n\n    See Also\n    --------\n    ndenumerate : Multidimensional array iterator.\n    flatiter : Flat array iterator.\n    memmap : Create a memory-map to an array stored in a binary file on disk.\n\n    Notes\n    -----\n    The algorithm works by first finding a "running dimension", along which\n    the blocks will be extracted. Given an array of dimensions\n    ``(d1, d2, ..., dn)``, e.g. if `buf_size` is smaller than ``d1``, the\n    first dimension will be used. If, on the other hand,\n    ``d1 < buf_size < d1*d2`` the second dimension will be used, and so on.\n    Blocks are extracted along this dimension, and when the last block is\n    returned the process continues from the next dimension, until all\n    elements have been read.\n\n    Examples\n    --------\n    >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)\n    >>> a_itor = np.lib.Arrayterator(a, 2)\n    >>> a_itor.shape\n    (3, 4, 5, 6)\n\n    Now we can iterate over ``a_itor``, and it will return arrays of size\n    two. Since `buf_size` was smaller than any dimension, the first\n    dimension will be iterated over first:\n\n    >>> for subarr in a_itor:\n    ...     if not subarr.all():\n    ...         print(subarr, subarr.shape)\n    ...\n    [[[[0 1]]]] (1, 1, 1, 2)\n\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 89)
        None_104745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 37), 'None')
        defaults = [None_104745]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 89, 4, False)
        # Assigning a type to the variable 'self' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arrayterator.__init__', ['var', 'buf_size'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['var', 'buf_size'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 90):
        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'var' (line 90)
        var_104746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'var')
        # Getting the type of 'self' (line 90)
        self_104747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'var' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_104747, 'var', var_104746)
        
        # Assigning a Name to a Attribute (line 91):
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'buf_size' (line 91)
        buf_size_104748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 24), 'buf_size')
        # Getting the type of 'self' (line 91)
        self_104749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'buf_size' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_104749, 'buf_size', buf_size_104748)
        
        # Assigning a ListComp to a Attribute (line 93):
        
        # Assigning a ListComp to a Attribute (line 93):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'var' (line 93)
        var_104751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'var')
        # Obtaining the member 'shape' of a type (line 93)
        shape_104752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 35), var_104751, 'shape')
        comprehension_104753 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 22), shape_104752)
        # Assigning a type to the variable 'dim' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'dim', comprehension_104753)
        int_104750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'int')
        list_104754 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 22), list_104754, int_104750)
        # Getting the type of 'self' (line 93)
        self_104755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'self')
        # Setting the type of the member 'start' of a type (line 93)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 8), self_104755, 'start', list_104754)
        
        # Assigning a ListComp to a Attribute (line 94):
        
        # Assigning a ListComp to a Attribute (line 94):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'var' (line 94)
        var_104757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 36), 'var')
        # Obtaining the member 'shape' of a type (line 94)
        shape_104758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 36), var_104757, 'shape')
        comprehension_104759 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), shape_104758)
        # Assigning a type to the variable 'dim' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'dim', comprehension_104759)
        # Getting the type of 'dim' (line 94)
        dim_104756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 21), 'dim')
        list_104760 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 94, 21), list_104760, dim_104756)
        # Getting the type of 'self' (line 94)
        self_104761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 8), 'self')
        # Setting the type of the member 'stop' of a type (line 94)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 8), self_104761, 'stop', list_104760)
        
        # Assigning a ListComp to a Attribute (line 95):
        
        # Assigning a ListComp to a Attribute (line 95):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'var' (line 95)
        var_104763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 34), 'var')
        # Obtaining the member 'shape' of a type (line 95)
        shape_104764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 34), var_104763, 'shape')
        comprehension_104765 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), shape_104764)
        # Assigning a type to the variable 'dim' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'dim', comprehension_104765)
        int_104762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'int')
        list_104766 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 21), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 21), list_104766, int_104762)
        # Getting the type of 'self' (line 95)
        self_104767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'self')
        # Setting the type of the member 'step' of a type (line 95)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 8), self_104767, 'step', list_104766)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def __getattr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getattr__'
        module_type_store = module_type_store.open_function_context('__getattr__', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_localization', localization)
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_function_name', 'Arrayterator.__getattr__')
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_param_names_list', ['attr'])
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Arrayterator.__getattr__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arrayterator.__getattr__', ['attr'], None, None, defaults, varargs, kwargs)

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

        
        # Call to getattr(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'self' (line 98)
        self_104769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 23), 'self', False)
        # Obtaining the member 'var' of a type (line 98)
        var_104770 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 23), self_104769, 'var')
        # Getting the type of 'attr' (line 98)
        attr_104771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'attr', False)
        # Processing the call keyword arguments (line 98)
        kwargs_104772 = {}
        # Getting the type of 'getattr' (line 98)
        getattr_104768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'getattr', False)
        # Calling getattr(args, kwargs) (line 98)
        getattr_call_result_104773 = invoke(stypy.reporting.localization.Localization(__file__, 98, 15), getattr_104768, *[var_104770, attr_104771], **kwargs_104772)
        
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', getattr_call_result_104773)
        
        # ################# End of '__getattr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getattr__' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_104774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_104774)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getattr__'
        return stypy_return_type_104774


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 100, 4, False)
        # Assigning a type to the variable 'self' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_function_name', 'Arrayterator.__getitem__')
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['index'])
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Arrayterator.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arrayterator.__getitem__', ['index'], None, None, defaults, varargs, kwargs)

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

        str_104775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, (-1)), 'str', '\n        Return a new arrayterator.\n\n        ')
        
        # Type idiom detected: calculating its left and rigth part (line 106)
        # Getting the type of 'tuple' (line 106)
        tuple_104776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 33), 'tuple')
        # Getting the type of 'index' (line 106)
        index_104777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 26), 'index')
        
        (may_be_104778, more_types_in_union_104779) = may_not_be_subtype(tuple_104776, index_104777)

        if may_be_104778:

            if more_types_in_union_104779:
                # Runtime conditional SSA (line 106)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'index' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'index', remove_subtype_from_union(index_104777, tuple))
            
            # Assigning a Tuple to a Name (line 107):
            
            # Assigning a Tuple to a Name (line 107):
            
            # Obtaining an instance of the builtin type 'tuple' (line 107)
            tuple_104780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 21), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 107)
            # Adding element type (line 107)
            # Getting the type of 'index' (line 107)
            index_104781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'index')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 21), tuple_104780, index_104781)
            
            # Assigning a type to the variable 'index' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'index', tuple_104780)

            if more_types_in_union_104779:
                # SSA join for if statement (line 106)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Assigning a List to a Name (line 108):
        
        # Assigning a List to a Name (line 108):
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_104782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        
        # Assigning a type to the variable 'fixed' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'fixed', list_104782)
        
        # Assigning a Tuple to a Tuple (line 109):
        
        # Assigning a Call to a Name (line 109):
        
        # Call to len(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'index' (line 109)
        index_104784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 27), 'index', False)
        # Processing the call keyword arguments (line 109)
        kwargs_104785 = {}
        # Getting the type of 'len' (line 109)
        len_104783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'len', False)
        # Calling len(args, kwargs) (line 109)
        len_call_result_104786 = invoke(stypy.reporting.localization.Localization(__file__, 109, 23), len_104783, *[index_104784], **kwargs_104785)
        
        # Assigning a type to the variable 'tuple_assignment_104737' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_104737', len_call_result_104786)
        
        # Assigning a Call to a Name (line 109):
        
        # Call to len(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'self' (line 109)
        self_104788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 39), 'self', False)
        # Obtaining the member 'shape' of a type (line 109)
        shape_104789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 39), self_104788, 'shape')
        # Processing the call keyword arguments (line 109)
        kwargs_104790 = {}
        # Getting the type of 'len' (line 109)
        len_104787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 35), 'len', False)
        # Calling len(args, kwargs) (line 109)
        len_call_result_104791 = invoke(stypy.reporting.localization.Localization(__file__, 109, 35), len_104787, *[shape_104789], **kwargs_104790)
        
        # Assigning a type to the variable 'tuple_assignment_104738' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_104738', len_call_result_104791)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_assignment_104737' (line 109)
        tuple_assignment_104737_104792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_104737')
        # Assigning a type to the variable 'length' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'length', tuple_assignment_104737_104792)
        
        # Assigning a Name to a Name (line 109):
        # Getting the type of 'tuple_assignment_104738' (line 109)
        tuple_assignment_104738_104793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'tuple_assignment_104738')
        # Assigning a type to the variable 'dims' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'dims', tuple_assignment_104738_104793)
        
        # Getting the type of 'index' (line 110)
        index_104794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 22), 'index')
        # Testing the type of a for loop iterable (line 110)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 8), index_104794)
        # Getting the type of the for loop variable (line 110)
        for_loop_var_104795 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 8), index_104794)
        # Assigning a type to the variable 'slice_' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'slice_', for_loop_var_104795)
        # SSA begins for a for statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'slice_' (line 111)
        slice__104796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 15), 'slice_')
        # Getting the type of 'Ellipsis' (line 111)
        Ellipsis_104797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 25), 'Ellipsis')
        # Applying the binary operator 'is' (line 111)
        result_is__104798 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 15), 'is', slice__104796, Ellipsis_104797)
        
        # Testing the type of an if condition (line 111)
        if_condition_104799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 12), result_is__104798)
        # Assigning a type to the variable 'if_condition_104799' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'if_condition_104799', if_condition_104799)
        # SSA begins for if statement (line 111)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to extend(...): (line 112)
        # Processing the call arguments (line 112)
        
        # Obtaining an instance of the builtin type 'list' (line 112)
        list_104802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 29), 'list')
        # Adding type elements to the builtin type 'list' instance (line 112)
        # Adding element type (line 112)
        
        # Call to slice(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'None' (line 112)
        None_104804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 36), 'None', False)
        # Processing the call keyword arguments (line 112)
        kwargs_104805 = {}
        # Getting the type of 'slice' (line 112)
        slice_104803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'slice', False)
        # Calling slice(args, kwargs) (line 112)
        slice_call_result_104806 = invoke(stypy.reporting.localization.Localization(__file__, 112, 30), slice_104803, *[None_104804], **kwargs_104805)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 29), list_104802, slice_call_result_104806)
        
        # Getting the type of 'dims' (line 112)
        dims_104807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 46), 'dims', False)
        # Getting the type of 'length' (line 112)
        length_104808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 51), 'length', False)
        # Applying the binary operator '-' (line 112)
        result_sub_104809 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 46), '-', dims_104807, length_104808)
        
        int_104810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 58), 'int')
        # Applying the binary operator '+' (line 112)
        result_add_104811 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 57), '+', result_sub_104809, int_104810)
        
        # Applying the binary operator '*' (line 112)
        result_mul_104812 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 29), '*', list_104802, result_add_104811)
        
        # Processing the call keyword arguments (line 112)
        kwargs_104813 = {}
        # Getting the type of 'fixed' (line 112)
        fixed_104800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'fixed', False)
        # Obtaining the member 'extend' of a type (line 112)
        extend_104801 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), fixed_104800, 'extend')
        # Calling extend(args, kwargs) (line 112)
        extend_call_result_104814 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), extend_104801, *[result_mul_104812], **kwargs_104813)
        
        
        # Assigning a Call to a Name (line 113):
        
        # Assigning a Call to a Name (line 113):
        
        # Call to len(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'fixed' (line 113)
        fixed_104816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'fixed', False)
        # Processing the call keyword arguments (line 113)
        kwargs_104817 = {}
        # Getting the type of 'len' (line 113)
        len_104815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'len', False)
        # Calling len(args, kwargs) (line 113)
        len_call_result_104818 = invoke(stypy.reporting.localization.Localization(__file__, 113, 25), len_104815, *[fixed_104816], **kwargs_104817)
        
        # Assigning a type to the variable 'length' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'length', len_call_result_104818)
        # SSA branch for the else part of an if statement (line 111)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'slice_' (line 114)
        slice__104820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'slice_', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 114)
        tuple_104821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 114)
        # Adding element type (line 114)
        # Getting the type of 'int' (line 114)
        int_104822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 37), 'int', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 37), tuple_104821, int_104822)
        # Adding element type (line 114)
        # Getting the type of 'long' (line 114)
        long_104823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 42), 'long', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 114, 37), tuple_104821, long_104823)
        
        # Processing the call keyword arguments (line 114)
        kwargs_104824 = {}
        # Getting the type of 'isinstance' (line 114)
        isinstance_104819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 114)
        isinstance_call_result_104825 = invoke(stypy.reporting.localization.Localization(__file__, 114, 17), isinstance_104819, *[slice__104820, tuple_104821], **kwargs_104824)
        
        # Testing the type of an if condition (line 114)
        if_condition_104826 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 17), isinstance_call_result_104825)
        # Assigning a type to the variable 'if_condition_104826' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 17), 'if_condition_104826', if_condition_104826)
        # SSA begins for if statement (line 114)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 115)
        # Processing the call arguments (line 115)
        
        # Call to slice(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'slice_' (line 115)
        slice__104830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'slice_', False)
        # Getting the type of 'slice_' (line 115)
        slice__104831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 43), 'slice_', False)
        int_104832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 50), 'int')
        # Applying the binary operator '+' (line 115)
        result_add_104833 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 43), '+', slice__104831, int_104832)
        
        int_104834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 53), 'int')
        # Processing the call keyword arguments (line 115)
        kwargs_104835 = {}
        # Getting the type of 'slice' (line 115)
        slice_104829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 29), 'slice', False)
        # Calling slice(args, kwargs) (line 115)
        slice_call_result_104836 = invoke(stypy.reporting.localization.Localization(__file__, 115, 29), slice_104829, *[slice__104830, result_add_104833, int_104834], **kwargs_104835)
        
        # Processing the call keyword arguments (line 115)
        kwargs_104837 = {}
        # Getting the type of 'fixed' (line 115)
        fixed_104827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 16), 'fixed', False)
        # Obtaining the member 'append' of a type (line 115)
        append_104828 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 16), fixed_104827, 'append')
        # Calling append(args, kwargs) (line 115)
        append_call_result_104838 = invoke(stypy.reporting.localization.Localization(__file__, 115, 16), append_104828, *[slice_call_result_104836], **kwargs_104837)
        
        # SSA branch for the else part of an if statement (line 114)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'slice_' (line 117)
        slice__104841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 29), 'slice_', False)
        # Processing the call keyword arguments (line 117)
        kwargs_104842 = {}
        # Getting the type of 'fixed' (line 117)
        fixed_104839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'fixed', False)
        # Obtaining the member 'append' of a type (line 117)
        append_104840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 16), fixed_104839, 'append')
        # Calling append(args, kwargs) (line 117)
        append_call_result_104843 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), append_104840, *[slice__104841], **kwargs_104842)
        
        # SSA join for if statement (line 114)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 111)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 118):
        
        # Assigning a Call to a Name (line 118):
        
        # Call to tuple(...): (line 118)
        # Processing the call arguments (line 118)
        # Getting the type of 'fixed' (line 118)
        fixed_104845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'fixed', False)
        # Processing the call keyword arguments (line 118)
        kwargs_104846 = {}
        # Getting the type of 'tuple' (line 118)
        tuple_104844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'tuple', False)
        # Calling tuple(args, kwargs) (line 118)
        tuple_call_result_104847 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), tuple_104844, *[fixed_104845], **kwargs_104846)
        
        # Assigning a type to the variable 'index' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'index', tuple_call_result_104847)
        
        
        
        # Call to len(...): (line 119)
        # Processing the call arguments (line 119)
        # Getting the type of 'index' (line 119)
        index_104849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'index', False)
        # Processing the call keyword arguments (line 119)
        kwargs_104850 = {}
        # Getting the type of 'len' (line 119)
        len_104848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'len', False)
        # Calling len(args, kwargs) (line 119)
        len_call_result_104851 = invoke(stypy.reporting.localization.Localization(__file__, 119, 11), len_104848, *[index_104849], **kwargs_104850)
        
        # Getting the type of 'dims' (line 119)
        dims_104852 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 24), 'dims')
        # Applying the binary operator '<' (line 119)
        result_lt_104853 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '<', len_call_result_104851, dims_104852)
        
        # Testing the type of an if condition (line 119)
        if_condition_104854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_lt_104853)
        # Assigning a type to the variable 'if_condition_104854' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_104854', if_condition_104854)
        # SSA begins for if statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'index' (line 120)
        index_104855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'index')
        
        # Obtaining an instance of the builtin type 'tuple' (line 120)
        tuple_104856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 22), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 120)
        # Adding element type (line 120)
        
        # Call to slice(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'None' (line 120)
        None_104858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 28), 'None', False)
        # Processing the call keyword arguments (line 120)
        kwargs_104859 = {}
        # Getting the type of 'slice' (line 120)
        slice_104857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 22), 'slice', False)
        # Calling slice(args, kwargs) (line 120)
        slice_call_result_104860 = invoke(stypy.reporting.localization.Localization(__file__, 120, 22), slice_104857, *[None_104858], **kwargs_104859)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 22), tuple_104856, slice_call_result_104860)
        
        # Getting the type of 'dims' (line 120)
        dims_104861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 39), 'dims')
        
        # Call to len(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'index' (line 120)
        index_104863 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 48), 'index', False)
        # Processing the call keyword arguments (line 120)
        kwargs_104864 = {}
        # Getting the type of 'len' (line 120)
        len_104862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 44), 'len', False)
        # Calling len(args, kwargs) (line 120)
        len_call_result_104865 = invoke(stypy.reporting.localization.Localization(__file__, 120, 44), len_104862, *[index_104863], **kwargs_104864)
        
        # Applying the binary operator '-' (line 120)
        result_sub_104866 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 39), '-', dims_104861, len_call_result_104865)
        
        # Applying the binary operator '*' (line 120)
        result_mul_104867 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 21), '*', tuple_104856, result_sub_104866)
        
        # Applying the binary operator '+=' (line 120)
        result_iadd_104868 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 12), '+=', index_104855, result_mul_104867)
        # Assigning a type to the variable 'index' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'index', result_iadd_104868)
        
        # SSA join for if statement (line 119)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 123):
        
        # Assigning a Call to a Name (line 123):
        
        # Call to __class__(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'self' (line 123)
        self_104871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 29), 'self', False)
        # Obtaining the member 'var' of a type (line 123)
        var_104872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 29), self_104871, 'var')
        # Getting the type of 'self' (line 123)
        self_104873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 39), 'self', False)
        # Obtaining the member 'buf_size' of a type (line 123)
        buf_size_104874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 39), self_104873, 'buf_size')
        # Processing the call keyword arguments (line 123)
        kwargs_104875 = {}
        # Getting the type of 'self' (line 123)
        self_104869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'self', False)
        # Obtaining the member '__class__' of a type (line 123)
        class___104870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 14), self_104869, '__class__')
        # Calling __class__(args, kwargs) (line 123)
        class___call_result_104876 = invoke(stypy.reporting.localization.Localization(__file__, 123, 14), class___104870, *[var_104872, buf_size_104874], **kwargs_104875)
        
        # Assigning a type to the variable 'out' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'out', class___call_result_104876)
        
        
        # Call to enumerate(...): (line 124)
        # Processing the call arguments (line 124)
        
        # Call to zip(...): (line 125)
        # Processing the call arguments (line 125)
        # Getting the type of 'self' (line 125)
        self_104879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 20), 'self', False)
        # Obtaining the member 'start' of a type (line 125)
        start_104880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 20), self_104879, 'start')
        # Getting the type of 'self' (line 125)
        self_104881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 32), 'self', False)
        # Obtaining the member 'stop' of a type (line 125)
        stop_104882 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 32), self_104881, 'stop')
        # Getting the type of 'self' (line 125)
        self_104883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 43), 'self', False)
        # Obtaining the member 'step' of a type (line 125)
        step_104884 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 43), self_104883, 'step')
        # Getting the type of 'index' (line 125)
        index_104885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 54), 'index', False)
        # Processing the call keyword arguments (line 125)
        kwargs_104886 = {}
        # Getting the type of 'zip' (line 125)
        zip_104878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 16), 'zip', False)
        # Calling zip(args, kwargs) (line 125)
        zip_call_result_104887 = invoke(stypy.reporting.localization.Localization(__file__, 125, 16), zip_104878, *[start_104880, stop_104882, step_104884, index_104885], **kwargs_104886)
        
        # Processing the call keyword arguments (line 124)
        kwargs_104888 = {}
        # Getting the type of 'enumerate' (line 124)
        enumerate_104877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 46), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 124)
        enumerate_call_result_104889 = invoke(stypy.reporting.localization.Localization(__file__, 124, 46), enumerate_104877, *[zip_call_result_104887], **kwargs_104888)
        
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 8), enumerate_call_result_104889)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_104890 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 8), enumerate_call_result_104889)
        # Assigning a type to the variable 'i' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), for_loop_var_104890))
        # Assigning a type to the variable 'start' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'start', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), for_loop_var_104890))
        # Assigning a type to the variable 'stop' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'stop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), for_loop_var_104890))
        # Assigning a type to the variable 'step' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'step', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), for_loop_var_104890))
        # Assigning a type to the variable 'slice_' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'slice_', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 124, 8), for_loop_var_104890))
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a BinOp to a Subscript (line 126):
        
        # Assigning a BinOp to a Subscript (line 126):
        # Getting the type of 'start' (line 126)
        start_104891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'start')
        
        # Evaluating a boolean operation
        # Getting the type of 'slice_' (line 126)
        slice__104892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 36), 'slice_')
        # Obtaining the member 'start' of a type (line 126)
        start_104893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 36), slice__104892, 'start')
        int_104894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 52), 'int')
        # Applying the binary operator 'or' (line 126)
        result_or_keyword_104895 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 36), 'or', start_104893, int_104894)
        
        # Applying the binary operator '+' (line 126)
        result_add_104896 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 27), '+', start_104891, result_or_keyword_104895)
        
        # Getting the type of 'out' (line 126)
        out_104897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'out')
        # Obtaining the member 'start' of a type (line 126)
        start_104898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 12), out_104897, 'start')
        # Getting the type of 'i' (line 126)
        i_104899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'i')
        # Storing an element on a container (line 126)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 126, 12), start_104898, (i_104899, result_add_104896))
        
        # Assigning a BinOp to a Subscript (line 127):
        
        # Assigning a BinOp to a Subscript (line 127):
        # Getting the type of 'step' (line 127)
        step_104900 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 26), 'step')
        
        # Evaluating a boolean operation
        # Getting the type of 'slice_' (line 127)
        slice__104901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 34), 'slice_')
        # Obtaining the member 'step' of a type (line 127)
        step_104902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 34), slice__104901, 'step')
        int_104903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 49), 'int')
        # Applying the binary operator 'or' (line 127)
        result_or_keyword_104904 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 34), 'or', step_104902, int_104903)
        
        # Applying the binary operator '*' (line 127)
        result_mul_104905 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 26), '*', step_104900, result_or_keyword_104904)
        
        # Getting the type of 'out' (line 127)
        out_104906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'out')
        # Obtaining the member 'step' of a type (line 127)
        step_104907 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 12), out_104906, 'step')
        # Getting the type of 'i' (line 127)
        i_104908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 21), 'i')
        # Storing an element on a container (line 127)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 127, 12), step_104907, (i_104908, result_mul_104905))
        
        # Assigning a BinOp to a Subscript (line 128):
        
        # Assigning a BinOp to a Subscript (line 128):
        # Getting the type of 'start' (line 128)
        start_104909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 26), 'start')
        
        # Evaluating a boolean operation
        # Getting the type of 'slice_' (line 128)
        slice__104910 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 35), 'slice_')
        # Obtaining the member 'stop' of a type (line 128)
        stop_104911 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 35), slice__104910, 'stop')
        # Getting the type of 'stop' (line 128)
        stop_104912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 50), 'stop')
        # Getting the type of 'start' (line 128)
        start_104913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 55), 'start')
        # Applying the binary operator '-' (line 128)
        result_sub_104914 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 50), '-', stop_104912, start_104913)
        
        # Applying the binary operator 'or' (line 128)
        result_or_keyword_104915 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 35), 'or', stop_104911, result_sub_104914)
        
        # Applying the binary operator '+' (line 128)
        result_add_104916 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 26), '+', start_104909, result_or_keyword_104915)
        
        # Getting the type of 'out' (line 128)
        out_104917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'out')
        # Obtaining the member 'stop' of a type (line 128)
        stop_104918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 12), out_104917, 'stop')
        # Getting the type of 'i' (line 128)
        i_104919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 21), 'i')
        # Storing an element on a container (line 128)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 128, 12), stop_104918, (i_104919, result_add_104916))
        
        # Assigning a Call to a Subscript (line 129):
        
        # Assigning a Call to a Subscript (line 129):
        
        # Call to min(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'stop' (line 129)
        stop_104921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 30), 'stop', False)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 129)
        i_104922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 45), 'i', False)
        # Getting the type of 'out' (line 129)
        out_104923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'out', False)
        # Obtaining the member 'stop' of a type (line 129)
        stop_104924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 36), out_104923, 'stop')
        # Obtaining the member '__getitem__' of a type (line 129)
        getitem___104925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 36), stop_104924, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 129)
        subscript_call_result_104926 = invoke(stypy.reporting.localization.Localization(__file__, 129, 36), getitem___104925, i_104922)
        
        # Processing the call keyword arguments (line 129)
        kwargs_104927 = {}
        # Getting the type of 'min' (line 129)
        min_104920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'min', False)
        # Calling min(args, kwargs) (line 129)
        min_call_result_104928 = invoke(stypy.reporting.localization.Localization(__file__, 129, 26), min_104920, *[stop_104921, subscript_call_result_104926], **kwargs_104927)
        
        # Getting the type of 'out' (line 129)
        out_104929 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'out')
        # Obtaining the member 'stop' of a type (line 129)
        stop_104930 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 12), out_104929, 'stop')
        # Getting the type of 'i' (line 129)
        i_104931 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 21), 'i')
        # Storing an element on a container (line 129)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 12), stop_104930, (i_104931, min_call_result_104928))
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'out' (line 130)
        out_104932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 15), 'out')
        # Assigning a type to the variable 'stypy_return_type' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'stypy_return_type', out_104932)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 100)
        stypy_return_type_104933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_104933)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_104933


    @norecursion
    def __array__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__array__'
        module_type_store = module_type_store.open_function_context('__array__', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Arrayterator.__array__.__dict__.__setitem__('stypy_localization', localization)
        Arrayterator.__array__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Arrayterator.__array__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Arrayterator.__array__.__dict__.__setitem__('stypy_function_name', 'Arrayterator.__array__')
        Arrayterator.__array__.__dict__.__setitem__('stypy_param_names_list', [])
        Arrayterator.__array__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Arrayterator.__array__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Arrayterator.__array__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Arrayterator.__array__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Arrayterator.__array__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Arrayterator.__array__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arrayterator.__array__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__array__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__array__(...)' code ##################

        str_104934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, (-1)), 'str', '\n        Return corresponding data.\n\n        ')
        
        # Assigning a Call to a Name (line 137):
        
        # Assigning a Call to a Name (line 137):
        
        # Call to tuple(...): (line 137)
        # Processing the call arguments (line 137)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 137, 23, True)
        # Calculating comprehension expression
        
        # Call to zip(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'self' (line 138)
        self_104941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 16), 'self', False)
        # Obtaining the member 'start' of a type (line 138)
        start_104942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 16), self_104941, 'start')
        # Getting the type of 'self' (line 138)
        self_104943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 28), 'self', False)
        # Obtaining the member 'stop' of a type (line 138)
        stop_104944 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 28), self_104943, 'stop')
        # Getting the type of 'self' (line 138)
        self_104945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 39), 'self', False)
        # Obtaining the member 'step' of a type (line 138)
        step_104946 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 39), self_104945, 'step')
        # Processing the call keyword arguments (line 137)
        kwargs_104947 = {}
        # Getting the type of 'zip' (line 137)
        zip_104940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 42), 'zip', False)
        # Calling zip(args, kwargs) (line 137)
        zip_call_result_104948 = invoke(stypy.reporting.localization.Localization(__file__, 137, 42), zip_104940, *[start_104942, stop_104944, step_104946], **kwargs_104947)
        
        comprehension_104949 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 23), zip_call_result_104948)
        # Assigning a type to the variable 't' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 't', comprehension_104949)
        
        # Call to slice(...): (line 137)
        # Getting the type of 't' (line 137)
        t_104937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 30), 't', False)
        # Processing the call keyword arguments (line 137)
        kwargs_104938 = {}
        # Getting the type of 'slice' (line 137)
        slice_104936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 23), 'slice', False)
        # Calling slice(args, kwargs) (line 137)
        slice_call_result_104939 = invoke(stypy.reporting.localization.Localization(__file__, 137, 23), slice_104936, *[t_104937], **kwargs_104938)
        
        list_104950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 23), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 23), list_104950, slice_call_result_104939)
        # Processing the call keyword arguments (line 137)
        kwargs_104951 = {}
        # Getting the type of 'tuple' (line 137)
        tuple_104935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 17), 'tuple', False)
        # Calling tuple(args, kwargs) (line 137)
        tuple_call_result_104952 = invoke(stypy.reporting.localization.Localization(__file__, 137, 17), tuple_104935, *[list_104950], **kwargs_104951)
        
        # Assigning a type to the variable 'slice_' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'slice_', tuple_call_result_104952)
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice_' (line 139)
        slice__104953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'slice_')
        # Getting the type of 'self' (line 139)
        self_104954 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'self')
        # Obtaining the member 'var' of a type (line 139)
        var_104955 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), self_104954, 'var')
        # Obtaining the member '__getitem__' of a type (line 139)
        getitem___104956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 15), var_104955, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 139)
        subscript_call_result_104957 = invoke(stypy.reporting.localization.Localization(__file__, 139, 15), getitem___104956, slice__104953)
        
        # Assigning a type to the variable 'stypy_return_type' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'stypy_return_type', subscript_call_result_104957)
        
        # ################# End of '__array__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__array__' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_104958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_104958)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__array__'
        return stypy_return_type_104958


    @norecursion
    def flat(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'flat'
        module_type_store = module_type_store.open_function_context('flat', 141, 4, False)
        # Assigning a type to the variable 'self' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Arrayterator.flat.__dict__.__setitem__('stypy_localization', localization)
        Arrayterator.flat.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Arrayterator.flat.__dict__.__setitem__('stypy_type_store', module_type_store)
        Arrayterator.flat.__dict__.__setitem__('stypy_function_name', 'Arrayterator.flat')
        Arrayterator.flat.__dict__.__setitem__('stypy_param_names_list', [])
        Arrayterator.flat.__dict__.__setitem__('stypy_varargs_param_name', None)
        Arrayterator.flat.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Arrayterator.flat.__dict__.__setitem__('stypy_call_defaults', defaults)
        Arrayterator.flat.__dict__.__setitem__('stypy_call_varargs', varargs)
        Arrayterator.flat.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Arrayterator.flat.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arrayterator.flat', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'flat', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'flat(...)' code ##################

        str_104959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, (-1)), 'str', "\n        A 1-D flat iterator for Arrayterator objects.\n\n        This iterator returns elements of the array to be iterated over in\n        `Arrayterator` one by one. It is similar to `flatiter`.\n\n        See Also\n        --------\n        Arrayterator\n        flatiter\n\n        Examples\n        --------\n        >>> a = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)\n        >>> a_itor = np.lib.Arrayterator(a, 2)\n\n        >>> for subarr in a_itor.flat:\n        ...     if not subarr:\n        ...         print(subarr, type(subarr))\n        ...\n        0 <type 'numpy.int32'>\n\n        ")
        
        # Getting the type of 'self' (line 166)
        self_104960 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 21), 'self')
        # Testing the type of a for loop iterable (line 166)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 166, 8), self_104960)
        # Getting the type of the for loop variable (line 166)
        for_loop_var_104961 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 166, 8), self_104960)
        # Assigning a type to the variable 'block' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 8), 'block', for_loop_var_104961)
        # SSA begins for a for statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'block' (line 167)
        block_104962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 25), 'block')
        # Obtaining the member 'flat' of a type (line 167)
        flat_104963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 25), block_104962, 'flat')
        # Testing the type of a for loop iterable (line 167)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 167, 12), flat_104963)
        # Getting the type of the for loop variable (line 167)
        for_loop_var_104964 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 167, 12), flat_104963)
        # Assigning a type to the variable 'value' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'value', for_loop_var_104964)
        # SSA begins for a for statement (line 167)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        # Creating a generator
        # Getting the type of 'value' (line 168)
        value_104965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 22), 'value')
        GeneratorType_104966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 16), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 168, 16), GeneratorType_104966, value_104965)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 16), 'stypy_return_type', GeneratorType_104966)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'flat(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'flat' in the type store
        # Getting the type of 'stypy_return_type' (line 141)
        stypy_return_type_104967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_104967)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'flat'
        return stypy_return_type_104967


    @norecursion
    def shape(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'shape'
        module_type_store = module_type_store.open_function_context('shape', 170, 4, False)
        # Assigning a type to the variable 'self' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Arrayterator.shape.__dict__.__setitem__('stypy_localization', localization)
        Arrayterator.shape.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Arrayterator.shape.__dict__.__setitem__('stypy_type_store', module_type_store)
        Arrayterator.shape.__dict__.__setitem__('stypy_function_name', 'Arrayterator.shape')
        Arrayterator.shape.__dict__.__setitem__('stypy_param_names_list', [])
        Arrayterator.shape.__dict__.__setitem__('stypy_varargs_param_name', None)
        Arrayterator.shape.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Arrayterator.shape.__dict__.__setitem__('stypy_call_defaults', defaults)
        Arrayterator.shape.__dict__.__setitem__('stypy_call_varargs', varargs)
        Arrayterator.shape.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Arrayterator.shape.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arrayterator.shape', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'shape', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'shape(...)' code ##################

        str_104968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, (-1)), 'str', '\n        The shape of the array to be iterated over.\n\n        For an example, see `Arrayterator`.\n\n        ')
        
        # Call to tuple(...): (line 178)
        # Processing the call arguments (line 178)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 178, 21, True)
        # Calculating comprehension expression
        
        # Call to zip(...): (line 179)
        # Processing the call arguments (line 179)
        # Getting the type of 'self' (line 179)
        self_104980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 20), 'self', False)
        # Obtaining the member 'start' of a type (line 179)
        start_104981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 20), self_104980, 'start')
        # Getting the type of 'self' (line 179)
        self_104982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 32), 'self', False)
        # Obtaining the member 'stop' of a type (line 179)
        stop_104983 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 32), self_104982, 'stop')
        # Getting the type of 'self' (line 179)
        self_104984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 43), 'self', False)
        # Obtaining the member 'step' of a type (line 179)
        step_104985 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 43), self_104984, 'step')
        # Processing the call keyword arguments (line 179)
        kwargs_104986 = {}
        # Getting the type of 'zip' (line 179)
        zip_104979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 16), 'zip', False)
        # Calling zip(args, kwargs) (line 179)
        zip_call_result_104987 = invoke(stypy.reporting.localization.Localization(__file__, 179, 16), zip_104979, *[start_104981, stop_104983, step_104985], **kwargs_104986)
        
        comprehension_104988 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), zip_call_result_104987)
        # Assigning a type to the variable 'start' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'start', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), comprehension_104988))
        # Assigning a type to the variable 'stop' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'stop', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), comprehension_104988))
        # Assigning a type to the variable 'step' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'step', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), comprehension_104988))
        # Getting the type of 'stop' (line 178)
        stop_104970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 23), 'stop', False)
        # Getting the type of 'start' (line 178)
        start_104971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'start', False)
        # Applying the binary operator '-' (line 178)
        result_sub_104972 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 23), '-', stop_104970, start_104971)
        
        int_104973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 34), 'int')
        # Applying the binary operator '-' (line 178)
        result_sub_104974 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 33), '-', result_sub_104972, int_104973)
        
        # Getting the type of 'step' (line 178)
        step_104975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 38), 'step', False)
        # Applying the binary operator '//' (line 178)
        result_floordiv_104976 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 22), '//', result_sub_104974, step_104975)
        
        int_104977 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 43), 'int')
        # Applying the binary operator '+' (line 178)
        result_add_104978 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 22), '+', result_floordiv_104976, int_104977)
        
        list_104989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 21), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 21), list_104989, result_add_104978)
        # Processing the call keyword arguments (line 178)
        kwargs_104990 = {}
        # Getting the type of 'tuple' (line 178)
        tuple_104969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 15), 'tuple', False)
        # Calling tuple(args, kwargs) (line 178)
        tuple_call_result_104991 = invoke(stypy.reporting.localization.Localization(__file__, 178, 15), tuple_104969, *[list_104989], **kwargs_104990)
        
        # Assigning a type to the variable 'stypy_return_type' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'stypy_return_type', tuple_call_result_104991)
        
        # ################# End of 'shape(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'shape' in the type store
        # Getting the type of 'stypy_return_type' (line 170)
        stypy_return_type_104992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_104992)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'shape'
        return stypy_return_type_104992


    @norecursion
    def __iter__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__iter__'
        module_type_store = module_type_store.open_function_context('__iter__', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        Arrayterator.__iter__.__dict__.__setitem__('stypy_localization', localization)
        Arrayterator.__iter__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        Arrayterator.__iter__.__dict__.__setitem__('stypy_type_store', module_type_store)
        Arrayterator.__iter__.__dict__.__setitem__('stypy_function_name', 'Arrayterator.__iter__')
        Arrayterator.__iter__.__dict__.__setitem__('stypy_param_names_list', [])
        Arrayterator.__iter__.__dict__.__setitem__('stypy_varargs_param_name', None)
        Arrayterator.__iter__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        Arrayterator.__iter__.__dict__.__setitem__('stypy_call_defaults', defaults)
        Arrayterator.__iter__.__dict__.__setitem__('stypy_call_varargs', varargs)
        Arrayterator.__iter__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        Arrayterator.__iter__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Arrayterator.__iter__', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__iter__', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__iter__(...)' code ##################

        
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 183)
        self_104997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 27), 'self')
        # Obtaining the member 'shape' of a type (line 183)
        shape_104998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 27), self_104997, 'shape')
        comprehension_104999 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 12), shape_104998)
        # Assigning a type to the variable 'dim' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'dim', comprehension_104999)
        
        # Getting the type of 'dim' (line 183)
        dim_104994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 41), 'dim')
        int_104995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 48), 'int')
        # Applying the binary operator '<=' (line 183)
        result_le_104996 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 41), '<=', dim_104994, int_104995)
        
        # Getting the type of 'dim' (line 183)
        dim_104993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'dim')
        list_105000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 12), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 183, 12), list_105000, dim_104993)
        # Testing the type of an if condition (line 183)
        if_condition_105001 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 8), list_105000)
        # Assigning a type to the variable 'if_condition_105001' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 8), 'if_condition_105001', if_condition_105001)
        # SSA begins for if statement (line 183)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 183)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Subscript to a Name (line 186):
        
        # Assigning a Subscript to a Name (line 186):
        
        # Obtaining the type of the subscript
        slice_105002 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 186, 16), None, None, None)
        # Getting the type of 'self' (line 186)
        self_105003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 16), 'self')
        # Obtaining the member 'start' of a type (line 186)
        start_105004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 16), self_105003, 'start')
        # Obtaining the member '__getitem__' of a type (line 186)
        getitem___105005 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 16), start_105004, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 186)
        subscript_call_result_105006 = invoke(stypy.reporting.localization.Localization(__file__, 186, 16), getitem___105005, slice_105002)
        
        # Assigning a type to the variable 'start' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 8), 'start', subscript_call_result_105006)
        
        # Assigning a Subscript to a Name (line 187):
        
        # Assigning a Subscript to a Name (line 187):
        
        # Obtaining the type of the subscript
        slice_105007 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 187, 15), None, None, None)
        # Getting the type of 'self' (line 187)
        self_105008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'self')
        # Obtaining the member 'stop' of a type (line 187)
        stop_105009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), self_105008, 'stop')
        # Obtaining the member '__getitem__' of a type (line 187)
        getitem___105010 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 15), stop_105009, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 187)
        subscript_call_result_105011 = invoke(stypy.reporting.localization.Localization(__file__, 187, 15), getitem___105010, slice_105007)
        
        # Assigning a type to the variable 'stop' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stop', subscript_call_result_105011)
        
        # Assigning a Subscript to a Name (line 188):
        
        # Assigning a Subscript to a Name (line 188):
        
        # Obtaining the type of the subscript
        slice_105012 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 188, 15), None, None, None)
        # Getting the type of 'self' (line 188)
        self_105013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 15), 'self')
        # Obtaining the member 'step' of a type (line 188)
        step_105014 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 15), self_105013, 'step')
        # Obtaining the member '__getitem__' of a type (line 188)
        getitem___105015 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 15), step_105014, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 188)
        subscript_call_result_105016 = invoke(stypy.reporting.localization.Localization(__file__, 188, 15), getitem___105015, slice_105012)
        
        # Assigning a type to the variable 'step' (line 188)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 8), 'step', subscript_call_result_105016)
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Call to len(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'self' (line 189)
        self_105018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 20), 'self', False)
        # Obtaining the member 'var' of a type (line 189)
        var_105019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 20), self_105018, 'var')
        # Obtaining the member 'shape' of a type (line 189)
        shape_105020 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 20), var_105019, 'shape')
        # Processing the call keyword arguments (line 189)
        kwargs_105021 = {}
        # Getting the type of 'len' (line 189)
        len_105017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'len', False)
        # Calling len(args, kwargs) (line 189)
        len_call_result_105022 = invoke(stypy.reporting.localization.Localization(__file__, 189, 16), len_105017, *[shape_105020], **kwargs_105021)
        
        # Assigning a type to the variable 'ndims' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'ndims', len_call_result_105022)
        
        # Getting the type of 'True' (line 191)
        True_105023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 14), 'True')
        # Testing the type of an if condition (line 191)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 8), True_105023)
        # SSA begins for while statement (line 191)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a BoolOp to a Name (line 192):
        
        # Assigning a BoolOp to a Name (line 192):
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 192)
        self_105024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 20), 'self')
        # Obtaining the member 'buf_size' of a type (line 192)
        buf_size_105025 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 20), self_105024, 'buf_size')
        
        # Call to reduce(...): (line 192)
        # Processing the call arguments (line 192)
        # Getting the type of 'mul' (line 192)
        mul_105027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 44), 'mul', False)
        # Getting the type of 'self' (line 192)
        self_105028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 49), 'self', False)
        # Obtaining the member 'shape' of a type (line 192)
        shape_105029 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 49), self_105028, 'shape')
        # Processing the call keyword arguments (line 192)
        kwargs_105030 = {}
        # Getting the type of 'reduce' (line 192)
        reduce_105026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 37), 'reduce', False)
        # Calling reduce(args, kwargs) (line 192)
        reduce_call_result_105031 = invoke(stypy.reporting.localization.Localization(__file__, 192, 37), reduce_105026, *[mul_105027, shape_105029], **kwargs_105030)
        
        # Applying the binary operator 'or' (line 192)
        result_or_keyword_105032 = python_operator(stypy.reporting.localization.Localization(__file__, 192, 20), 'or', buf_size_105025, reduce_call_result_105031)
        
        # Assigning a type to the variable 'count' (line 192)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 12), 'count', result_or_keyword_105032)
        
        # Assigning a Num to a Name (line 197):
        
        # Assigning a Num to a Name (line 197):
        int_105033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 21), 'int')
        # Assigning a type to the variable 'rundim' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'rundim', int_105033)
        
        
        # Call to range(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'ndims' (line 198)
        ndims_105035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 27), 'ndims', False)
        int_105036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 33), 'int')
        # Applying the binary operator '-' (line 198)
        result_sub_105037 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 27), '-', ndims_105035, int_105036)
        
        int_105038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 36), 'int')
        int_105039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 40), 'int')
        # Processing the call keyword arguments (line 198)
        kwargs_105040 = {}
        # Getting the type of 'range' (line 198)
        range_105034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'range', False)
        # Calling range(args, kwargs) (line 198)
        range_call_result_105041 = invoke(stypy.reporting.localization.Localization(__file__, 198, 21), range_105034, *[result_sub_105037, int_105038, int_105039], **kwargs_105040)
        
        # Testing the type of a for loop iterable (line 198)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 198, 12), range_call_result_105041)
        # Getting the type of the for loop variable (line 198)
        for_loop_var_105042 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 198, 12), range_call_result_105041)
        # Assigning a type to the variable 'i' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'i', for_loop_var_105042)
        # SSA begins for a for statement (line 198)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'count' (line 201)
        count_105043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 19), 'count')
        int_105044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 28), 'int')
        # Applying the binary operator '==' (line 201)
        result_eq_105045 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 19), '==', count_105043, int_105044)
        
        # Testing the type of an if condition (line 201)
        if_condition_105046 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 16), result_eq_105045)
        # Assigning a type to the variable 'if_condition_105046' (line 201)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 16), 'if_condition_105046', if_condition_105046)
        # SSA begins for if statement (line 201)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 202):
        
        # Assigning a BinOp to a Subscript (line 202):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 202)
        i_105047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 36), 'i')
        # Getting the type of 'start' (line 202)
        start_105048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 30), 'start')
        # Obtaining the member '__getitem__' of a type (line 202)
        getitem___105049 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 30), start_105048, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 202)
        subscript_call_result_105050 = invoke(stypy.reporting.localization.Localization(__file__, 202, 30), getitem___105049, i_105047)
        
        int_105051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 39), 'int')
        # Applying the binary operator '+' (line 202)
        result_add_105052 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 30), '+', subscript_call_result_105050, int_105051)
        
        # Getting the type of 'stop' (line 202)
        stop_105053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'stop')
        # Getting the type of 'i' (line 202)
        i_105054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 25), 'i')
        # Storing an element on a container (line 202)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 20), stop_105053, (i_105054, result_add_105052))
        # SSA branch for the else part of an if statement (line 201)
        module_type_store.open_ssa_branch('else')
        
        
        # Getting the type of 'count' (line 203)
        count_105055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'count')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 203)
        i_105056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 41), 'i')
        # Getting the type of 'self' (line 203)
        self_105057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 30), 'self')
        # Obtaining the member 'shape' of a type (line 203)
        shape_105058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 30), self_105057, 'shape')
        # Obtaining the member '__getitem__' of a type (line 203)
        getitem___105059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 30), shape_105058, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 203)
        subscript_call_result_105060 = invoke(stypy.reporting.localization.Localization(__file__, 203, 30), getitem___105059, i_105056)
        
        # Applying the binary operator '<=' (line 203)
        result_le_105061 = python_operator(stypy.reporting.localization.Localization(__file__, 203, 21), '<=', count_105055, subscript_call_result_105060)
        
        # Testing the type of an if condition (line 203)
        if_condition_105062 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 203, 21), result_le_105061)
        # Assigning a type to the variable 'if_condition_105062' (line 203)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 21), 'if_condition_105062', if_condition_105062)
        # SSA begins for if statement (line 203)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a BinOp to a Subscript (line 205):
        
        # Assigning a BinOp to a Subscript (line 205):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 205)
        i_105063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 36), 'i')
        # Getting the type of 'start' (line 205)
        start_105064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 30), 'start')
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___105065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 30), start_105064, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_105066 = invoke(stypy.reporting.localization.Localization(__file__, 205, 30), getitem___105065, i_105063)
        
        # Getting the type of 'count' (line 205)
        count_105067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 41), 'count')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 205)
        i_105068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 52), 'i')
        # Getting the type of 'step' (line 205)
        step_105069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 47), 'step')
        # Obtaining the member '__getitem__' of a type (line 205)
        getitem___105070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 205, 47), step_105069, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 205)
        subscript_call_result_105071 = invoke(stypy.reporting.localization.Localization(__file__, 205, 47), getitem___105070, i_105068)
        
        # Applying the binary operator '*' (line 205)
        result_mul_105072 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 41), '*', count_105067, subscript_call_result_105071)
        
        # Applying the binary operator '+' (line 205)
        result_add_105073 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 30), '+', subscript_call_result_105066, result_mul_105072)
        
        # Getting the type of 'stop' (line 205)
        stop_105074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 20), 'stop')
        # Getting the type of 'i' (line 205)
        i_105075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 25), 'i')
        # Storing an element on a container (line 205)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 205, 20), stop_105074, (i_105075, result_add_105073))
        
        # Assigning a Name to a Name (line 206):
        
        # Assigning a Name to a Name (line 206):
        # Getting the type of 'i' (line 206)
        i_105076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 29), 'i')
        # Assigning a type to the variable 'rundim' (line 206)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'rundim', i_105076)
        # SSA branch for the else part of an if statement (line 203)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Subscript (line 209):
        
        # Assigning a Subscript to a Subscript (line 209):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 209)
        i_105077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 40), 'i')
        # Getting the type of 'self' (line 209)
        self_105078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 30), 'self')
        # Obtaining the member 'stop' of a type (line 209)
        stop_105079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 30), self_105078, 'stop')
        # Obtaining the member '__getitem__' of a type (line 209)
        getitem___105080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 30), stop_105079, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 209)
        subscript_call_result_105081 = invoke(stypy.reporting.localization.Localization(__file__, 209, 30), getitem___105080, i_105077)
        
        # Getting the type of 'stop' (line 209)
        stop_105082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 20), 'stop')
        # Getting the type of 'i' (line 209)
        i_105083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 25), 'i')
        # Storing an element on a container (line 209)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 20), stop_105082, (i_105083, subscript_call_result_105081))
        # SSA join for if statement (line 203)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 201)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Subscript (line 210):
        
        # Assigning a Call to a Subscript (line 210):
        
        # Call to min(...): (line 210)
        # Processing the call arguments (line 210)
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 210)
        i_105085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 40), 'i', False)
        # Getting the type of 'self' (line 210)
        self_105086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 30), 'self', False)
        # Obtaining the member 'stop' of a type (line 210)
        stop_105087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 30), self_105086, 'stop')
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___105088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 30), stop_105087, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_105089 = invoke(stypy.reporting.localization.Localization(__file__, 210, 30), getitem___105088, i_105085)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 210)
        i_105090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 49), 'i', False)
        # Getting the type of 'stop' (line 210)
        stop_105091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 44), 'stop', False)
        # Obtaining the member '__getitem__' of a type (line 210)
        getitem___105092 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 44), stop_105091, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 210)
        subscript_call_result_105093 = invoke(stypy.reporting.localization.Localization(__file__, 210, 44), getitem___105092, i_105090)
        
        # Processing the call keyword arguments (line 210)
        kwargs_105094 = {}
        # Getting the type of 'min' (line 210)
        min_105084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 26), 'min', False)
        # Calling min(args, kwargs) (line 210)
        min_call_result_105095 = invoke(stypy.reporting.localization.Localization(__file__, 210, 26), min_105084, *[subscript_call_result_105089, subscript_call_result_105093], **kwargs_105094)
        
        # Getting the type of 'stop' (line 210)
        stop_105096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 16), 'stop')
        # Getting the type of 'i' (line 210)
        i_105097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 21), 'i')
        # Storing an element on a container (line 210)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 210, 16), stop_105096, (i_105097, min_call_result_105095))
        
        # Assigning a BinOp to a Name (line 211):
        
        # Assigning a BinOp to a Name (line 211):
        # Getting the type of 'count' (line 211)
        count_105098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 24), 'count')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 211)
        i_105099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 42), 'i')
        # Getting the type of 'self' (line 211)
        self_105100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 31), 'self')
        # Obtaining the member 'shape' of a type (line 211)
        shape_105101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 31), self_105100, 'shape')
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___105102 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 31), shape_105101, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 211)
        subscript_call_result_105103 = invoke(stypy.reporting.localization.Localization(__file__, 211, 31), getitem___105102, i_105099)
        
        # Applying the binary operator '//' (line 211)
        result_floordiv_105104 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 24), '//', count_105098, subscript_call_result_105103)
        
        # Assigning a type to the variable 'count' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 16), 'count', result_floordiv_105104)
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 214):
        
        # Assigning a Call to a Name (line 214):
        
        # Call to tuple(...): (line 214)
        # Processing the call arguments (line 214)
        # Calculating generator expression
        module_type_store = module_type_store.open_function_context('list comprehension expression', 214, 27, True)
        # Calculating comprehension expression
        
        # Call to zip(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'start' (line 214)
        start_105111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 50), 'start', False)
        # Getting the type of 'stop' (line 214)
        stop_105112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 57), 'stop', False)
        # Getting the type of 'step' (line 214)
        step_105113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 63), 'step', False)
        # Processing the call keyword arguments (line 214)
        kwargs_105114 = {}
        # Getting the type of 'zip' (line 214)
        zip_105110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 46), 'zip', False)
        # Calling zip(args, kwargs) (line 214)
        zip_call_result_105115 = invoke(stypy.reporting.localization.Localization(__file__, 214, 46), zip_105110, *[start_105111, stop_105112, step_105113], **kwargs_105114)
        
        comprehension_105116 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 27), zip_call_result_105115)
        # Assigning a type to the variable 't' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 't', comprehension_105116)
        
        # Call to slice(...): (line 214)
        # Getting the type of 't' (line 214)
        t_105107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 't', False)
        # Processing the call keyword arguments (line 214)
        kwargs_105108 = {}
        # Getting the type of 'slice' (line 214)
        slice_105106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 27), 'slice', False)
        # Calling slice(args, kwargs) (line 214)
        slice_call_result_105109 = invoke(stypy.reporting.localization.Localization(__file__, 214, 27), slice_105106, *[t_105107], **kwargs_105108)
        
        list_105117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 214, 27), 'list')
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 214, 27), list_105117, slice_call_result_105109)
        # Processing the call keyword arguments (line 214)
        kwargs_105118 = {}
        # Getting the type of 'tuple' (line 214)
        tuple_105105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 21), 'tuple', False)
        # Calling tuple(args, kwargs) (line 214)
        tuple_call_result_105119 = invoke(stypy.reporting.localization.Localization(__file__, 214, 21), tuple_105105, *[list_105117], **kwargs_105118)
        
        # Assigning a type to the variable 'slice_' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'slice_', tuple_call_result_105119)
        # Creating a generator
        
        # Obtaining the type of the subscript
        # Getting the type of 'slice_' (line 215)
        slice__105120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 27), 'slice_')
        # Getting the type of 'self' (line 215)
        self_105121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 18), 'self')
        # Obtaining the member 'var' of a type (line 215)
        var_105122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 18), self_105121, 'var')
        # Obtaining the member '__getitem__' of a type (line 215)
        getitem___105123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 18), var_105122, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 215)
        subscript_call_result_105124 = invoke(stypy.reporting.localization.Localization(__file__, 215, 18), getitem___105123, slice__105120)
        
        GeneratorType_105125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 12), 'GeneratorType')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 12), GeneratorType_105125, subscript_call_result_105124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'stypy_return_type', GeneratorType_105125)
        
        # Assigning a Subscript to a Subscript (line 219):
        
        # Assigning a Subscript to a Subscript (line 219):
        
        # Obtaining the type of the subscript
        # Getting the type of 'rundim' (line 219)
        rundim_105126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 33), 'rundim')
        # Getting the type of 'stop' (line 219)
        stop_105127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 28), 'stop')
        # Obtaining the member '__getitem__' of a type (line 219)
        getitem___105128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 28), stop_105127, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 219)
        subscript_call_result_105129 = invoke(stypy.reporting.localization.Localization(__file__, 219, 28), getitem___105128, rundim_105126)
        
        # Getting the type of 'start' (line 219)
        start_105130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 12), 'start')
        # Getting the type of 'rundim' (line 219)
        rundim_105131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 18), 'rundim')
        # Storing an element on a container (line 219)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 219, 12), start_105130, (rundim_105131, subscript_call_result_105129))
        
        
        # Call to range(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'ndims' (line 220)
        ndims_105133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 27), 'ndims', False)
        int_105134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 33), 'int')
        # Applying the binary operator '-' (line 220)
        result_sub_105135 = python_operator(stypy.reporting.localization.Localization(__file__, 220, 27), '-', ndims_105133, int_105134)
        
        int_105136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 36), 'int')
        int_105137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 39), 'int')
        # Processing the call keyword arguments (line 220)
        kwargs_105138 = {}
        # Getting the type of 'range' (line 220)
        range_105132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 21), 'range', False)
        # Calling range(args, kwargs) (line 220)
        range_call_result_105139 = invoke(stypy.reporting.localization.Localization(__file__, 220, 21), range_105132, *[result_sub_105135, int_105136, int_105137], **kwargs_105138)
        
        # Testing the type of a for loop iterable (line 220)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 220, 12), range_call_result_105139)
        # Getting the type of the for loop variable (line 220)
        for_loop_var_105140 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 220, 12), range_call_result_105139)
        # Assigning a type to the variable 'i' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'i', for_loop_var_105140)
        # SSA begins for a for statement (line 220)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 221)
        i_105141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'i')
        # Getting the type of 'start' (line 221)
        start_105142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 19), 'start')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___105143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 19), start_105142, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_105144 = invoke(stypy.reporting.localization.Localization(__file__, 221, 19), getitem___105143, i_105141)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 221)
        i_105145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 41), 'i')
        # Getting the type of 'self' (line 221)
        self_105146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 31), 'self')
        # Obtaining the member 'stop' of a type (line 221)
        stop_105147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 31), self_105146, 'stop')
        # Obtaining the member '__getitem__' of a type (line 221)
        getitem___105148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 31), stop_105147, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 221)
        subscript_call_result_105149 = invoke(stypy.reporting.localization.Localization(__file__, 221, 31), getitem___105148, i_105145)
        
        # Applying the binary operator '>=' (line 221)
        result_ge_105150 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 19), '>=', subscript_call_result_105144, subscript_call_result_105149)
        
        # Testing the type of an if condition (line 221)
        if_condition_105151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 16), result_ge_105150)
        # Assigning a type to the variable 'if_condition_105151' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'if_condition_105151', if_condition_105151)
        # SSA begins for if statement (line 221)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Subscript (line 222):
        
        # Assigning a Subscript to a Subscript (line 222):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 222)
        i_105152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 42), 'i')
        # Getting the type of 'self' (line 222)
        self_105153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 31), 'self')
        # Obtaining the member 'start' of a type (line 222)
        start_105154 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 31), self_105153, 'start')
        # Obtaining the member '__getitem__' of a type (line 222)
        getitem___105155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 31), start_105154, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 222)
        subscript_call_result_105156 = invoke(stypy.reporting.localization.Localization(__file__, 222, 31), getitem___105155, i_105152)
        
        # Getting the type of 'start' (line 222)
        start_105157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'start')
        # Getting the type of 'i' (line 222)
        i_105158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 26), 'i')
        # Storing an element on a container (line 222)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 20), start_105157, (i_105158, subscript_call_result_105156))
        
        # Getting the type of 'start' (line 223)
        start_105159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'start')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 223)
        i_105160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'i')
        int_105161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 28), 'int')
        # Applying the binary operator '-' (line 223)
        result_sub_105162 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 26), '-', i_105160, int_105161)
        
        # Getting the type of 'start' (line 223)
        start_105163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'start')
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___105164 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 20), start_105163, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_105165 = invoke(stypy.reporting.localization.Localization(__file__, 223, 20), getitem___105164, result_sub_105162)
        
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 223)
        i_105166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 44), 'i')
        int_105167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 46), 'int')
        # Applying the binary operator '-' (line 223)
        result_sub_105168 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 44), '-', i_105166, int_105167)
        
        # Getting the type of 'self' (line 223)
        self_105169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 34), 'self')
        # Obtaining the member 'step' of a type (line 223)
        step_105170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 34), self_105169, 'step')
        # Obtaining the member '__getitem__' of a type (line 223)
        getitem___105171 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 223, 34), step_105170, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 223)
        subscript_call_result_105172 = invoke(stypy.reporting.localization.Localization(__file__, 223, 34), getitem___105171, result_sub_105168)
        
        # Applying the binary operator '+=' (line 223)
        result_iadd_105173 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 20), '+=', subscript_call_result_105165, subscript_call_result_105172)
        # Getting the type of 'start' (line 223)
        start_105174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'start')
        # Getting the type of 'i' (line 223)
        i_105175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 26), 'i')
        int_105176 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 223, 28), 'int')
        # Applying the binary operator '-' (line 223)
        result_sub_105177 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 26), '-', i_105175, int_105176)
        
        # Storing an element on a container (line 223)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 223, 20), start_105174, (result_sub_105177, result_iadd_105173))
        
        # SSA join for if statement (line 221)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Obtaining the type of the subscript
        int_105178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 21), 'int')
        # Getting the type of 'start' (line 224)
        start_105179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'start')
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___105180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 15), start_105179, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_105181 = invoke(stypy.reporting.localization.Localization(__file__, 224, 15), getitem___105180, int_105178)
        
        
        # Obtaining the type of the subscript
        int_105182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 37), 'int')
        # Getting the type of 'self' (line 224)
        self_105183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 27), 'self')
        # Obtaining the member 'stop' of a type (line 224)
        stop_105184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 27), self_105183, 'stop')
        # Obtaining the member '__getitem__' of a type (line 224)
        getitem___105185 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 27), stop_105184, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 224)
        subscript_call_result_105186 = invoke(stypy.reporting.localization.Localization(__file__, 224, 27), getitem___105185, int_105182)
        
        # Applying the binary operator '>=' (line 224)
        result_ge_105187 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 15), '>=', subscript_call_result_105181, subscript_call_result_105186)
        
        # Testing the type of an if condition (line 224)
        if_condition_105188 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 224, 12), result_ge_105187)
        # Assigning a type to the variable 'if_condition_105188' (line 224)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 12), 'if_condition_105188', if_condition_105188)
        # SSA begins for if statement (line 224)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 225)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 16), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 224)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for while statement (line 191)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__iter__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__iter__' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_105189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_105189)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__iter__'
        return stypy_return_type_105189


# Assigning a type to the variable 'Arrayterator' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'Arrayterator', Arrayterator)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
