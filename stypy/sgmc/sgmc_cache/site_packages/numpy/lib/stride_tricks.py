
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''
2: Utilities that manipulate strides to achieve desirable effects.
3: 
4: An explanation of strides can be found in the "ndarray.rst" file in the
5: NumPy reference guide.
6: 
7: '''
8: from __future__ import division, absolute_import, print_function
9: 
10: import numpy as np
11: 
12: __all__ = ['broadcast_to', 'broadcast_arrays']
13: 
14: 
15: class DummyArray(object):
16:     '''Dummy object that just exists to hang __array_interface__ dictionaries
17:     and possibly keep alive a reference to a base array.
18:     '''
19: 
20:     def __init__(self, interface, base=None):
21:         self.__array_interface__ = interface
22:         self.base = base
23: 
24: 
25: def _maybe_view_as_subclass(original_array, new_array):
26:     if type(original_array) is not type(new_array):
27:         # if input was an ndarray subclass and subclasses were OK,
28:         # then view the result as that subclass.
29:         new_array = new_array.view(type=type(original_array))
30:         # Since we have done something akin to a view from original_array, we
31:         # should let the subclass finalize (if it has it implemented, i.e., is
32:         # not None).
33:         if new_array.__array_finalize__:
34:             new_array.__array_finalize__(original_array)
35:     return new_array
36: 
37: 
38: def as_strided(x, shape=None, strides=None, subok=False):
39:     ''' Make an ndarray from the given array with the given shape and strides.
40:     '''
41:     # first convert input to array, possibly keeping subclass
42:     x = np.array(x, copy=False, subok=subok)
43:     interface = dict(x.__array_interface__)
44:     if shape is not None:
45:         interface['shape'] = tuple(shape)
46:     if strides is not None:
47:         interface['strides'] = tuple(strides)
48:     array = np.asarray(DummyArray(interface, base=x))
49: 
50:     if array.dtype.fields is None and x.dtype.fields is not None:
51:         # This should only happen if x.dtype is [('', 'Vx')]
52:         array.dtype = x.dtype
53: 
54:     return _maybe_view_as_subclass(x, array)
55: 
56: 
57: def _broadcast_to(array, shape, subok, readonly):
58:     shape = tuple(shape) if np.iterable(shape) else (shape,)
59:     array = np.array(array, copy=False, subok=subok)
60:     if not shape and array.shape:
61:         raise ValueError('cannot broadcast a non-scalar to a scalar array')
62:     if any(size < 0 for size in shape):
63:         raise ValueError('all elements of broadcast shape must be non-'
64:                          'negative')
65:     needs_writeable = not readonly and array.flags.writeable
66:     extras = ['reduce_ok'] if needs_writeable else []
67:     op_flag = 'readwrite' if needs_writeable else 'readonly'
68:     broadcast = np.nditer(
69:         (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras,
70:         op_flags=[op_flag], itershape=shape, order='C').itviews[0]
71:     result = _maybe_view_as_subclass(array, broadcast)
72:     if needs_writeable and not result.flags.writeable:
73:         result.flags.writeable = True
74:     return result
75: 
76: 
77: def broadcast_to(array, shape, subok=False):
78:     '''Broadcast an array to a new shape.
79: 
80:     Parameters
81:     ----------
82:     array : array_like
83:         The array to broadcast.
84:     shape : tuple
85:         The shape of the desired array.
86:     subok : bool, optional
87:         If True, then sub-classes will be passed-through, otherwise
88:         the returned array will be forced to be a base-class array (default).
89: 
90:     Returns
91:     -------
92:     broadcast : array
93:         A readonly view on the original array with the given shape. It is
94:         typically not contiguous. Furthermore, more than one element of a
95:         broadcasted array may refer to a single memory location.
96: 
97:     Raises
98:     ------
99:     ValueError
100:         If the array is not compatible with the new shape according to NumPy's
101:         broadcasting rules.
102: 
103:     Notes
104:     -----
105:     .. versionadded:: 1.10.0
106: 
107:     Examples
108:     --------
109:     >>> x = np.array([1, 2, 3])
110:     >>> np.broadcast_to(x, (3, 3))
111:     array([[1, 2, 3],
112:            [1, 2, 3],
113:            [1, 2, 3]])
114:     '''
115:     return _broadcast_to(array, shape, subok=subok, readonly=True)
116: 
117: 
118: def _broadcast_shape(*args):
119:     '''Returns the shape of the ararys that would result from broadcasting the
120:     supplied arrays against each other.
121:     '''
122:     if not args:
123:         raise ValueError('must provide at least one argument')
124:     # use the old-iterator because np.nditer does not handle size 0 arrays
125:     # consistently
126:     b = np.broadcast(*args[:32])
127:     # unfortunately, it cannot handle 32 or more arguments directly
128:     for pos in range(32, len(args), 31):
129:         # ironically, np.broadcast does not properly handle np.broadcast
130:         # objects (it treats them as scalars)
131:         # use broadcasting to avoid allocating the full array
132:         b = broadcast_to(0, b.shape)
133:         b = np.broadcast(b, *args[pos:(pos + 31)])
134:     return b.shape
135: 
136: 
137: def broadcast_arrays(*args, **kwargs):
138:     '''
139:     Broadcast any number of arrays against each other.
140: 
141:     Parameters
142:     ----------
143:     `*args` : array_likes
144:         The arrays to broadcast.
145: 
146:     subok : bool, optional
147:         If True, then sub-classes will be passed-through, otherwise
148:         the returned arrays will be forced to be a base-class array (default).
149: 
150:     Returns
151:     -------
152:     broadcasted : list of arrays
153:         These arrays are views on the original arrays.  They are typically
154:         not contiguous.  Furthermore, more than one element of a
155:         broadcasted array may refer to a single memory location.  If you
156:         need to write to the arrays, make copies first.
157: 
158:     Examples
159:     --------
160:     >>> x = np.array([[1,2,3]])
161:     >>> y = np.array([[1],[2],[3]])
162:     >>> np.broadcast_arrays(x, y)
163:     [array([[1, 2, 3],
164:            [1, 2, 3],
165:            [1, 2, 3]]), array([[1, 1, 1],
166:            [2, 2, 2],
167:            [3, 3, 3]])]
168: 
169:     Here is a useful idiom for getting contiguous copies instead of
170:     non-contiguous views.
171: 
172:     >>> [np.array(a) for a in np.broadcast_arrays(x, y)]
173:     [array([[1, 2, 3],
174:            [1, 2, 3],
175:            [1, 2, 3]]), array([[1, 1, 1],
176:            [2, 2, 2],
177:            [3, 3, 3]])]
178: 
179:     '''
180:     # nditer is not used here to avoid the limit of 32 arrays.
181:     # Otherwise, something like the following one-liner would suffice:
182:     # return np.nditer(args, flags=['multi_index', 'zerosize_ok'],
183:     #                  order='C').itviews
184: 
185:     subok = kwargs.pop('subok', False)
186:     if kwargs:
187:         raise TypeError('broadcast_arrays() got an unexpected keyword '
188:                         'argument {}'.format(kwargs.pop()))
189:     args = [np.array(_m, copy=False, subok=subok) for _m in args]
190: 
191:     shape = _broadcast_shape(*args)
192: 
193:     if all(array.shape == shape for array in args):
194:         # Common case where nothing needs to be broadcasted.
195:         return args
196: 
197:     # TODO: consider making the results of broadcast_arrays readonly to match
198:     # broadcast_to. This will require a deprecation cycle.
199:     return [_broadcast_to(array, shape, subok=subok, readonly=False)
200:             for array in args]
201: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_126296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', '\nUtilities that manipulate strides to achieve desirable effects.\n\nAn explanation of strides can be found in the "ndarray.rst" file in the\nNumPy reference guide.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import numpy' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_126297 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy')

if (type(import_126297) is not StypyTypeError):

    if (import_126297 != 'pyd_module'):
        __import__(import_126297)
        sys_modules_126298 = sys.modules[import_126297]
        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', sys_modules_126298.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'numpy', import_126297)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 12):
__all__ = ['broadcast_to', 'broadcast_arrays']
module_type_store.set_exportable_members(['broadcast_to', 'broadcast_arrays'])

# Obtaining an instance of the builtin type 'list' (line 12)
list_126299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 12)
# Adding element type (line 12)
str_126300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 11), 'str', 'broadcast_to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_126299, str_126300)
# Adding element type (line 12)
str_126301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 27), 'str', 'broadcast_arrays')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 10), list_126299, str_126301)

# Assigning a type to the variable '__all__' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '__all__', list_126299)
# Declaration of the 'DummyArray' class

class DummyArray(object, ):
    str_126302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', 'Dummy object that just exists to hang __array_interface__ dictionaries\n    and possibly keep alive a reference to a base array.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 20)
        None_126303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 39), 'None')
        defaults = [None_126303]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyArray.__init__', ['interface', 'base'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['interface', 'base'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a Name to a Attribute (line 21):
        # Getting the type of 'interface' (line 21)
        interface_126304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 35), 'interface')
        # Getting the type of 'self' (line 21)
        self_126305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'self')
        # Setting the type of the member '__array_interface__' of a type (line 21)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), self_126305, '__array_interface__', interface_126304)
        
        # Assigning a Name to a Attribute (line 22):
        # Getting the type of 'base' (line 22)
        base_126306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 20), 'base')
        # Getting the type of 'self' (line 22)
        self_126307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self')
        # Setting the type of the member 'base' of a type (line 22)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), self_126307, 'base', base_126306)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'DummyArray' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'DummyArray', DummyArray)

@norecursion
def _maybe_view_as_subclass(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_maybe_view_as_subclass'
    module_type_store = module_type_store.open_function_context('_maybe_view_as_subclass', 25, 0, False)
    
    # Passed parameters checking function
    _maybe_view_as_subclass.stypy_localization = localization
    _maybe_view_as_subclass.stypy_type_of_self = None
    _maybe_view_as_subclass.stypy_type_store = module_type_store
    _maybe_view_as_subclass.stypy_function_name = '_maybe_view_as_subclass'
    _maybe_view_as_subclass.stypy_param_names_list = ['original_array', 'new_array']
    _maybe_view_as_subclass.stypy_varargs_param_name = None
    _maybe_view_as_subclass.stypy_kwargs_param_name = None
    _maybe_view_as_subclass.stypy_call_defaults = defaults
    _maybe_view_as_subclass.stypy_call_varargs = varargs
    _maybe_view_as_subclass.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_maybe_view_as_subclass', ['original_array', 'new_array'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_maybe_view_as_subclass', localization, ['original_array', 'new_array'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_maybe_view_as_subclass(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 26)
    # Getting the type of 'original_array' (line 26)
    original_array_126308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'original_array')
    
    # Call to type(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'new_array' (line 26)
    new_array_126310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 40), 'new_array', False)
    # Processing the call keyword arguments (line 26)
    kwargs_126311 = {}
    # Getting the type of 'type' (line 26)
    type_126309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 35), 'type', False)
    # Calling type(args, kwargs) (line 26)
    type_call_result_126312 = invoke(stypy.reporting.localization.Localization(__file__, 26, 35), type_126309, *[new_array_126310], **kwargs_126311)
    
    
    (may_be_126313, more_types_in_union_126314) = may_not_be_type(original_array_126308, type_call_result_126312)

    if may_be_126313:

        if more_types_in_union_126314:
            # Runtime conditional SSA (line 26)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'original_array' (line 26)
        original_array_126315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'original_array')
        # Assigning a type to the variable 'original_array' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'original_array', remove_type_from_union(original_array_126315, type_call_result_126312))
        
        # Assigning a Call to a Name (line 29):
        
        # Call to view(...): (line 29)
        # Processing the call keyword arguments (line 29)
        
        # Call to type(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'original_array' (line 29)
        original_array_126319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 45), 'original_array', False)
        # Processing the call keyword arguments (line 29)
        kwargs_126320 = {}
        # Getting the type of 'type' (line 29)
        type_126318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 40), 'type', False)
        # Calling type(args, kwargs) (line 29)
        type_call_result_126321 = invoke(stypy.reporting.localization.Localization(__file__, 29, 40), type_126318, *[original_array_126319], **kwargs_126320)
        
        keyword_126322 = type_call_result_126321
        kwargs_126323 = {'type': keyword_126322}
        # Getting the type of 'new_array' (line 29)
        new_array_126316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 20), 'new_array', False)
        # Obtaining the member 'view' of a type (line 29)
        view_126317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 20), new_array_126316, 'view')
        # Calling view(args, kwargs) (line 29)
        view_call_result_126324 = invoke(stypy.reporting.localization.Localization(__file__, 29, 20), view_126317, *[], **kwargs_126323)
        
        # Assigning a type to the variable 'new_array' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'new_array', view_call_result_126324)
        
        # Getting the type of 'new_array' (line 33)
        new_array_126325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), 'new_array')
        # Obtaining the member '__array_finalize__' of a type (line 33)
        array_finalize___126326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 11), new_array_126325, '__array_finalize__')
        # Testing the type of an if condition (line 33)
        if_condition_126327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 8), array_finalize___126326)
        # Assigning a type to the variable 'if_condition_126327' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'if_condition_126327', if_condition_126327)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __array_finalize__(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'original_array' (line 34)
        original_array_126330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 41), 'original_array', False)
        # Processing the call keyword arguments (line 34)
        kwargs_126331 = {}
        # Getting the type of 'new_array' (line 34)
        new_array_126328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'new_array', False)
        # Obtaining the member '__array_finalize__' of a type (line 34)
        array_finalize___126329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 12), new_array_126328, '__array_finalize__')
        # Calling __array_finalize__(args, kwargs) (line 34)
        array_finalize___call_result_126332 = invoke(stypy.reporting.localization.Localization(__file__, 34, 12), array_finalize___126329, *[original_array_126330], **kwargs_126331)
        
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_126314:
            # SSA join for if statement (line 26)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'new_array' (line 35)
    new_array_126333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'new_array')
    # Assigning a type to the variable 'stypy_return_type' (line 35)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type', new_array_126333)
    
    # ################# End of '_maybe_view_as_subclass(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_maybe_view_as_subclass' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_126334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126334)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_maybe_view_as_subclass'
    return stypy_return_type_126334

# Assigning a type to the variable '_maybe_view_as_subclass' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), '_maybe_view_as_subclass', _maybe_view_as_subclass)

@norecursion
def as_strided(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 38)
    None_126335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 24), 'None')
    # Getting the type of 'None' (line 38)
    None_126336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 38), 'None')
    # Getting the type of 'False' (line 38)
    False_126337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 50), 'False')
    defaults = [None_126335, None_126336, False_126337]
    # Create a new context for function 'as_strided'
    module_type_store = module_type_store.open_function_context('as_strided', 38, 0, False)
    
    # Passed parameters checking function
    as_strided.stypy_localization = localization
    as_strided.stypy_type_of_self = None
    as_strided.stypy_type_store = module_type_store
    as_strided.stypy_function_name = 'as_strided'
    as_strided.stypy_param_names_list = ['x', 'shape', 'strides', 'subok']
    as_strided.stypy_varargs_param_name = None
    as_strided.stypy_kwargs_param_name = None
    as_strided.stypy_call_defaults = defaults
    as_strided.stypy_call_varargs = varargs
    as_strided.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'as_strided', ['x', 'shape', 'strides', 'subok'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'as_strided', localization, ['x', 'shape', 'strides', 'subok'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'as_strided(...)' code ##################

    str_126338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', ' Make an ndarray from the given array with the given shape and strides.\n    ')
    
    # Assigning a Call to a Name (line 42):
    
    # Call to array(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'x' (line 42)
    x_126341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 17), 'x', False)
    # Processing the call keyword arguments (line 42)
    # Getting the type of 'False' (line 42)
    False_126342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'False', False)
    keyword_126343 = False_126342
    # Getting the type of 'subok' (line 42)
    subok_126344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'subok', False)
    keyword_126345 = subok_126344
    kwargs_126346 = {'subok': keyword_126345, 'copy': keyword_126343}
    # Getting the type of 'np' (line 42)
    np_126339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'np', False)
    # Obtaining the member 'array' of a type (line 42)
    array_126340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 8), np_126339, 'array')
    # Calling array(args, kwargs) (line 42)
    array_call_result_126347 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), array_126340, *[x_126341], **kwargs_126346)
    
    # Assigning a type to the variable 'x' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'x', array_call_result_126347)
    
    # Assigning a Call to a Name (line 43):
    
    # Call to dict(...): (line 43)
    # Processing the call arguments (line 43)
    # Getting the type of 'x' (line 43)
    x_126349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'x', False)
    # Obtaining the member '__array_interface__' of a type (line 43)
    array_interface___126350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 21), x_126349, '__array_interface__')
    # Processing the call keyword arguments (line 43)
    kwargs_126351 = {}
    # Getting the type of 'dict' (line 43)
    dict_126348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'dict', False)
    # Calling dict(args, kwargs) (line 43)
    dict_call_result_126352 = invoke(stypy.reporting.localization.Localization(__file__, 43, 16), dict_126348, *[array_interface___126350], **kwargs_126351)
    
    # Assigning a type to the variable 'interface' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'interface', dict_call_result_126352)
    
    # Type idiom detected: calculating its left and rigth part (line 44)
    # Getting the type of 'shape' (line 44)
    shape_126353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'shape')
    # Getting the type of 'None' (line 44)
    None_126354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 20), 'None')
    
    (may_be_126355, more_types_in_union_126356) = may_not_be_none(shape_126353, None_126354)

    if may_be_126355:

        if more_types_in_union_126356:
            # Runtime conditional SSA (line 44)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 45):
        
        # Call to tuple(...): (line 45)
        # Processing the call arguments (line 45)
        # Getting the type of 'shape' (line 45)
        shape_126358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 35), 'shape', False)
        # Processing the call keyword arguments (line 45)
        kwargs_126359 = {}
        # Getting the type of 'tuple' (line 45)
        tuple_126357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 29), 'tuple', False)
        # Calling tuple(args, kwargs) (line 45)
        tuple_call_result_126360 = invoke(stypy.reporting.localization.Localization(__file__, 45, 29), tuple_126357, *[shape_126358], **kwargs_126359)
        
        # Getting the type of 'interface' (line 45)
        interface_126361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'interface')
        str_126362 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'str', 'shape')
        # Storing an element on a container (line 45)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 8), interface_126361, (str_126362, tuple_call_result_126360))

        if more_types_in_union_126356:
            # SSA join for if statement (line 44)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 46)
    # Getting the type of 'strides' (line 46)
    strides_126363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'strides')
    # Getting the type of 'None' (line 46)
    None_126364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 22), 'None')
    
    (may_be_126365, more_types_in_union_126366) = may_not_be_none(strides_126363, None_126364)

    if may_be_126365:

        if more_types_in_union_126366:
            # Runtime conditional SSA (line 46)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Subscript (line 47):
        
        # Call to tuple(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'strides' (line 47)
        strides_126368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'strides', False)
        # Processing the call keyword arguments (line 47)
        kwargs_126369 = {}
        # Getting the type of 'tuple' (line 47)
        tuple_126367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 31), 'tuple', False)
        # Calling tuple(args, kwargs) (line 47)
        tuple_call_result_126370 = invoke(stypy.reporting.localization.Localization(__file__, 47, 31), tuple_126367, *[strides_126368], **kwargs_126369)
        
        # Getting the type of 'interface' (line 47)
        interface_126371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'interface')
        str_126372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'str', 'strides')
        # Storing an element on a container (line 47)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 8), interface_126371, (str_126372, tuple_call_result_126370))

        if more_types_in_union_126366:
            # SSA join for if statement (line 46)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 48):
    
    # Call to asarray(...): (line 48)
    # Processing the call arguments (line 48)
    
    # Call to DummyArray(...): (line 48)
    # Processing the call arguments (line 48)
    # Getting the type of 'interface' (line 48)
    interface_126376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 34), 'interface', False)
    # Processing the call keyword arguments (line 48)
    # Getting the type of 'x' (line 48)
    x_126377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 50), 'x', False)
    keyword_126378 = x_126377
    kwargs_126379 = {'base': keyword_126378}
    # Getting the type of 'DummyArray' (line 48)
    DummyArray_126375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'DummyArray', False)
    # Calling DummyArray(args, kwargs) (line 48)
    DummyArray_call_result_126380 = invoke(stypy.reporting.localization.Localization(__file__, 48, 23), DummyArray_126375, *[interface_126376], **kwargs_126379)
    
    # Processing the call keyword arguments (line 48)
    kwargs_126381 = {}
    # Getting the type of 'np' (line 48)
    np_126373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'np', False)
    # Obtaining the member 'asarray' of a type (line 48)
    asarray_126374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 12), np_126373, 'asarray')
    # Calling asarray(args, kwargs) (line 48)
    asarray_call_result_126382 = invoke(stypy.reporting.localization.Localization(__file__, 48, 12), asarray_126374, *[DummyArray_call_result_126380], **kwargs_126381)
    
    # Assigning a type to the variable 'array' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'array', asarray_call_result_126382)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'array' (line 50)
    array_126383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 7), 'array')
    # Obtaining the member 'dtype' of a type (line 50)
    dtype_126384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 7), array_126383, 'dtype')
    # Obtaining the member 'fields' of a type (line 50)
    fields_126385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 7), dtype_126384, 'fields')
    # Getting the type of 'None' (line 50)
    None_126386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'None')
    # Applying the binary operator 'is' (line 50)
    result_is__126387 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 7), 'is', fields_126385, None_126386)
    
    
    # Getting the type of 'x' (line 50)
    x_126388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 38), 'x')
    # Obtaining the member 'dtype' of a type (line 50)
    dtype_126389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 38), x_126388, 'dtype')
    # Obtaining the member 'fields' of a type (line 50)
    fields_126390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 38), dtype_126389, 'fields')
    # Getting the type of 'None' (line 50)
    None_126391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 60), 'None')
    # Applying the binary operator 'isnot' (line 50)
    result_is_not_126392 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 38), 'isnot', fields_126390, None_126391)
    
    # Applying the binary operator 'and' (line 50)
    result_and_keyword_126393 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 7), 'and', result_is__126387, result_is_not_126392)
    
    # Testing the type of an if condition (line 50)
    if_condition_126394 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 4), result_and_keyword_126393)
    # Assigning a type to the variable 'if_condition_126394' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'if_condition_126394', if_condition_126394)
    # SSA begins for if statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Attribute (line 52):
    # Getting the type of 'x' (line 52)
    x_126395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 22), 'x')
    # Obtaining the member 'dtype' of a type (line 52)
    dtype_126396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 22), x_126395, 'dtype')
    # Getting the type of 'array' (line 52)
    array_126397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'array')
    # Setting the type of the member 'dtype' of a type (line 52)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), array_126397, 'dtype', dtype_126396)
    # SSA join for if statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _maybe_view_as_subclass(...): (line 54)
    # Processing the call arguments (line 54)
    # Getting the type of 'x' (line 54)
    x_126399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 35), 'x', False)
    # Getting the type of 'array' (line 54)
    array_126400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'array', False)
    # Processing the call keyword arguments (line 54)
    kwargs_126401 = {}
    # Getting the type of '_maybe_view_as_subclass' (line 54)
    _maybe_view_as_subclass_126398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), '_maybe_view_as_subclass', False)
    # Calling _maybe_view_as_subclass(args, kwargs) (line 54)
    _maybe_view_as_subclass_call_result_126402 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), _maybe_view_as_subclass_126398, *[x_126399, array_126400], **kwargs_126401)
    
    # Assigning a type to the variable 'stypy_return_type' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type', _maybe_view_as_subclass_call_result_126402)
    
    # ################# End of 'as_strided(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'as_strided' in the type store
    # Getting the type of 'stypy_return_type' (line 38)
    stypy_return_type_126403 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126403)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'as_strided'
    return stypy_return_type_126403

# Assigning a type to the variable 'as_strided' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'as_strided', as_strided)

@norecursion
def _broadcast_to(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_broadcast_to'
    module_type_store = module_type_store.open_function_context('_broadcast_to', 57, 0, False)
    
    # Passed parameters checking function
    _broadcast_to.stypy_localization = localization
    _broadcast_to.stypy_type_of_self = None
    _broadcast_to.stypy_type_store = module_type_store
    _broadcast_to.stypy_function_name = '_broadcast_to'
    _broadcast_to.stypy_param_names_list = ['array', 'shape', 'subok', 'readonly']
    _broadcast_to.stypy_varargs_param_name = None
    _broadcast_to.stypy_kwargs_param_name = None
    _broadcast_to.stypy_call_defaults = defaults
    _broadcast_to.stypy_call_varargs = varargs
    _broadcast_to.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_broadcast_to', ['array', 'shape', 'subok', 'readonly'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_broadcast_to', localization, ['array', 'shape', 'subok', 'readonly'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_broadcast_to(...)' code ##################

    
    # Assigning a IfExp to a Name (line 58):
    
    
    # Call to iterable(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'shape' (line 58)
    shape_126406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 40), 'shape', False)
    # Processing the call keyword arguments (line 58)
    kwargs_126407 = {}
    # Getting the type of 'np' (line 58)
    np_126404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'np', False)
    # Obtaining the member 'iterable' of a type (line 58)
    iterable_126405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 28), np_126404, 'iterable')
    # Calling iterable(args, kwargs) (line 58)
    iterable_call_result_126408 = invoke(stypy.reporting.localization.Localization(__file__, 58, 28), iterable_126405, *[shape_126406], **kwargs_126407)
    
    # Testing the type of an if expression (line 58)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 12), iterable_call_result_126408)
    # SSA begins for if expression (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to tuple(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'shape' (line 58)
    shape_126410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'shape', False)
    # Processing the call keyword arguments (line 58)
    kwargs_126411 = {}
    # Getting the type of 'tuple' (line 58)
    tuple_126409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 58)
    tuple_call_result_126412 = invoke(stypy.reporting.localization.Localization(__file__, 58, 12), tuple_126409, *[shape_126410], **kwargs_126411)
    
    # SSA branch for the else part of an if expression (line 58)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 58)
    tuple_126413 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 58)
    # Adding element type (line 58)
    # Getting the type of 'shape' (line 58)
    shape_126414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 53), 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 53), tuple_126413, shape_126414)
    
    # SSA join for if expression (line 58)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_126415 = union_type.UnionType.add(tuple_call_result_126412, tuple_126413)
    
    # Assigning a type to the variable 'shape' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'shape', if_exp_126415)
    
    # Assigning a Call to a Name (line 59):
    
    # Call to array(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'array' (line 59)
    array_126418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 21), 'array', False)
    # Processing the call keyword arguments (line 59)
    # Getting the type of 'False' (line 59)
    False_126419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 33), 'False', False)
    keyword_126420 = False_126419
    # Getting the type of 'subok' (line 59)
    subok_126421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 46), 'subok', False)
    keyword_126422 = subok_126421
    kwargs_126423 = {'subok': keyword_126422, 'copy': keyword_126420}
    # Getting the type of 'np' (line 59)
    np_126416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 59)
    array_126417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 12), np_126416, 'array')
    # Calling array(args, kwargs) (line 59)
    array_call_result_126424 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), array_126417, *[array_126418], **kwargs_126423)
    
    # Assigning a type to the variable 'array' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'array', array_call_result_126424)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 60)
    shape_126425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'shape')
    # Applying the 'not' unary operator (line 60)
    result_not__126426 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 7), 'not', shape_126425)
    
    # Getting the type of 'array' (line 60)
    array_126427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'array')
    # Obtaining the member 'shape' of a type (line 60)
    shape_126428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 21), array_126427, 'shape')
    # Applying the binary operator 'and' (line 60)
    result_and_keyword_126429 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 7), 'and', result_not__126426, shape_126428)
    
    # Testing the type of an if condition (line 60)
    if_condition_126430 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 4), result_and_keyword_126429)
    # Assigning a type to the variable 'if_condition_126430' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'if_condition_126430', if_condition_126430)
    # SSA begins for if statement (line 60)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 61)
    # Processing the call arguments (line 61)
    str_126432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 25), 'str', 'cannot broadcast a non-scalar to a scalar array')
    # Processing the call keyword arguments (line 61)
    kwargs_126433 = {}
    # Getting the type of 'ValueError' (line 61)
    ValueError_126431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 61)
    ValueError_call_result_126434 = invoke(stypy.reporting.localization.Localization(__file__, 61, 14), ValueError_126431, *[str_126432], **kwargs_126433)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 61, 8), ValueError_call_result_126434, 'raise parameter', BaseException)
    # SSA join for if statement (line 60)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 62)
    # Processing the call arguments (line 62)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 62, 11, True)
    # Calculating comprehension expression
    # Getting the type of 'shape' (line 62)
    shape_126439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 32), 'shape', False)
    comprehension_126440 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 11), shape_126439)
    # Assigning a type to the variable 'size' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'size', comprehension_126440)
    
    # Getting the type of 'size' (line 62)
    size_126436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 11), 'size', False)
    int_126437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'int')
    # Applying the binary operator '<' (line 62)
    result_lt_126438 = python_operator(stypy.reporting.localization.Localization(__file__, 62, 11), '<', size_126436, int_126437)
    
    list_126441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 11), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 11), list_126441, result_lt_126438)
    # Processing the call keyword arguments (line 62)
    kwargs_126442 = {}
    # Getting the type of 'any' (line 62)
    any_126435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 7), 'any', False)
    # Calling any(args, kwargs) (line 62)
    any_call_result_126443 = invoke(stypy.reporting.localization.Localization(__file__, 62, 7), any_126435, *[list_126441], **kwargs_126442)
    
    # Testing the type of an if condition (line 62)
    if_condition_126444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 62, 4), any_call_result_126443)
    # Assigning a type to the variable 'if_condition_126444' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'if_condition_126444', if_condition_126444)
    # SSA begins for if statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 63)
    # Processing the call arguments (line 63)
    str_126446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 25), 'str', 'all elements of broadcast shape must be non-negative')
    # Processing the call keyword arguments (line 63)
    kwargs_126447 = {}
    # Getting the type of 'ValueError' (line 63)
    ValueError_126445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 63)
    ValueError_call_result_126448 = invoke(stypy.reporting.localization.Localization(__file__, 63, 14), ValueError_126445, *[str_126446], **kwargs_126447)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 63, 8), ValueError_call_result_126448, 'raise parameter', BaseException)
    # SSA join for if statement (line 62)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 65):
    
    # Evaluating a boolean operation
    
    # Getting the type of 'readonly' (line 65)
    readonly_126449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 26), 'readonly')
    # Applying the 'not' unary operator (line 65)
    result_not__126450 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 22), 'not', readonly_126449)
    
    # Getting the type of 'array' (line 65)
    array_126451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 39), 'array')
    # Obtaining the member 'flags' of a type (line 65)
    flags_126452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 39), array_126451, 'flags')
    # Obtaining the member 'writeable' of a type (line 65)
    writeable_126453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 39), flags_126452, 'writeable')
    # Applying the binary operator 'and' (line 65)
    result_and_keyword_126454 = python_operator(stypy.reporting.localization.Localization(__file__, 65, 22), 'and', result_not__126450, writeable_126453)
    
    # Assigning a type to the variable 'needs_writeable' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'needs_writeable', result_and_keyword_126454)
    
    # Assigning a IfExp to a Name (line 66):
    
    # Getting the type of 'needs_writeable' (line 66)
    needs_writeable_126455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 30), 'needs_writeable')
    # Testing the type of an if expression (line 66)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 13), needs_writeable_126455)
    # SSA begins for if expression (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining an instance of the builtin type 'list' (line 66)
    list_126456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 66)
    # Adding element type (line 66)
    str_126457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 14), 'str', 'reduce_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 66, 13), list_126456, str_126457)
    
    # SSA branch for the else part of an if expression (line 66)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining an instance of the builtin type 'list' (line 66)
    list_126458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 66)
    
    # SSA join for if expression (line 66)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_126459 = union_type.UnionType.add(list_126456, list_126458)
    
    # Assigning a type to the variable 'extras' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'extras', if_exp_126459)
    
    # Assigning a IfExp to a Name (line 67):
    
    # Getting the type of 'needs_writeable' (line 67)
    needs_writeable_126460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'needs_writeable')
    # Testing the type of an if expression (line 67)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 14), needs_writeable_126460)
    # SSA begins for if expression (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_126461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 14), 'str', 'readwrite')
    # SSA branch for the else part of an if expression (line 67)
    module_type_store.open_ssa_branch('if expression else')
    str_126462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 50), 'str', 'readonly')
    # SSA join for if expression (line 67)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_126463 = union_type.UnionType.add(str_126461, str_126462)
    
    # Assigning a type to the variable 'op_flag' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'op_flag', if_exp_126463)
    
    # Assigning a Subscript to a Name (line 68):
    
    # Obtaining the type of the subscript
    int_126464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 64), 'int')
    
    # Call to nditer(...): (line 68)
    # Processing the call arguments (line 68)
    
    # Obtaining an instance of the builtin type 'tuple' (line 69)
    tuple_126467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'array' (line 69)
    array_126468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 9), 'array', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 9), tuple_126467, array_126468)
    
    # Processing the call keyword arguments (line 68)
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_126469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    # Adding element type (line 69)
    str_126470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 25), 'str', 'multi_index')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 24), list_126469, str_126470)
    # Adding element type (line 69)
    str_126471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 40), 'str', 'refs_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 24), list_126469, str_126471)
    # Adding element type (line 69)
    str_126472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 51), 'str', 'zerosize_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 24), list_126469, str_126472)
    
    # Getting the type of 'extras' (line 69)
    extras_126473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 68), 'extras', False)
    # Applying the binary operator '+' (line 69)
    result_add_126474 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 24), '+', list_126469, extras_126473)
    
    keyword_126475 = result_add_126474
    
    # Obtaining an instance of the builtin type 'list' (line 70)
    list_126476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 70)
    # Adding element type (line 70)
    # Getting the type of 'op_flag' (line 70)
    op_flag_126477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 18), 'op_flag', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 70, 17), list_126476, op_flag_126477)
    
    keyword_126478 = list_126476
    # Getting the type of 'shape' (line 70)
    shape_126479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'shape', False)
    keyword_126480 = shape_126479
    str_126481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 51), 'str', 'C')
    keyword_126482 = str_126481
    kwargs_126483 = {'itershape': keyword_126480, 'op_flags': keyword_126478, 'flags': keyword_126475, 'order': keyword_126482}
    # Getting the type of 'np' (line 68)
    np_126465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'np', False)
    # Obtaining the member 'nditer' of a type (line 68)
    nditer_126466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), np_126465, 'nditer')
    # Calling nditer(args, kwargs) (line 68)
    nditer_call_result_126484 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), nditer_126466, *[tuple_126467], **kwargs_126483)
    
    # Obtaining the member 'itviews' of a type (line 68)
    itviews_126485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), nditer_call_result_126484, 'itviews')
    # Obtaining the member '__getitem__' of a type (line 68)
    getitem___126486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 16), itviews_126485, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 68)
    subscript_call_result_126487 = invoke(stypy.reporting.localization.Localization(__file__, 68, 16), getitem___126486, int_126464)
    
    # Assigning a type to the variable 'broadcast' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'broadcast', subscript_call_result_126487)
    
    # Assigning a Call to a Name (line 71):
    
    # Call to _maybe_view_as_subclass(...): (line 71)
    # Processing the call arguments (line 71)
    # Getting the type of 'array' (line 71)
    array_126489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 37), 'array', False)
    # Getting the type of 'broadcast' (line 71)
    broadcast_126490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 44), 'broadcast', False)
    # Processing the call keyword arguments (line 71)
    kwargs_126491 = {}
    # Getting the type of '_maybe_view_as_subclass' (line 71)
    _maybe_view_as_subclass_126488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 13), '_maybe_view_as_subclass', False)
    # Calling _maybe_view_as_subclass(args, kwargs) (line 71)
    _maybe_view_as_subclass_call_result_126492 = invoke(stypy.reporting.localization.Localization(__file__, 71, 13), _maybe_view_as_subclass_126488, *[array_126489, broadcast_126490], **kwargs_126491)
    
    # Assigning a type to the variable 'result' (line 71)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'result', _maybe_view_as_subclass_call_result_126492)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'needs_writeable' (line 72)
    needs_writeable_126493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 7), 'needs_writeable')
    
    # Getting the type of 'result' (line 72)
    result_126494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'result')
    # Obtaining the member 'flags' of a type (line 72)
    flags_126495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 31), result_126494, 'flags')
    # Obtaining the member 'writeable' of a type (line 72)
    writeable_126496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 31), flags_126495, 'writeable')
    # Applying the 'not' unary operator (line 72)
    result_not__126497 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 27), 'not', writeable_126496)
    
    # Applying the binary operator 'and' (line 72)
    result_and_keyword_126498 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 7), 'and', needs_writeable_126493, result_not__126497)
    
    # Testing the type of an if condition (line 72)
    if_condition_126499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 4), result_and_keyword_126498)
    # Assigning a type to the variable 'if_condition_126499' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'if_condition_126499', if_condition_126499)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 73):
    # Getting the type of 'True' (line 73)
    True_126500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 33), 'True')
    # Getting the type of 'result' (line 73)
    result_126501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'result')
    # Obtaining the member 'flags' of a type (line 73)
    flags_126502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), result_126501, 'flags')
    # Setting the type of the member 'writeable' of a type (line 73)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), flags_126502, 'writeable', True_126500)
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 74)
    result_126503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 4), 'stypy_return_type', result_126503)
    
    # ################# End of '_broadcast_to(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_broadcast_to' in the type store
    # Getting the type of 'stypy_return_type' (line 57)
    stypy_return_type_126504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126504)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_broadcast_to'
    return stypy_return_type_126504

# Assigning a type to the variable '_broadcast_to' (line 57)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 0), '_broadcast_to', _broadcast_to)

@norecursion
def broadcast_to(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 77)
    False_126505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 37), 'False')
    defaults = [False_126505]
    # Create a new context for function 'broadcast_to'
    module_type_store = module_type_store.open_function_context('broadcast_to', 77, 0, False)
    
    # Passed parameters checking function
    broadcast_to.stypy_localization = localization
    broadcast_to.stypy_type_of_self = None
    broadcast_to.stypy_type_store = module_type_store
    broadcast_to.stypy_function_name = 'broadcast_to'
    broadcast_to.stypy_param_names_list = ['array', 'shape', 'subok']
    broadcast_to.stypy_varargs_param_name = None
    broadcast_to.stypy_kwargs_param_name = None
    broadcast_to.stypy_call_defaults = defaults
    broadcast_to.stypy_call_varargs = varargs
    broadcast_to.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'broadcast_to', ['array', 'shape', 'subok'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'broadcast_to', localization, ['array', 'shape', 'subok'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'broadcast_to(...)' code ##################

    str_126506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, (-1)), 'str', "Broadcast an array to a new shape.\n\n    Parameters\n    ----------\n    array : array_like\n        The array to broadcast.\n    shape : tuple\n        The shape of the desired array.\n    subok : bool, optional\n        If True, then sub-classes will be passed-through, otherwise\n        the returned array will be forced to be a base-class array (default).\n\n    Returns\n    -------\n    broadcast : array\n        A readonly view on the original array with the given shape. It is\n        typically not contiguous. Furthermore, more than one element of a\n        broadcasted array may refer to a single memory location.\n\n    Raises\n    ------\n    ValueError\n        If the array is not compatible with the new shape according to NumPy's\n        broadcasting rules.\n\n    Notes\n    -----\n    .. versionadded:: 1.10.0\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> np.broadcast_to(x, (3, 3))\n    array([[1, 2, 3],\n           [1, 2, 3],\n           [1, 2, 3]])\n    ")
    
    # Call to _broadcast_to(...): (line 115)
    # Processing the call arguments (line 115)
    # Getting the type of 'array' (line 115)
    array_126508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 25), 'array', False)
    # Getting the type of 'shape' (line 115)
    shape_126509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'shape', False)
    # Processing the call keyword arguments (line 115)
    # Getting the type of 'subok' (line 115)
    subok_126510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 45), 'subok', False)
    keyword_126511 = subok_126510
    # Getting the type of 'True' (line 115)
    True_126512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 61), 'True', False)
    keyword_126513 = True_126512
    kwargs_126514 = {'subok': keyword_126511, 'readonly': keyword_126513}
    # Getting the type of '_broadcast_to' (line 115)
    _broadcast_to_126507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 11), '_broadcast_to', False)
    # Calling _broadcast_to(args, kwargs) (line 115)
    _broadcast_to_call_result_126515 = invoke(stypy.reporting.localization.Localization(__file__, 115, 11), _broadcast_to_126507, *[array_126508, shape_126509], **kwargs_126514)
    
    # Assigning a type to the variable 'stypy_return_type' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 4), 'stypy_return_type', _broadcast_to_call_result_126515)
    
    # ################# End of 'broadcast_to(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'broadcast_to' in the type store
    # Getting the type of 'stypy_return_type' (line 77)
    stypy_return_type_126516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126516)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'broadcast_to'
    return stypy_return_type_126516

# Assigning a type to the variable 'broadcast_to' (line 77)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'broadcast_to', broadcast_to)

@norecursion
def _broadcast_shape(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_broadcast_shape'
    module_type_store = module_type_store.open_function_context('_broadcast_shape', 118, 0, False)
    
    # Passed parameters checking function
    _broadcast_shape.stypy_localization = localization
    _broadcast_shape.stypy_type_of_self = None
    _broadcast_shape.stypy_type_store = module_type_store
    _broadcast_shape.stypy_function_name = '_broadcast_shape'
    _broadcast_shape.stypy_param_names_list = []
    _broadcast_shape.stypy_varargs_param_name = 'args'
    _broadcast_shape.stypy_kwargs_param_name = None
    _broadcast_shape.stypy_call_defaults = defaults
    _broadcast_shape.stypy_call_varargs = varargs
    _broadcast_shape.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_broadcast_shape', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_broadcast_shape', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_broadcast_shape(...)' code ##################

    str_126517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, (-1)), 'str', 'Returns the shape of the ararys that would result from broadcasting the\n    supplied arrays against each other.\n    ')
    
    
    # Getting the type of 'args' (line 122)
    args_126518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 11), 'args')
    # Applying the 'not' unary operator (line 122)
    result_not__126519 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 7), 'not', args_126518)
    
    # Testing the type of an if condition (line 122)
    if_condition_126520 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 122, 4), result_not__126519)
    # Assigning a type to the variable 'if_condition_126520' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'if_condition_126520', if_condition_126520)
    # SSA begins for if statement (line 122)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 123)
    # Processing the call arguments (line 123)
    str_126522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 25), 'str', 'must provide at least one argument')
    # Processing the call keyword arguments (line 123)
    kwargs_126523 = {}
    # Getting the type of 'ValueError' (line 123)
    ValueError_126521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 123)
    ValueError_call_result_126524 = invoke(stypy.reporting.localization.Localization(__file__, 123, 14), ValueError_126521, *[str_126522], **kwargs_126523)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 123, 8), ValueError_call_result_126524, 'raise parameter', BaseException)
    # SSA join for if statement (line 122)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 126):
    
    # Call to broadcast(...): (line 126)
    
    # Obtaining the type of the subscript
    int_126527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 28), 'int')
    slice_126528 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 126, 22), None, int_126527, None)
    # Getting the type of 'args' (line 126)
    args_126529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 22), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 126)
    getitem___126530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 22), args_126529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 126)
    subscript_call_result_126531 = invoke(stypy.reporting.localization.Localization(__file__, 126, 22), getitem___126530, slice_126528)
    
    # Processing the call keyword arguments (line 126)
    kwargs_126532 = {}
    # Getting the type of 'np' (line 126)
    np_126525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'np', False)
    # Obtaining the member 'broadcast' of a type (line 126)
    broadcast_126526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 8), np_126525, 'broadcast')
    # Calling broadcast(args, kwargs) (line 126)
    broadcast_call_result_126533 = invoke(stypy.reporting.localization.Localization(__file__, 126, 8), broadcast_126526, *[subscript_call_result_126531], **kwargs_126532)
    
    # Assigning a type to the variable 'b' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'b', broadcast_call_result_126533)
    
    
    # Call to range(...): (line 128)
    # Processing the call arguments (line 128)
    int_126535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 21), 'int')
    
    # Call to len(...): (line 128)
    # Processing the call arguments (line 128)
    # Getting the type of 'args' (line 128)
    args_126537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 29), 'args', False)
    # Processing the call keyword arguments (line 128)
    kwargs_126538 = {}
    # Getting the type of 'len' (line 128)
    len_126536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 25), 'len', False)
    # Calling len(args, kwargs) (line 128)
    len_call_result_126539 = invoke(stypy.reporting.localization.Localization(__file__, 128, 25), len_126536, *[args_126537], **kwargs_126538)
    
    int_126540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 36), 'int')
    # Processing the call keyword arguments (line 128)
    kwargs_126541 = {}
    # Getting the type of 'range' (line 128)
    range_126534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'range', False)
    # Calling range(args, kwargs) (line 128)
    range_call_result_126542 = invoke(stypy.reporting.localization.Localization(__file__, 128, 15), range_126534, *[int_126535, len_call_result_126539, int_126540], **kwargs_126541)
    
    # Testing the type of a for loop iterable (line 128)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 128, 4), range_call_result_126542)
    # Getting the type of the for loop variable (line 128)
    for_loop_var_126543 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 128, 4), range_call_result_126542)
    # Assigning a type to the variable 'pos' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'pos', for_loop_var_126543)
    # SSA begins for a for statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 132):
    
    # Call to broadcast_to(...): (line 132)
    # Processing the call arguments (line 132)
    int_126545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 25), 'int')
    # Getting the type of 'b' (line 132)
    b_126546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 28), 'b', False)
    # Obtaining the member 'shape' of a type (line 132)
    shape_126547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 28), b_126546, 'shape')
    # Processing the call keyword arguments (line 132)
    kwargs_126548 = {}
    # Getting the type of 'broadcast_to' (line 132)
    broadcast_to_126544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'broadcast_to', False)
    # Calling broadcast_to(args, kwargs) (line 132)
    broadcast_to_call_result_126549 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), broadcast_to_126544, *[int_126545, shape_126547], **kwargs_126548)
    
    # Assigning a type to the variable 'b' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'b', broadcast_to_call_result_126549)
    
    # Assigning a Call to a Name (line 133):
    
    # Call to broadcast(...): (line 133)
    # Processing the call arguments (line 133)
    # Getting the type of 'b' (line 133)
    b_126552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 25), 'b', False)
    
    # Obtaining the type of the subscript
    # Getting the type of 'pos' (line 133)
    pos_126553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 34), 'pos', False)
    # Getting the type of 'pos' (line 133)
    pos_126554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 39), 'pos', False)
    int_126555 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 45), 'int')
    # Applying the binary operator '+' (line 133)
    result_add_126556 = python_operator(stypy.reporting.localization.Localization(__file__, 133, 39), '+', pos_126554, int_126555)
    
    slice_126557 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 133, 29), pos_126553, result_add_126556, None)
    # Getting the type of 'args' (line 133)
    args_126558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 29), 'args', False)
    # Obtaining the member '__getitem__' of a type (line 133)
    getitem___126559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 29), args_126558, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 133)
    subscript_call_result_126560 = invoke(stypy.reporting.localization.Localization(__file__, 133, 29), getitem___126559, slice_126557)
    
    # Processing the call keyword arguments (line 133)
    kwargs_126561 = {}
    # Getting the type of 'np' (line 133)
    np_126550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'np', False)
    # Obtaining the member 'broadcast' of a type (line 133)
    broadcast_126551 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 12), np_126550, 'broadcast')
    # Calling broadcast(args, kwargs) (line 133)
    broadcast_call_result_126562 = invoke(stypy.reporting.localization.Localization(__file__, 133, 12), broadcast_126551, *[b_126552, subscript_call_result_126560], **kwargs_126561)
    
    # Assigning a type to the variable 'b' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'b', broadcast_call_result_126562)
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'b' (line 134)
    b_126563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 11), 'b')
    # Obtaining the member 'shape' of a type (line 134)
    shape_126564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 11), b_126563, 'shape')
    # Assigning a type to the variable 'stypy_return_type' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'stypy_return_type', shape_126564)
    
    # ################# End of '_broadcast_shape(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_broadcast_shape' in the type store
    # Getting the type of 'stypy_return_type' (line 118)
    stypy_return_type_126565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126565)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_broadcast_shape'
    return stypy_return_type_126565

# Assigning a type to the variable '_broadcast_shape' (line 118)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 0), '_broadcast_shape', _broadcast_shape)

@norecursion
def broadcast_arrays(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'broadcast_arrays'
    module_type_store = module_type_store.open_function_context('broadcast_arrays', 137, 0, False)
    
    # Passed parameters checking function
    broadcast_arrays.stypy_localization = localization
    broadcast_arrays.stypy_type_of_self = None
    broadcast_arrays.stypy_type_store = module_type_store
    broadcast_arrays.stypy_function_name = 'broadcast_arrays'
    broadcast_arrays.stypy_param_names_list = []
    broadcast_arrays.stypy_varargs_param_name = 'args'
    broadcast_arrays.stypy_kwargs_param_name = 'kwargs'
    broadcast_arrays.stypy_call_defaults = defaults
    broadcast_arrays.stypy_call_varargs = varargs
    broadcast_arrays.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'broadcast_arrays', [], 'args', 'kwargs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'broadcast_arrays', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'broadcast_arrays(...)' code ##################

    str_126566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, (-1)), 'str', '\n    Broadcast any number of arrays against each other.\n\n    Parameters\n    ----------\n    `*args` : array_likes\n        The arrays to broadcast.\n\n    subok : bool, optional\n        If True, then sub-classes will be passed-through, otherwise\n        the returned arrays will be forced to be a base-class array (default).\n\n    Returns\n    -------\n    broadcasted : list of arrays\n        These arrays are views on the original arrays.  They are typically\n        not contiguous.  Furthermore, more than one element of a\n        broadcasted array may refer to a single memory location.  If you\n        need to write to the arrays, make copies first.\n\n    Examples\n    --------\n    >>> x = np.array([[1,2,3]])\n    >>> y = np.array([[1],[2],[3]])\n    >>> np.broadcast_arrays(x, y)\n    [array([[1, 2, 3],\n           [1, 2, 3],\n           [1, 2, 3]]), array([[1, 1, 1],\n           [2, 2, 2],\n           [3, 3, 3]])]\n\n    Here is a useful idiom for getting contiguous copies instead of\n    non-contiguous views.\n\n    >>> [np.array(a) for a in np.broadcast_arrays(x, y)]\n    [array([[1, 2, 3],\n           [1, 2, 3],\n           [1, 2, 3]]), array([[1, 1, 1],\n           [2, 2, 2],\n           [3, 3, 3]])]\n\n    ')
    
    # Assigning a Call to a Name (line 185):
    
    # Call to pop(...): (line 185)
    # Processing the call arguments (line 185)
    str_126569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 23), 'str', 'subok')
    # Getting the type of 'False' (line 185)
    False_126570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 32), 'False', False)
    # Processing the call keyword arguments (line 185)
    kwargs_126571 = {}
    # Getting the type of 'kwargs' (line 185)
    kwargs_126567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 12), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 185)
    pop_126568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 185, 12), kwargs_126567, 'pop')
    # Calling pop(args, kwargs) (line 185)
    pop_call_result_126572 = invoke(stypy.reporting.localization.Localization(__file__, 185, 12), pop_126568, *[str_126569, False_126570], **kwargs_126571)
    
    # Assigning a type to the variable 'subok' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 4), 'subok', pop_call_result_126572)
    
    # Getting the type of 'kwargs' (line 186)
    kwargs_126573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 7), 'kwargs')
    # Testing the type of an if condition (line 186)
    if_condition_126574 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 4), kwargs_126573)
    # Assigning a type to the variable 'if_condition_126574' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'if_condition_126574', if_condition_126574)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to TypeError(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Call to format(...): (line 187)
    # Processing the call arguments (line 187)
    
    # Call to pop(...): (line 188)
    # Processing the call keyword arguments (line 188)
    kwargs_126580 = {}
    # Getting the type of 'kwargs' (line 188)
    kwargs_126578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 45), 'kwargs', False)
    # Obtaining the member 'pop' of a type (line 188)
    pop_126579 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 45), kwargs_126578, 'pop')
    # Calling pop(args, kwargs) (line 188)
    pop_call_result_126581 = invoke(stypy.reporting.localization.Localization(__file__, 188, 45), pop_126579, *[], **kwargs_126580)
    
    # Processing the call keyword arguments (line 187)
    kwargs_126582 = {}
    str_126576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 24), 'str', 'broadcast_arrays() got an unexpected keyword argument {}')
    # Obtaining the member 'format' of a type (line 187)
    format_126577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 187, 24), str_126576, 'format')
    # Calling format(args, kwargs) (line 187)
    format_call_result_126583 = invoke(stypy.reporting.localization.Localization(__file__, 187, 24), format_126577, *[pop_call_result_126581], **kwargs_126582)
    
    # Processing the call keyword arguments (line 187)
    kwargs_126584 = {}
    # Getting the type of 'TypeError' (line 187)
    TypeError_126575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 14), 'TypeError', False)
    # Calling TypeError(args, kwargs) (line 187)
    TypeError_call_result_126585 = invoke(stypy.reporting.localization.Localization(__file__, 187, 14), TypeError_126575, *[format_call_result_126583], **kwargs_126584)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 187, 8), TypeError_call_result_126585, 'raise parameter', BaseException)
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a ListComp to a Name (line 189):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'args' (line 189)
    args_126595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 60), 'args')
    comprehension_126596 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), args_126595)
    # Assigning a type to the variable '_m' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), '_m', comprehension_126596)
    
    # Call to array(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of '_m' (line 189)
    _m_126588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 21), '_m', False)
    # Processing the call keyword arguments (line 189)
    # Getting the type of 'False' (line 189)
    False_126589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 30), 'False', False)
    keyword_126590 = False_126589
    # Getting the type of 'subok' (line 189)
    subok_126591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 43), 'subok', False)
    keyword_126592 = subok_126591
    kwargs_126593 = {'subok': keyword_126592, 'copy': keyword_126590}
    # Getting the type of 'np' (line 189)
    np_126586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 189)
    array_126587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 12), np_126586, 'array')
    # Calling array(args, kwargs) (line 189)
    array_call_result_126594 = invoke(stypy.reporting.localization.Localization(__file__, 189, 12), array_126587, *[_m_126588], **kwargs_126593)
    
    list_126597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 189, 12), list_126597, array_call_result_126594)
    # Assigning a type to the variable 'args' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'args', list_126597)
    
    # Assigning a Call to a Name (line 191):
    
    # Call to _broadcast_shape(...): (line 191)
    # Getting the type of 'args' (line 191)
    args_126599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 30), 'args', False)
    # Processing the call keyword arguments (line 191)
    kwargs_126600 = {}
    # Getting the type of '_broadcast_shape' (line 191)
    _broadcast_shape_126598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), '_broadcast_shape', False)
    # Calling _broadcast_shape(args, kwargs) (line 191)
    _broadcast_shape_call_result_126601 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), _broadcast_shape_126598, *[args_126599], **kwargs_126600)
    
    # Assigning a type to the variable 'shape' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'shape', _broadcast_shape_call_result_126601)
    
    
    # Call to all(...): (line 193)
    # Processing the call arguments (line 193)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 193, 11, True)
    # Calculating comprehension expression
    # Getting the type of 'args' (line 193)
    args_126607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 45), 'args', False)
    comprehension_126608 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 11), args_126607)
    # Assigning a type to the variable 'array' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'array', comprehension_126608)
    
    # Getting the type of 'array' (line 193)
    array_126603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 11), 'array', False)
    # Obtaining the member 'shape' of a type (line 193)
    shape_126604 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 11), array_126603, 'shape')
    # Getting the type of 'shape' (line 193)
    shape_126605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 26), 'shape', False)
    # Applying the binary operator '==' (line 193)
    result_eq_126606 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 11), '==', shape_126604, shape_126605)
    
    list_126609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 11), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 193, 11), list_126609, result_eq_126606)
    # Processing the call keyword arguments (line 193)
    kwargs_126610 = {}
    # Getting the type of 'all' (line 193)
    all_126602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 7), 'all', False)
    # Calling all(args, kwargs) (line 193)
    all_call_result_126611 = invoke(stypy.reporting.localization.Localization(__file__, 193, 7), all_126602, *[list_126609], **kwargs_126610)
    
    # Testing the type of an if condition (line 193)
    if_condition_126612 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 193, 4), all_call_result_126611)
    # Assigning a type to the variable 'if_condition_126612' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 4), 'if_condition_126612', if_condition_126612)
    # SSA begins for if statement (line 193)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'args' (line 195)
    args_126613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 15), 'args')
    # Assigning a type to the variable 'stypy_return_type' (line 195)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 195, 8), 'stypy_return_type', args_126613)
    # SSA join for if statement (line 193)
    module_type_store = module_type_store.join_ssa_context()
    
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'args' (line 200)
    args_126623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 25), 'args')
    comprehension_126624 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 12), args_126623)
    # Assigning a type to the variable 'array' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), 'array', comprehension_126624)
    
    # Call to _broadcast_to(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'array' (line 199)
    array_126615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'array', False)
    # Getting the type of 'shape' (line 199)
    shape_126616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 33), 'shape', False)
    # Processing the call keyword arguments (line 199)
    # Getting the type of 'subok' (line 199)
    subok_126617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 46), 'subok', False)
    keyword_126618 = subok_126617
    # Getting the type of 'False' (line 199)
    False_126619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 62), 'False', False)
    keyword_126620 = False_126619
    kwargs_126621 = {'subok': keyword_126618, 'readonly': keyword_126620}
    # Getting the type of '_broadcast_to' (line 199)
    _broadcast_to_126614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 12), '_broadcast_to', False)
    # Calling _broadcast_to(args, kwargs) (line 199)
    _broadcast_to_call_result_126622 = invoke(stypy.reporting.localization.Localization(__file__, 199, 12), _broadcast_to_126614, *[array_126615, shape_126616], **kwargs_126621)
    
    list_126625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 12), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 199, 12), list_126625, _broadcast_to_call_result_126622)
    # Assigning a type to the variable 'stypy_return_type' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'stypy_return_type', list_126625)
    
    # ################# End of 'broadcast_arrays(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'broadcast_arrays' in the type store
    # Getting the type of 'stypy_return_type' (line 137)
    stypy_return_type_126626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_126626)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'broadcast_arrays'
    return stypy_return_type_126626

# Assigning a type to the variable 'broadcast_arrays' (line 137)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 0), 'broadcast_arrays', broadcast_arrays)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
