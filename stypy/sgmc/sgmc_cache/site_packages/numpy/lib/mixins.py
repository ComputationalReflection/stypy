
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Mixin classes for custom array types that don't inherit from ndarray.'''
2: from __future__ import division, absolute_import, print_function
3: 
4: import sys
5: 
6: from numpy.core import umath as um
7: 
8: # Nothing should be exposed in the top-level NumPy module.
9: __all__ = []
10: 
11: 
12: def _disables_array_ufunc(obj):
13:     '''True when __array_ufunc__ is set to None.'''
14:     try:
15:         return obj.__array_ufunc__ is None
16:     except AttributeError:
17:         return False
18: 
19: 
20: def _binary_method(ufunc, name):
21:     '''Implement a forward binary method with a ufunc, e.g., __add__.'''
22:     def func(self, other):
23:         if _disables_array_ufunc(other):
24:             return NotImplemented
25:         return ufunc(self, other)
26:     func.__name__ = '__{}__'.format(name)
27:     return func
28: 
29: 
30: def _reflected_binary_method(ufunc, name):
31:     '''Implement a reflected binary method with a ufunc, e.g., __radd__.'''
32:     def func(self, other):
33:         if _disables_array_ufunc(other):
34:             return NotImplemented
35:         return ufunc(other, self)
36:     func.__name__ = '__r{}__'.format(name)
37:     return func
38: 
39: 
40: def _inplace_binary_method(ufunc, name):
41:     '''Implement an in-place binary method with a ufunc, e.g., __iadd__.'''
42:     def func(self, other):
43:         return ufunc(self, other, out=(self,))
44:     func.__name__ = '__i{}__'.format(name)
45:     return func
46: 
47: 
48: def _numeric_methods(ufunc, name):
49:     '''Implement forward, reflected and inplace binary methods with a ufunc.'''
50:     return (_binary_method(ufunc, name),
51:             _reflected_binary_method(ufunc, name),
52:             _inplace_binary_method(ufunc, name))
53: 
54: 
55: def _unary_method(ufunc, name):
56:     '''Implement a unary special method with a ufunc.'''
57:     def func(self):
58:         return ufunc(self)
59:     func.__name__ = '__{}__'.format(name)
60:     return func
61: 
62: 
63: class NDArrayOperatorsMixin(object):
64:     '''Mixin defining all operator special methods using __array_ufunc__.
65: 
66:     This class implements the special methods for almost all of Python's
67:     builtin operators defined in the `operator` module, including comparisons
68:     (``==``, ``>``, etc.) and arithmetic (``+``, ``*``, ``-``, etc.), by
69:     deferring to the ``__array_ufunc__`` method, which subclasses must
70:     implement.
71: 
72:     This class does not yet implement the special operators corresponding
73:     to ``matmul`` (``@``), because ``np.matmul`` is not yet a NumPy ufunc.
74: 
75:     It is useful for writing classes that do not inherit from `numpy.ndarray`,
76:     but that should support arithmetic and numpy universal functions like
77:     arrays as described in :ref:`A Mechanism for Overriding Ufuncs
78:     <neps.ufunc-overrides>`.
79: 
80:     As an trivial example, consider this implementation of an ``ArrayLike``
81:     class that simply wraps a NumPy array and ensures that the result of any
82:     arithmetic operation is also an ``ArrayLike`` object::
83: 
84:         class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
85:             def __init__(self, value):
86:                 self.value = np.asarray(value)
87: 
88:             # One might also consider adding the built-in list type to this
89:             # list, to support operations like np.add(array_like, list)
90:             _HANDLED_TYPES = (np.ndarray, numbers.Number)
91: 
92:             def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
93:                 out = kwargs.get('out', ())
94:                 for x in inputs + out:
95:                     # Only support operations with instances of _HANDLED_TYPES.
96:                     # Use ArrayLike instead of type(self) for isinstance to
97:                     # allow subclasses that don't override __array_ufunc__ to
98:                     # handle ArrayLike objects.
99:                     if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):
100:                         return NotImplemented
101: 
102:                 # Defer to the implementation of the ufunc on unwrapped values.
103:                 inputs = tuple(x.value if isinstance(x, ArrayLike) else x
104:                                for x in inputs)
105:                 if out:
106:                     kwargs['out'] = tuple(
107:                         x.value if isinstance(x, ArrayLike) else x
108:                         for x in out)
109:                 result = getattr(ufunc, method)(*inputs, **kwargs)
110: 
111:                 if type(result) is tuple:
112:                     # multiple return values
113:                     return tuple(type(self)(x) for x in result)
114:                 elif method == 'at':
115:                     # no return value
116:                     return None
117:                 else:
118:                     # one return value
119:                     return type(self)(result)
120: 
121:             def __repr__(self):
122:                 return '%s(%r)' % (type(self).__name__, self.value)
123: 
124:     In interactions between ``ArrayLike`` objects and numbers or numpy arrays,
125:     the result is always another ``ArrayLike``:
126: 
127:         >>> x = ArrayLike([1, 2, 3])
128:         >>> x - 1
129:         ArrayLike(array([0, 1, 2]))
130:         >>> 1 - x
131:         ArrayLike(array([ 0, -1, -2]))
132:         >>> np.arange(3) - x
133:         ArrayLike(array([-1, -1, -1]))
134:         >>> x - np.arange(3)
135:         ArrayLike(array([1, 1, 1]))
136: 
137:     Note that unlike ``numpy.ndarray``, ``ArrayLike`` does not allow operations
138:     with arbitrary, unrecognized types. This ensures that interactions with
139:     ArrayLike preserve a well-defined casting hierarchy.
140:     '''
141:     # Like np.ndarray, this mixin class implements "Option 1" from the ufunc
142:     # overrides NEP.
143: 
144:     # comparisons don't have reflected and in-place versions
145:     __lt__ = _binary_method(um.less, 'lt')
146:     __le__ = _binary_method(um.less_equal, 'le')
147:     __eq__ = _binary_method(um.equal, 'eq')
148:     __ne__ = _binary_method(um.not_equal, 'ne')
149:     __gt__ = _binary_method(um.greater, 'gt')
150:     __ge__ = _binary_method(um.greater_equal, 'ge')
151: 
152:     # numeric methods
153:     __add__, __radd__, __iadd__ = _numeric_methods(um.add, 'add')
154:     __sub__, __rsub__, __isub__ = _numeric_methods(um.subtract, 'sub')
155:     __mul__, __rmul__, __imul__ = _numeric_methods(um.multiply, 'mul')
156:     if sys.version_info.major < 3:
157:         # Python 3 uses only __truediv__ and __floordiv__
158:         __div__, __rdiv__, __idiv__ = _numeric_methods(um.divide, 'div')
159:     __truediv__, __rtruediv__, __itruediv__ = _numeric_methods(
160:         um.true_divide, 'truediv')
161:     __floordiv__, __rfloordiv__, __ifloordiv__ = _numeric_methods(
162:         um.floor_divide, 'floordiv')
163:     __mod__, __rmod__, __imod__ = _numeric_methods(um.remainder, 'mod')
164:     __divmod__ = _binary_method(um.divmod, 'divmod')
165:     __rdivmod__ = _reflected_binary_method(um.divmod, 'divmod')
166:     # __idivmod__ does not exist
167:     # TODO: handle the optional third argument for __pow__?
168:     __pow__, __rpow__, __ipow__ = _numeric_methods(um.power, 'pow')
169:     __lshift__, __rlshift__, __ilshift__ = _numeric_methods(
170:         um.left_shift, 'lshift')
171:     __rshift__, __rrshift__, __irshift__ = _numeric_methods(
172:         um.right_shift, 'rshift')
173:     __and__, __rand__, __iand__ = _numeric_methods(um.bitwise_and, 'and')
174:     __xor__, __rxor__, __ixor__ = _numeric_methods(um.bitwise_xor, 'xor')
175:     __or__, __ror__, __ior__ = _numeric_methods(um.bitwise_or, 'or')
176: 
177:     # unary methods
178:     __neg__ = _unary_method(um.negative, 'neg')
179:     __pos__ = _unary_method(um.positive, 'pos')
180:     __abs__ = _unary_method(um.absolute, 'abs')
181:     __invert__ = _unary_method(um.invert, 'invert')
182: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1868 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', "Mixin classes for custom array types that don't inherit from ndarray.")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from numpy.core import um' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/lib/')
import_1869 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core')

if (type(import_1869) is not StypyTypeError):

    if (import_1869 != 'pyd_module'):
        __import__(import_1869)
        sys_modules_1870 = sys.modules[import_1869]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core', sys_modules_1870.module_type_store, module_type_store, ['umath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_1870, sys_modules_1870.module_type_store, module_type_store)
    else:
        from numpy.core import umath as um

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core', None, module_type_store, ['umath'], [um])

else:
    # Assigning a type to the variable 'numpy.core' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy.core', import_1869)

# Adding an alias
module_type_store.add_alias('um', 'umath')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/lib/')


# Assigning a List to a Name (line 9):

# Assigning a List to a Name (line 9):
__all__ = []
module_type_store.set_exportable_members([])

# Obtaining an instance of the builtin type 'list' (line 9)
list_1871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 10), 'list')
# Adding type elements to the builtin type 'list' instance (line 9)

# Assigning a type to the variable '__all__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__all__', list_1871)

@norecursion
def _disables_array_ufunc(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_disables_array_ufunc'
    module_type_store = module_type_store.open_function_context('_disables_array_ufunc', 12, 0, False)
    
    # Passed parameters checking function
    _disables_array_ufunc.stypy_localization = localization
    _disables_array_ufunc.stypy_type_of_self = None
    _disables_array_ufunc.stypy_type_store = module_type_store
    _disables_array_ufunc.stypy_function_name = '_disables_array_ufunc'
    _disables_array_ufunc.stypy_param_names_list = ['obj']
    _disables_array_ufunc.stypy_varargs_param_name = None
    _disables_array_ufunc.stypy_kwargs_param_name = None
    _disables_array_ufunc.stypy_call_defaults = defaults
    _disables_array_ufunc.stypy_call_varargs = varargs
    _disables_array_ufunc.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_disables_array_ufunc', ['obj'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_disables_array_ufunc', localization, ['obj'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_disables_array_ufunc(...)' code ##################

    str_1872 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 4), 'str', 'True when __array_ufunc__ is set to None.')
    
    
    # SSA begins for try-except statement (line 14)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Getting the type of 'obj' (line 15)
    obj_1873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'obj')
    # Obtaining the member '__array_ufunc__' of a type (line 15)
    array_ufunc___1874 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 15), obj_1873, '__array_ufunc__')
    # Getting the type of 'None' (line 15)
    None_1875 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 38), 'None')
    # Applying the binary operator 'is' (line 15)
    result_is__1876 = python_operator(stypy.reporting.localization.Localization(__file__, 15, 15), 'is', array_ufunc___1874, None_1875)
    
    # Assigning a type to the variable 'stypy_return_type' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', result_is__1876)
    # SSA branch for the except part of a try statement (line 14)
    # SSA branch for the except 'AttributeError' branch of a try statement (line 14)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'False' (line 17)
    False_1877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 15), 'False')
    # Assigning a type to the variable 'stypy_return_type' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'stypy_return_type', False_1877)
    # SSA join for try-except statement (line 14)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_disables_array_ufunc(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_disables_array_ufunc' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_1878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_disables_array_ufunc'
    return stypy_return_type_1878

# Assigning a type to the variable '_disables_array_ufunc' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '_disables_array_ufunc', _disables_array_ufunc)

@norecursion
def _binary_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_binary_method'
    module_type_store = module_type_store.open_function_context('_binary_method', 20, 0, False)
    
    # Passed parameters checking function
    _binary_method.stypy_localization = localization
    _binary_method.stypy_type_of_self = None
    _binary_method.stypy_type_store = module_type_store
    _binary_method.stypy_function_name = '_binary_method'
    _binary_method.stypy_param_names_list = ['ufunc', 'name']
    _binary_method.stypy_varargs_param_name = None
    _binary_method.stypy_kwargs_param_name = None
    _binary_method.stypy_call_defaults = defaults
    _binary_method.stypy_call_varargs = varargs
    _binary_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_binary_method', ['ufunc', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_binary_method', localization, ['ufunc', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_binary_method(...)' code ##################

    str_1879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 4), 'str', 'Implement a forward binary method with a ufunc, e.g., __add__.')

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 22, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['self', 'other']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['self', 'other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['self', 'other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        
        # Call to _disables_array_ufunc(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'other' (line 23)
        other_1881 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 33), 'other', False)
        # Processing the call keyword arguments (line 23)
        kwargs_1882 = {}
        # Getting the type of '_disables_array_ufunc' (line 23)
        _disables_array_ufunc_1880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 11), '_disables_array_ufunc', False)
        # Calling _disables_array_ufunc(args, kwargs) (line 23)
        _disables_array_ufunc_call_result_1883 = invoke(stypy.reporting.localization.Localization(__file__, 23, 11), _disables_array_ufunc_1880, *[other_1881], **kwargs_1882)
        
        # Testing the type of an if condition (line 23)
        if_condition_1884 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 23, 8), _disables_array_ufunc_call_result_1883)
        # Assigning a type to the variable 'if_condition_1884' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'if_condition_1884', if_condition_1884)
        # SSA begins for if statement (line 23)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 24)
        NotImplemented_1885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'stypy_return_type', NotImplemented_1885)
        # SSA join for if statement (line 23)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to ufunc(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'self' (line 25)
        self_1887 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 21), 'self', False)
        # Getting the type of 'other' (line 25)
        other_1888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'other', False)
        # Processing the call keyword arguments (line 25)
        kwargs_1889 = {}
        # Getting the type of 'ufunc' (line 25)
        ufunc_1886 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'ufunc', False)
        # Calling ufunc(args, kwargs) (line 25)
        ufunc_call_result_1890 = invoke(stypy.reporting.localization.Localization(__file__, 25, 15), ufunc_1886, *[self_1887, other_1888], **kwargs_1889)
        
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', ufunc_call_result_1890)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_1891 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1891)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_1891

    # Assigning a type to the variable 'func' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'func', func)
    
    # Assigning a Call to a Attribute (line 26):
    
    # Assigning a Call to a Attribute (line 26):
    
    # Call to format(...): (line 26)
    # Processing the call arguments (line 26)
    # Getting the type of 'name' (line 26)
    name_1894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 36), 'name', False)
    # Processing the call keyword arguments (line 26)
    kwargs_1895 = {}
    str_1892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 20), 'str', '__{}__')
    # Obtaining the member 'format' of a type (line 26)
    format_1893 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 20), str_1892, 'format')
    # Calling format(args, kwargs) (line 26)
    format_call_result_1896 = invoke(stypy.reporting.localization.Localization(__file__, 26, 20), format_1893, *[name_1894], **kwargs_1895)
    
    # Getting the type of 'func' (line 26)
    func_1897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'func')
    # Setting the type of the member '__name__' of a type (line 26)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 4), func_1897, '__name__', format_call_result_1896)
    # Getting the type of 'func' (line 27)
    func_1898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type', func_1898)
    
    # ################# End of '_binary_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_binary_method' in the type store
    # Getting the type of 'stypy_return_type' (line 20)
    stypy_return_type_1899 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1899)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_binary_method'
    return stypy_return_type_1899

# Assigning a type to the variable '_binary_method' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), '_binary_method', _binary_method)

@norecursion
def _reflected_binary_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_reflected_binary_method'
    module_type_store = module_type_store.open_function_context('_reflected_binary_method', 30, 0, False)
    
    # Passed parameters checking function
    _reflected_binary_method.stypy_localization = localization
    _reflected_binary_method.stypy_type_of_self = None
    _reflected_binary_method.stypy_type_store = module_type_store
    _reflected_binary_method.stypy_function_name = '_reflected_binary_method'
    _reflected_binary_method.stypy_param_names_list = ['ufunc', 'name']
    _reflected_binary_method.stypy_varargs_param_name = None
    _reflected_binary_method.stypy_kwargs_param_name = None
    _reflected_binary_method.stypy_call_defaults = defaults
    _reflected_binary_method.stypy_call_varargs = varargs
    _reflected_binary_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_reflected_binary_method', ['ufunc', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_reflected_binary_method', localization, ['ufunc', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_reflected_binary_method(...)' code ##################

    str_1900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 4), 'str', 'Implement a reflected binary method with a ufunc, e.g., __radd__.')

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 32, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['self', 'other']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['self', 'other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['self', 'other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        
        # Call to _disables_array_ufunc(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'other' (line 33)
        other_1902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 33), 'other', False)
        # Processing the call keyword arguments (line 33)
        kwargs_1903 = {}
        # Getting the type of '_disables_array_ufunc' (line 33)
        _disables_array_ufunc_1901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 11), '_disables_array_ufunc', False)
        # Calling _disables_array_ufunc(args, kwargs) (line 33)
        _disables_array_ufunc_call_result_1904 = invoke(stypy.reporting.localization.Localization(__file__, 33, 11), _disables_array_ufunc_1901, *[other_1902], **kwargs_1903)
        
        # Testing the type of an if condition (line 33)
        if_condition_1905 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 33, 8), _disables_array_ufunc_call_result_1904)
        # Assigning a type to the variable 'if_condition_1905' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'if_condition_1905', if_condition_1905)
        # SSA begins for if statement (line 33)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'NotImplemented' (line 34)
        NotImplemented_1906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 19), 'NotImplemented')
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 12), 'stypy_return_type', NotImplemented_1906)
        # SSA join for if statement (line 33)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to ufunc(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'other' (line 35)
        other_1908 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'other', False)
        # Getting the type of 'self' (line 35)
        self_1909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 28), 'self', False)
        # Processing the call keyword arguments (line 35)
        kwargs_1910 = {}
        # Getting the type of 'ufunc' (line 35)
        ufunc_1907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'ufunc', False)
        # Calling ufunc(args, kwargs) (line 35)
        ufunc_call_result_1911 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), ufunc_1907, *[other_1908, self_1909], **kwargs_1910)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', ufunc_call_result_1911)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_1912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1912)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_1912

    # Assigning a type to the variable 'func' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'func', func)
    
    # Assigning a Call to a Attribute (line 36):
    
    # Assigning a Call to a Attribute (line 36):
    
    # Call to format(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'name' (line 36)
    name_1915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 37), 'name', False)
    # Processing the call keyword arguments (line 36)
    kwargs_1916 = {}
    str_1913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 20), 'str', '__r{}__')
    # Obtaining the member 'format' of a type (line 36)
    format_1914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 20), str_1913, 'format')
    # Calling format(args, kwargs) (line 36)
    format_call_result_1917 = invoke(stypy.reporting.localization.Localization(__file__, 36, 20), format_1914, *[name_1915], **kwargs_1916)
    
    # Getting the type of 'func' (line 36)
    func_1918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'func')
    # Setting the type of the member '__name__' of a type (line 36)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 4), func_1918, '__name__', format_call_result_1917)
    # Getting the type of 'func' (line 37)
    func_1919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', func_1919)
    
    # ################# End of '_reflected_binary_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_reflected_binary_method' in the type store
    # Getting the type of 'stypy_return_type' (line 30)
    stypy_return_type_1920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1920)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_reflected_binary_method'
    return stypy_return_type_1920

# Assigning a type to the variable '_reflected_binary_method' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), '_reflected_binary_method', _reflected_binary_method)

@norecursion
def _inplace_binary_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_inplace_binary_method'
    module_type_store = module_type_store.open_function_context('_inplace_binary_method', 40, 0, False)
    
    # Passed parameters checking function
    _inplace_binary_method.stypy_localization = localization
    _inplace_binary_method.stypy_type_of_self = None
    _inplace_binary_method.stypy_type_store = module_type_store
    _inplace_binary_method.stypy_function_name = '_inplace_binary_method'
    _inplace_binary_method.stypy_param_names_list = ['ufunc', 'name']
    _inplace_binary_method.stypy_varargs_param_name = None
    _inplace_binary_method.stypy_kwargs_param_name = None
    _inplace_binary_method.stypy_call_defaults = defaults
    _inplace_binary_method.stypy_call_varargs = varargs
    _inplace_binary_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_inplace_binary_method', ['ufunc', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_inplace_binary_method', localization, ['ufunc', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_inplace_binary_method(...)' code ##################

    str_1921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 4), 'str', 'Implement an in-place binary method with a ufunc, e.g., __iadd__.')

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 42, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['self', 'other']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['self', 'other'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['self', 'other'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Call to ufunc(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'self' (line 43)
        self_1923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'self', False)
        # Getting the type of 'other' (line 43)
        other_1924 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 27), 'other', False)
        # Processing the call keyword arguments (line 43)
        
        # Obtaining an instance of the builtin type 'tuple' (line 43)
        tuple_1925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 43)
        # Adding element type (line 43)
        # Getting the type of 'self' (line 43)
        self_1926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 39), 'self', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 39), tuple_1925, self_1926)
        
        keyword_1927 = tuple_1925
        kwargs_1928 = {'out': keyword_1927}
        # Getting the type of 'ufunc' (line 43)
        ufunc_1922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'ufunc', False)
        # Calling ufunc(args, kwargs) (line 43)
        ufunc_call_result_1929 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), ufunc_1922, *[self_1923, other_1924], **kwargs_1928)
        
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', ufunc_call_result_1929)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_1930 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1930)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_1930

    # Assigning a type to the variable 'func' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'func', func)
    
    # Assigning a Call to a Attribute (line 44):
    
    # Assigning a Call to a Attribute (line 44):
    
    # Call to format(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'name' (line 44)
    name_1933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 37), 'name', False)
    # Processing the call keyword arguments (line 44)
    kwargs_1934 = {}
    str_1931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 20), 'str', '__i{}__')
    # Obtaining the member 'format' of a type (line 44)
    format_1932 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 20), str_1931, 'format')
    # Calling format(args, kwargs) (line 44)
    format_call_result_1935 = invoke(stypy.reporting.localization.Localization(__file__, 44, 20), format_1932, *[name_1933], **kwargs_1934)
    
    # Getting the type of 'func' (line 44)
    func_1936 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'func')
    # Setting the type of the member '__name__' of a type (line 44)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), func_1936, '__name__', format_call_result_1935)
    # Getting the type of 'func' (line 45)
    func_1937 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type', func_1937)
    
    # ################# End of '_inplace_binary_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_inplace_binary_method' in the type store
    # Getting the type of 'stypy_return_type' (line 40)
    stypy_return_type_1938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1938)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_inplace_binary_method'
    return stypy_return_type_1938

# Assigning a type to the variable '_inplace_binary_method' (line 40)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 0), '_inplace_binary_method', _inplace_binary_method)

@norecursion
def _numeric_methods(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_numeric_methods'
    module_type_store = module_type_store.open_function_context('_numeric_methods', 48, 0, False)
    
    # Passed parameters checking function
    _numeric_methods.stypy_localization = localization
    _numeric_methods.stypy_type_of_self = None
    _numeric_methods.stypy_type_store = module_type_store
    _numeric_methods.stypy_function_name = '_numeric_methods'
    _numeric_methods.stypy_param_names_list = ['ufunc', 'name']
    _numeric_methods.stypy_varargs_param_name = None
    _numeric_methods.stypy_kwargs_param_name = None
    _numeric_methods.stypy_call_defaults = defaults
    _numeric_methods.stypy_call_varargs = varargs
    _numeric_methods.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_numeric_methods', ['ufunc', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_numeric_methods', localization, ['ufunc', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_numeric_methods(...)' code ##################

    str_1939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'str', 'Implement forward, reflected and inplace binary methods with a ufunc.')
    
    # Obtaining an instance of the builtin type 'tuple' (line 50)
    tuple_1940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 50)
    # Adding element type (line 50)
    
    # Call to _binary_method(...): (line 50)
    # Processing the call arguments (line 50)
    # Getting the type of 'ufunc' (line 50)
    ufunc_1942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 27), 'ufunc', False)
    # Getting the type of 'name' (line 50)
    name_1943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 34), 'name', False)
    # Processing the call keyword arguments (line 50)
    kwargs_1944 = {}
    # Getting the type of '_binary_method' (line 50)
    _binary_method_1941 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), '_binary_method', False)
    # Calling _binary_method(args, kwargs) (line 50)
    _binary_method_call_result_1945 = invoke(stypy.reporting.localization.Localization(__file__, 50, 12), _binary_method_1941, *[ufunc_1942, name_1943], **kwargs_1944)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), tuple_1940, _binary_method_call_result_1945)
    # Adding element type (line 50)
    
    # Call to _reflected_binary_method(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'ufunc' (line 51)
    ufunc_1947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 37), 'ufunc', False)
    # Getting the type of 'name' (line 51)
    name_1948 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 44), 'name', False)
    # Processing the call keyword arguments (line 51)
    kwargs_1949 = {}
    # Getting the type of '_reflected_binary_method' (line 51)
    _reflected_binary_method_1946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), '_reflected_binary_method', False)
    # Calling _reflected_binary_method(args, kwargs) (line 51)
    _reflected_binary_method_call_result_1950 = invoke(stypy.reporting.localization.Localization(__file__, 51, 12), _reflected_binary_method_1946, *[ufunc_1947, name_1948], **kwargs_1949)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), tuple_1940, _reflected_binary_method_call_result_1950)
    # Adding element type (line 50)
    
    # Call to _inplace_binary_method(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'ufunc' (line 52)
    ufunc_1952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'ufunc', False)
    # Getting the type of 'name' (line 52)
    name_1953 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 42), 'name', False)
    # Processing the call keyword arguments (line 52)
    kwargs_1954 = {}
    # Getting the type of '_inplace_binary_method' (line 52)
    _inplace_binary_method_1951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), '_inplace_binary_method', False)
    # Calling _inplace_binary_method(args, kwargs) (line 52)
    _inplace_binary_method_call_result_1955 = invoke(stypy.reporting.localization.Localization(__file__, 52, 12), _inplace_binary_method_1951, *[ufunc_1952, name_1953], **kwargs_1954)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), tuple_1940, _inplace_binary_method_call_result_1955)
    
    # Assigning a type to the variable 'stypy_return_type' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'stypy_return_type', tuple_1940)
    
    # ################# End of '_numeric_methods(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_numeric_methods' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_1956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1956)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_numeric_methods'
    return stypy_return_type_1956

# Assigning a type to the variable '_numeric_methods' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), '_numeric_methods', _numeric_methods)

@norecursion
def _unary_method(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_unary_method'
    module_type_store = module_type_store.open_function_context('_unary_method', 55, 0, False)
    
    # Passed parameters checking function
    _unary_method.stypy_localization = localization
    _unary_method.stypy_type_of_self = None
    _unary_method.stypy_type_store = module_type_store
    _unary_method.stypy_function_name = '_unary_method'
    _unary_method.stypy_param_names_list = ['ufunc', 'name']
    _unary_method.stypy_varargs_param_name = None
    _unary_method.stypy_kwargs_param_name = None
    _unary_method.stypy_call_defaults = defaults
    _unary_method.stypy_call_varargs = varargs
    _unary_method.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_unary_method', ['ufunc', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_unary_method', localization, ['ufunc', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_unary_method(...)' code ##################

    str_1957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 4), 'str', 'Implement a unary special method with a ufunc.')

    @norecursion
    def func(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'func'
        module_type_store = module_type_store.open_function_context('func', 57, 4, False)
        
        # Passed parameters checking function
        func.stypy_localization = localization
        func.stypy_type_of_self = None
        func.stypy_type_store = module_type_store
        func.stypy_function_name = 'func'
        func.stypy_param_names_list = ['self']
        func.stypy_varargs_param_name = None
        func.stypy_kwargs_param_name = None
        func.stypy_call_defaults = defaults
        func.stypy_call_varargs = varargs
        func.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, 'func', ['self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'func', localization, ['self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'func(...)' code ##################

        
        # Call to ufunc(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'self' (line 58)
        self_1959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 21), 'self', False)
        # Processing the call keyword arguments (line 58)
        kwargs_1960 = {}
        # Getting the type of 'ufunc' (line 58)
        ufunc_1958 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 15), 'ufunc', False)
        # Calling ufunc(args, kwargs) (line 58)
        ufunc_call_result_1961 = invoke(stypy.reporting.localization.Localization(__file__, 58, 15), ufunc_1958, *[self_1959], **kwargs_1960)
        
        # Assigning a type to the variable 'stypy_return_type' (line 58)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'stypy_return_type', ufunc_call_result_1961)
        
        # ################# End of 'func(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'func' in the type store
        # Getting the type of 'stypy_return_type' (line 57)
        stypy_return_type_1962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_1962)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'func'
        return stypy_return_type_1962

    # Assigning a type to the variable 'func' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'func', func)
    
    # Assigning a Call to a Attribute (line 59):
    
    # Assigning a Call to a Attribute (line 59):
    
    # Call to format(...): (line 59)
    # Processing the call arguments (line 59)
    # Getting the type of 'name' (line 59)
    name_1965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 36), 'name', False)
    # Processing the call keyword arguments (line 59)
    kwargs_1966 = {}
    str_1963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 20), 'str', '__{}__')
    # Obtaining the member 'format' of a type (line 59)
    format_1964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 20), str_1963, 'format')
    # Calling format(args, kwargs) (line 59)
    format_call_result_1967 = invoke(stypy.reporting.localization.Localization(__file__, 59, 20), format_1964, *[name_1965], **kwargs_1966)
    
    # Getting the type of 'func' (line 59)
    func_1968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'func')
    # Setting the type of the member '__name__' of a type (line 59)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 4), func_1968, '__name__', format_call_result_1967)
    # Getting the type of 'func' (line 60)
    func_1969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'func')
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', func_1969)
    
    # ################# End of '_unary_method(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_unary_method' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_1970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_1970)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_unary_method'
    return stypy_return_type_1970

# Assigning a type to the variable '_unary_method' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), '_unary_method', _unary_method)
# Declaration of the 'NDArrayOperatorsMixin' class

class NDArrayOperatorsMixin(object, ):
    str_1971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, (-1)), 'str', "Mixin defining all operator special methods using __array_ufunc__.\n\n    This class implements the special methods for almost all of Python's\n    builtin operators defined in the `operator` module, including comparisons\n    (``==``, ``>``, etc.) and arithmetic (``+``, ``*``, ``-``, etc.), by\n    deferring to the ``__array_ufunc__`` method, which subclasses must\n    implement.\n\n    This class does not yet implement the special operators corresponding\n    to ``matmul`` (``@``), because ``np.matmul`` is not yet a NumPy ufunc.\n\n    It is useful for writing classes that do not inherit from `numpy.ndarray`,\n    but that should support arithmetic and numpy universal functions like\n    arrays as described in :ref:`A Mechanism for Overriding Ufuncs\n    <neps.ufunc-overrides>`.\n\n    As an trivial example, consider this implementation of an ``ArrayLike``\n    class that simply wraps a NumPy array and ensures that the result of any\n    arithmetic operation is also an ``ArrayLike`` object::\n\n        class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):\n            def __init__(self, value):\n                self.value = np.asarray(value)\n\n            # One might also consider adding the built-in list type to this\n            # list, to support operations like np.add(array_like, list)\n            _HANDLED_TYPES = (np.ndarray, numbers.Number)\n\n            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):\n                out = kwargs.get('out', ())\n                for x in inputs + out:\n                    # Only support operations with instances of _HANDLED_TYPES.\n                    # Use ArrayLike instead of type(self) for isinstance to\n                    # allow subclasses that don't override __array_ufunc__ to\n                    # handle ArrayLike objects.\n                    if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):\n                        return NotImplemented\n\n                # Defer to the implementation of the ufunc on unwrapped values.\n                inputs = tuple(x.value if isinstance(x, ArrayLike) else x\n                               for x in inputs)\n                if out:\n                    kwargs['out'] = tuple(\n                        x.value if isinstance(x, ArrayLike) else x\n                        for x in out)\n                result = getattr(ufunc, method)(*inputs, **kwargs)\n\n                if type(result) is tuple:\n                    # multiple return values\n                    return tuple(type(self)(x) for x in result)\n                elif method == 'at':\n                    # no return value\n                    return None\n                else:\n                    # one return value\n                    return type(self)(result)\n\n            def __repr__(self):\n                return '%s(%r)' % (type(self).__name__, self.value)\n\n    In interactions between ``ArrayLike`` objects and numbers or numpy arrays,\n    the result is always another ``ArrayLike``:\n\n        >>> x = ArrayLike([1, 2, 3])\n        >>> x - 1\n        ArrayLike(array([0, 1, 2]))\n        >>> 1 - x\n        ArrayLike(array([ 0, -1, -2]))\n        >>> np.arange(3) - x\n        ArrayLike(array([-1, -1, -1]))\n        >>> x - np.arange(3)\n        ArrayLike(array([1, 1, 1]))\n\n    Note that unlike ``numpy.ndarray``, ``ArrayLike`` does not allow operations\n    with arbitrary, unrecognized types. This ensures that interactions with\n    ArrayLike preserve a well-defined casting hierarchy.\n    ")
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 146):
    
    # Assigning a Call to a Name (line 147):
    
    # Assigning a Call to a Name (line 148):
    
    # Assigning a Call to a Name (line 149):
    
    # Assigning a Call to a Name (line 150):
    
    # Assigning a Call to a Tuple (line 153):
    
    # Assigning a Call to a Tuple (line 154):
    
    # Assigning a Call to a Tuple (line 159):
    
    # Assigning a Call to a Tuple (line 161):
    
    # Assigning a Call to a Tuple (line 163):
    
    # Assigning a Call to a Name (line 164):
    
    # Assigning a Call to a Name (line 165):
    
    # Assigning a Call to a Tuple (line 168):
    
    # Assigning a Call to a Tuple (line 169):
    
    # Assigning a Call to a Tuple (line 171):
    
    # Assigning a Call to a Tuple (line 173):
    
    # Assigning a Call to a Tuple (line 174):
    
    # Assigning a Call to a Tuple (line 175):
    
    # Assigning a Call to a Name (line 178):
    
    # Assigning a Call to a Name (line 179):
    
    # Assigning a Call to a Name (line 180):
    
    # Assigning a Call to a Name (line 181):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 63, 0, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NDArrayOperatorsMixin.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NDArrayOperatorsMixin' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'NDArrayOperatorsMixin', NDArrayOperatorsMixin)

# Assigning a Call to a Name (line 145):

# Call to _binary_method(...): (line 145)
# Processing the call arguments (line 145)
# Getting the type of 'um' (line 145)
um_1973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'um', False)
# Obtaining the member 'less' of a type (line 145)
less_1974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 28), um_1973, 'less')
str_1975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 37), 'str', 'lt')
# Processing the call keyword arguments (line 145)
kwargs_1976 = {}
# Getting the type of '_binary_method' (line 145)
_binary_method_1972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 13), '_binary_method', False)
# Calling _binary_method(args, kwargs) (line 145)
_binary_method_call_result_1977 = invoke(stypy.reporting.localization.Localization(__file__, 145, 13), _binary_method_1972, *[less_1974, str_1975], **kwargs_1976)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_1978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__lt__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_1978, '__lt__', _binary_method_call_result_1977)

# Assigning a Call to a Name (line 146):

# Call to _binary_method(...): (line 146)
# Processing the call arguments (line 146)
# Getting the type of 'um' (line 146)
um_1980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 28), 'um', False)
# Obtaining the member 'less_equal' of a type (line 146)
less_equal_1981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 28), um_1980, 'less_equal')
str_1982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 43), 'str', 'le')
# Processing the call keyword arguments (line 146)
kwargs_1983 = {}
# Getting the type of '_binary_method' (line 146)
_binary_method_1979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 13), '_binary_method', False)
# Calling _binary_method(args, kwargs) (line 146)
_binary_method_call_result_1984 = invoke(stypy.reporting.localization.Localization(__file__, 146, 13), _binary_method_1979, *[less_equal_1981, str_1982], **kwargs_1983)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_1985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__le__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_1985, '__le__', _binary_method_call_result_1984)

# Assigning a Call to a Name (line 147):

# Call to _binary_method(...): (line 147)
# Processing the call arguments (line 147)
# Getting the type of 'um' (line 147)
um_1987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 28), 'um', False)
# Obtaining the member 'equal' of a type (line 147)
equal_1988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 28), um_1987, 'equal')
str_1989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 38), 'str', 'eq')
# Processing the call keyword arguments (line 147)
kwargs_1990 = {}
# Getting the type of '_binary_method' (line 147)
_binary_method_1986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 13), '_binary_method', False)
# Calling _binary_method(args, kwargs) (line 147)
_binary_method_call_result_1991 = invoke(stypy.reporting.localization.Localization(__file__, 147, 13), _binary_method_1986, *[equal_1988, str_1989], **kwargs_1990)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_1992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__eq__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_1992, '__eq__', _binary_method_call_result_1991)

# Assigning a Call to a Name (line 148):

# Call to _binary_method(...): (line 148)
# Processing the call arguments (line 148)
# Getting the type of 'um' (line 148)
um_1994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 28), 'um', False)
# Obtaining the member 'not_equal' of a type (line 148)
not_equal_1995 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 28), um_1994, 'not_equal')
str_1996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 42), 'str', 'ne')
# Processing the call keyword arguments (line 148)
kwargs_1997 = {}
# Getting the type of '_binary_method' (line 148)
_binary_method_1993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 13), '_binary_method', False)
# Calling _binary_method(args, kwargs) (line 148)
_binary_method_call_result_1998 = invoke(stypy.reporting.localization.Localization(__file__, 148, 13), _binary_method_1993, *[not_equal_1995, str_1996], **kwargs_1997)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_1999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__ne__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_1999, '__ne__', _binary_method_call_result_1998)

# Assigning a Call to a Name (line 149):

# Call to _binary_method(...): (line 149)
# Processing the call arguments (line 149)
# Getting the type of 'um' (line 149)
um_2001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 28), 'um', False)
# Obtaining the member 'greater' of a type (line 149)
greater_2002 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 28), um_2001, 'greater')
str_2003 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 40), 'str', 'gt')
# Processing the call keyword arguments (line 149)
kwargs_2004 = {}
# Getting the type of '_binary_method' (line 149)
_binary_method_2000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 13), '_binary_method', False)
# Calling _binary_method(args, kwargs) (line 149)
_binary_method_call_result_2005 = invoke(stypy.reporting.localization.Localization(__file__, 149, 13), _binary_method_2000, *[greater_2002, str_2003], **kwargs_2004)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__gt__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2006, '__gt__', _binary_method_call_result_2005)

# Assigning a Call to a Name (line 150):

# Call to _binary_method(...): (line 150)
# Processing the call arguments (line 150)
# Getting the type of 'um' (line 150)
um_2008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 28), 'um', False)
# Obtaining the member 'greater_equal' of a type (line 150)
greater_equal_2009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 28), um_2008, 'greater_equal')
str_2010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 46), 'str', 'ge')
# Processing the call keyword arguments (line 150)
kwargs_2011 = {}
# Getting the type of '_binary_method' (line 150)
_binary_method_2007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), '_binary_method', False)
# Calling _binary_method(args, kwargs) (line 150)
_binary_method_call_result_2012 = invoke(stypy.reporting.localization.Localization(__file__, 150, 13), _binary_method_2007, *[greater_equal_2009, str_2010], **kwargs_2011)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__ge__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2013, '__ge__', _binary_method_call_result_2012)

# Assigning a Subscript to a Name (line 153):

# Obtaining the type of the subscript
int_2014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')

# Call to _numeric_methods(...): (line 153)
# Processing the call arguments (line 153)
# Getting the type of 'um' (line 153)
um_2016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 51), 'um', False)
# Obtaining the member 'add' of a type (line 153)
add_2017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 51), um_2016, 'add')
str_2018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 59), 'str', 'add')
# Processing the call keyword arguments (line 153)
kwargs_2019 = {}
# Getting the type of '_numeric_methods' (line 153)
_numeric_methods_2015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 153)
_numeric_methods_call_result_2020 = invoke(stypy.reporting.localization.Localization(__file__, 153, 34), _numeric_methods_2015, *[add_2017, str_2018], **kwargs_2019)

# Obtaining the member '__getitem__' of a type (line 153)
getitem___2021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), _numeric_methods_call_result_2020, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 153)
subscript_call_result_2022 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___2021, int_2014)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1829' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2023, 'tuple_var_assignment_1829', subscript_call_result_2022)

# Assigning a Subscript to a Name (line 153):

# Obtaining the type of the subscript
int_2024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')

# Call to _numeric_methods(...): (line 153)
# Processing the call arguments (line 153)
# Getting the type of 'um' (line 153)
um_2026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 51), 'um', False)
# Obtaining the member 'add' of a type (line 153)
add_2027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 51), um_2026, 'add')
str_2028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 59), 'str', 'add')
# Processing the call keyword arguments (line 153)
kwargs_2029 = {}
# Getting the type of '_numeric_methods' (line 153)
_numeric_methods_2025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 153)
_numeric_methods_call_result_2030 = invoke(stypy.reporting.localization.Localization(__file__, 153, 34), _numeric_methods_2025, *[add_2027, str_2028], **kwargs_2029)

# Obtaining the member '__getitem__' of a type (line 153)
getitem___2031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), _numeric_methods_call_result_2030, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 153)
subscript_call_result_2032 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___2031, int_2024)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1830' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2033, 'tuple_var_assignment_1830', subscript_call_result_2032)

# Assigning a Subscript to a Name (line 153):

# Obtaining the type of the subscript
int_2034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 4), 'int')

# Call to _numeric_methods(...): (line 153)
# Processing the call arguments (line 153)
# Getting the type of 'um' (line 153)
um_2036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 51), 'um', False)
# Obtaining the member 'add' of a type (line 153)
add_2037 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 51), um_2036, 'add')
str_2038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 59), 'str', 'add')
# Processing the call keyword arguments (line 153)
kwargs_2039 = {}
# Getting the type of '_numeric_methods' (line 153)
_numeric_methods_2035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 153)
_numeric_methods_call_result_2040 = invoke(stypy.reporting.localization.Localization(__file__, 153, 34), _numeric_methods_2035, *[add_2037, str_2038], **kwargs_2039)

# Obtaining the member '__getitem__' of a type (line 153)
getitem___2041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 153, 4), _numeric_methods_call_result_2040, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 153)
subscript_call_result_2042 = invoke(stypy.reporting.localization.Localization(__file__, 153, 4), getitem___2041, int_2034)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1831' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2043, 'tuple_var_assignment_1831', subscript_call_result_2042)

# Assigning a Name to a Name (line 153):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1829' of a type
tuple_var_assignment_1829_2045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2044, 'tuple_var_assignment_1829')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__add__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2046, '__add__', tuple_var_assignment_1829_2045)

# Assigning a Name to a Name (line 153):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1830' of a type
tuple_var_assignment_1830_2048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2047, 'tuple_var_assignment_1830')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__radd__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2049, '__radd__', tuple_var_assignment_1830_2048)

# Assigning a Name to a Name (line 153):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1831' of a type
tuple_var_assignment_1831_2051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2050, 'tuple_var_assignment_1831')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__iadd__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2052, '__iadd__', tuple_var_assignment_1831_2051)

# Assigning a Subscript to a Name (line 154):

# Obtaining the type of the subscript
int_2053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'int')

# Call to _numeric_methods(...): (line 154)
# Processing the call arguments (line 154)
# Getting the type of 'um' (line 154)
um_2055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 51), 'um', False)
# Obtaining the member 'subtract' of a type (line 154)
subtract_2056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 51), um_2055, 'subtract')
str_2057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 64), 'str', 'sub')
# Processing the call keyword arguments (line 154)
kwargs_2058 = {}
# Getting the type of '_numeric_methods' (line 154)
_numeric_methods_2054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 154)
_numeric_methods_call_result_2059 = invoke(stypy.reporting.localization.Localization(__file__, 154, 34), _numeric_methods_2054, *[subtract_2056, str_2057], **kwargs_2058)

# Obtaining the member '__getitem__' of a type (line 154)
getitem___2060 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), _numeric_methods_call_result_2059, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 154)
subscript_call_result_2061 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), getitem___2060, int_2053)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1832' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2062, 'tuple_var_assignment_1832', subscript_call_result_2061)

# Assigning a Subscript to a Name (line 154):

# Obtaining the type of the subscript
int_2063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'int')

# Call to _numeric_methods(...): (line 154)
# Processing the call arguments (line 154)
# Getting the type of 'um' (line 154)
um_2065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 51), 'um', False)
# Obtaining the member 'subtract' of a type (line 154)
subtract_2066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 51), um_2065, 'subtract')
str_2067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 64), 'str', 'sub')
# Processing the call keyword arguments (line 154)
kwargs_2068 = {}
# Getting the type of '_numeric_methods' (line 154)
_numeric_methods_2064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 154)
_numeric_methods_call_result_2069 = invoke(stypy.reporting.localization.Localization(__file__, 154, 34), _numeric_methods_2064, *[subtract_2066, str_2067], **kwargs_2068)

# Obtaining the member '__getitem__' of a type (line 154)
getitem___2070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), _numeric_methods_call_result_2069, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 154)
subscript_call_result_2071 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), getitem___2070, int_2063)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1833' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2072, 'tuple_var_assignment_1833', subscript_call_result_2071)

# Assigning a Subscript to a Name (line 154):

# Obtaining the type of the subscript
int_2073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 4), 'int')

# Call to _numeric_methods(...): (line 154)
# Processing the call arguments (line 154)
# Getting the type of 'um' (line 154)
um_2075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 51), 'um', False)
# Obtaining the member 'subtract' of a type (line 154)
subtract_2076 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 51), um_2075, 'subtract')
str_2077 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 64), 'str', 'sub')
# Processing the call keyword arguments (line 154)
kwargs_2078 = {}
# Getting the type of '_numeric_methods' (line 154)
_numeric_methods_2074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 154)
_numeric_methods_call_result_2079 = invoke(stypy.reporting.localization.Localization(__file__, 154, 34), _numeric_methods_2074, *[subtract_2076, str_2077], **kwargs_2078)

# Obtaining the member '__getitem__' of a type (line 154)
getitem___2080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), _numeric_methods_call_result_2079, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 154)
subscript_call_result_2081 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), getitem___2080, int_2073)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1834' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2082, 'tuple_var_assignment_1834', subscript_call_result_2081)

# Assigning a Name to a Name (line 154):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1832' of a type
tuple_var_assignment_1832_2084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2083, 'tuple_var_assignment_1832')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__sub__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2085, '__sub__', tuple_var_assignment_1832_2084)

# Assigning a Name to a Name (line 154):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1833' of a type
tuple_var_assignment_1833_2087 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2086, 'tuple_var_assignment_1833')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rsub__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2088, '__rsub__', tuple_var_assignment_1833_2087)

# Assigning a Name to a Name (line 154):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1834' of a type
tuple_var_assignment_1834_2090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2089, 'tuple_var_assignment_1834')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__isub__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2091, '__isub__', tuple_var_assignment_1834_2090)

# Assigning a Subscript to a Name (line 155):

# Obtaining the type of the subscript
int_2092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')

# Call to _numeric_methods(...): (line 155)
# Processing the call arguments (line 155)
# Getting the type of 'um' (line 155)
um_2094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 51), 'um', False)
# Obtaining the member 'multiply' of a type (line 155)
multiply_2095 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 51), um_2094, 'multiply')
str_2096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 64), 'str', 'mul')
# Processing the call keyword arguments (line 155)
kwargs_2097 = {}
# Getting the type of '_numeric_methods' (line 155)
_numeric_methods_2093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 155)
_numeric_methods_call_result_2098 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), _numeric_methods_2093, *[multiply_2095, str_2096], **kwargs_2097)

# Obtaining the member '__getitem__' of a type (line 155)
getitem___2099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), _numeric_methods_call_result_2098, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 155)
subscript_call_result_2100 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___2099, int_2092)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1835' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2101, 'tuple_var_assignment_1835', subscript_call_result_2100)

# Assigning a Subscript to a Name (line 155):

# Obtaining the type of the subscript
int_2102 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')

# Call to _numeric_methods(...): (line 155)
# Processing the call arguments (line 155)
# Getting the type of 'um' (line 155)
um_2104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 51), 'um', False)
# Obtaining the member 'multiply' of a type (line 155)
multiply_2105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 51), um_2104, 'multiply')
str_2106 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 64), 'str', 'mul')
# Processing the call keyword arguments (line 155)
kwargs_2107 = {}
# Getting the type of '_numeric_methods' (line 155)
_numeric_methods_2103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 155)
_numeric_methods_call_result_2108 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), _numeric_methods_2103, *[multiply_2105, str_2106], **kwargs_2107)

# Obtaining the member '__getitem__' of a type (line 155)
getitem___2109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), _numeric_methods_call_result_2108, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 155)
subscript_call_result_2110 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___2109, int_2102)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1836' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2111, 'tuple_var_assignment_1836', subscript_call_result_2110)

# Assigning a Subscript to a Name (line 155):

# Obtaining the type of the subscript
int_2112 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 4), 'int')

# Call to _numeric_methods(...): (line 155)
# Processing the call arguments (line 155)
# Getting the type of 'um' (line 155)
um_2114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 51), 'um', False)
# Obtaining the member 'multiply' of a type (line 155)
multiply_2115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 51), um_2114, 'multiply')
str_2116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 64), 'str', 'mul')
# Processing the call keyword arguments (line 155)
kwargs_2117 = {}
# Getting the type of '_numeric_methods' (line 155)
_numeric_methods_2113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 155)
_numeric_methods_call_result_2118 = invoke(stypy.reporting.localization.Localization(__file__, 155, 34), _numeric_methods_2113, *[multiply_2115, str_2116], **kwargs_2117)

# Obtaining the member '__getitem__' of a type (line 155)
getitem___2119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 4), _numeric_methods_call_result_2118, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 155)
subscript_call_result_2120 = invoke(stypy.reporting.localization.Localization(__file__, 155, 4), getitem___2119, int_2112)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1837' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2121, 'tuple_var_assignment_1837', subscript_call_result_2120)

# Assigning a Name to a Name (line 155):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1835' of a type
tuple_var_assignment_1835_2123 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2122, 'tuple_var_assignment_1835')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__mul__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2124, '__mul__', tuple_var_assignment_1835_2123)

# Assigning a Name to a Name (line 155):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1836' of a type
tuple_var_assignment_1836_2126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2125, 'tuple_var_assignment_1836')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rmul__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2127, '__rmul__', tuple_var_assignment_1836_2126)

# Assigning a Name to a Name (line 155):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1837' of a type
tuple_var_assignment_1837_2129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2128, 'tuple_var_assignment_1837')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__imul__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2130, '__imul__', tuple_var_assignment_1837_2129)

# Assigning a Call to a Tuple (line 155):


# Getting the type of 'sys' (line 156)
sys_2131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 7), 'sys')
# Obtaining the member 'version_info' of a type (line 156)
version_info_2132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 7), sys_2131, 'version_info')
# Obtaining the member 'major' of a type (line 156)
major_2133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 7), version_info_2132, 'major')
int_2134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 32), 'int')
# Applying the binary operator '<' (line 156)
result_lt_2135 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 7), '<', major_2133, int_2134)

# Testing the type of an if condition (line 156)
if_condition_2136 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 4), result_lt_2135)
# Assigning a type to the variable 'if_condition_2136' (line 156)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'if_condition_2136', if_condition_2136)
# SSA begins for if statement (line 156)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Call to a Tuple (line 158):

# Assigning a Subscript to a Name (line 158):

# Obtaining the type of the subscript
int_2137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'int')

# Call to _numeric_methods(...): (line 158)
# Processing the call arguments (line 158)
# Getting the type of 'um' (line 158)
um_2139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'um', False)
# Obtaining the member 'divide' of a type (line 158)
divide_2140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 55), um_2139, 'divide')
str_2141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 66), 'str', 'div')
# Processing the call keyword arguments (line 158)
kwargs_2142 = {}
# Getting the type of '_numeric_methods' (line 158)
_numeric_methods_2138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 158)
_numeric_methods_call_result_2143 = invoke(stypy.reporting.localization.Localization(__file__, 158, 38), _numeric_methods_2138, *[divide_2140, str_2141], **kwargs_2142)

# Obtaining the member '__getitem__' of a type (line 158)
getitem___2144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), _numeric_methods_call_result_2143, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 158)
subscript_call_result_2145 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___2144, int_2137)

# Assigning a type to the variable 'tuple_var_assignment_1838' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_1838', subscript_call_result_2145)

# Assigning a Subscript to a Name (line 158):

# Obtaining the type of the subscript
int_2146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'int')

# Call to _numeric_methods(...): (line 158)
# Processing the call arguments (line 158)
# Getting the type of 'um' (line 158)
um_2148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'um', False)
# Obtaining the member 'divide' of a type (line 158)
divide_2149 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 55), um_2148, 'divide')
str_2150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 66), 'str', 'div')
# Processing the call keyword arguments (line 158)
kwargs_2151 = {}
# Getting the type of '_numeric_methods' (line 158)
_numeric_methods_2147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 158)
_numeric_methods_call_result_2152 = invoke(stypy.reporting.localization.Localization(__file__, 158, 38), _numeric_methods_2147, *[divide_2149, str_2150], **kwargs_2151)

# Obtaining the member '__getitem__' of a type (line 158)
getitem___2153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), _numeric_methods_call_result_2152, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 158)
subscript_call_result_2154 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___2153, int_2146)

# Assigning a type to the variable 'tuple_var_assignment_1839' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_1839', subscript_call_result_2154)

# Assigning a Subscript to a Name (line 158):

# Obtaining the type of the subscript
int_2155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 8), 'int')

# Call to _numeric_methods(...): (line 158)
# Processing the call arguments (line 158)
# Getting the type of 'um' (line 158)
um_2157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 55), 'um', False)
# Obtaining the member 'divide' of a type (line 158)
divide_2158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 55), um_2157, 'divide')
str_2159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 66), 'str', 'div')
# Processing the call keyword arguments (line 158)
kwargs_2160 = {}
# Getting the type of '_numeric_methods' (line 158)
_numeric_methods_2156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 38), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 158)
_numeric_methods_call_result_2161 = invoke(stypy.reporting.localization.Localization(__file__, 158, 38), _numeric_methods_2156, *[divide_2158, str_2159], **kwargs_2160)

# Obtaining the member '__getitem__' of a type (line 158)
getitem___2162 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), _numeric_methods_call_result_2161, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 158)
subscript_call_result_2163 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), getitem___2162, int_2155)

# Assigning a type to the variable 'tuple_var_assignment_1840' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_1840', subscript_call_result_2163)

# Assigning a Name to a Name (line 158):
# Getting the type of 'tuple_var_assignment_1838' (line 158)
tuple_var_assignment_1838_2164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_1838')
# Assigning a type to the variable '__div__' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), '__div__', tuple_var_assignment_1838_2164)

# Assigning a Name to a Name (line 158):
# Getting the type of 'tuple_var_assignment_1839' (line 158)
tuple_var_assignment_1839_2165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_1839')
# Assigning a type to the variable '__rdiv__' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 17), '__rdiv__', tuple_var_assignment_1839_2165)

# Assigning a Name to a Name (line 158):
# Getting the type of 'tuple_var_assignment_1840' (line 158)
tuple_var_assignment_1840_2166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'tuple_var_assignment_1840')
# Assigning a type to the variable '__idiv__' (line 158)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 27), '__idiv__', tuple_var_assignment_1840_2166)
# SSA join for if statement (line 156)
module_type_store = module_type_store.join_ssa_context()


# Assigning a Subscript to a Name (line 159):

# Obtaining the type of the subscript
int_2167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'int')

# Call to _numeric_methods(...): (line 159)
# Processing the call arguments (line 159)
# Getting the type of 'um' (line 160)
um_2169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'um', False)
# Obtaining the member 'true_divide' of a type (line 160)
true_divide_2170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), um_2169, 'true_divide')
str_2171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'str', 'truediv')
# Processing the call keyword arguments (line 159)
kwargs_2172 = {}
# Getting the type of '_numeric_methods' (line 159)
_numeric_methods_2168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 159)
_numeric_methods_call_result_2173 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), _numeric_methods_2168, *[true_divide_2170, str_2171], **kwargs_2172)

# Obtaining the member '__getitem__' of a type (line 159)
getitem___2174 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), _numeric_methods_call_result_2173, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 159)
subscript_call_result_2175 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), getitem___2174, int_2167)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1841' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2176, 'tuple_var_assignment_1841', subscript_call_result_2175)

# Assigning a Subscript to a Name (line 159):

# Obtaining the type of the subscript
int_2177 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'int')

# Call to _numeric_methods(...): (line 159)
# Processing the call arguments (line 159)
# Getting the type of 'um' (line 160)
um_2179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'um', False)
# Obtaining the member 'true_divide' of a type (line 160)
true_divide_2180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), um_2179, 'true_divide')
str_2181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'str', 'truediv')
# Processing the call keyword arguments (line 159)
kwargs_2182 = {}
# Getting the type of '_numeric_methods' (line 159)
_numeric_methods_2178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 159)
_numeric_methods_call_result_2183 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), _numeric_methods_2178, *[true_divide_2180, str_2181], **kwargs_2182)

# Obtaining the member '__getitem__' of a type (line 159)
getitem___2184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), _numeric_methods_call_result_2183, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 159)
subscript_call_result_2185 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), getitem___2184, int_2177)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1842' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2186, 'tuple_var_assignment_1842', subscript_call_result_2185)

# Assigning a Subscript to a Name (line 159):

# Obtaining the type of the subscript
int_2187 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, 4), 'int')

# Call to _numeric_methods(...): (line 159)
# Processing the call arguments (line 159)
# Getting the type of 'um' (line 160)
um_2189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'um', False)
# Obtaining the member 'true_divide' of a type (line 160)
true_divide_2190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), um_2189, 'true_divide')
str_2191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 24), 'str', 'truediv')
# Processing the call keyword arguments (line 159)
kwargs_2192 = {}
# Getting the type of '_numeric_methods' (line 159)
_numeric_methods_2188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 46), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 159)
_numeric_methods_call_result_2193 = invoke(stypy.reporting.localization.Localization(__file__, 159, 46), _numeric_methods_2188, *[true_divide_2190, str_2191], **kwargs_2192)

# Obtaining the member '__getitem__' of a type (line 159)
getitem___2194 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 4), _numeric_methods_call_result_2193, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 159)
subscript_call_result_2195 = invoke(stypy.reporting.localization.Localization(__file__, 159, 4), getitem___2194, int_2187)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1843' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2196, 'tuple_var_assignment_1843', subscript_call_result_2195)

# Assigning a Name to a Name (line 159):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1841' of a type
tuple_var_assignment_1841_2198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2197, 'tuple_var_assignment_1841')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__truediv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2199, '__truediv__', tuple_var_assignment_1841_2198)

# Assigning a Name to a Name (line 159):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1842' of a type
tuple_var_assignment_1842_2201 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2200, 'tuple_var_assignment_1842')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rtruediv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2202, '__rtruediv__', tuple_var_assignment_1842_2201)

# Assigning a Name to a Name (line 159):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1843' of a type
tuple_var_assignment_1843_2204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2203, 'tuple_var_assignment_1843')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__itruediv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2205, '__itruediv__', tuple_var_assignment_1843_2204)

# Assigning a Subscript to a Name (line 161):

# Obtaining the type of the subscript
int_2206 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')

# Call to _numeric_methods(...): (line 161)
# Processing the call arguments (line 161)
# Getting the type of 'um' (line 162)
um_2208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'um', False)
# Obtaining the member 'floor_divide' of a type (line 162)
floor_divide_2209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), um_2208, 'floor_divide')
str_2210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 25), 'str', 'floordiv')
# Processing the call keyword arguments (line 161)
kwargs_2211 = {}
# Getting the type of '_numeric_methods' (line 161)
_numeric_methods_2207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 49), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 161)
_numeric_methods_call_result_2212 = invoke(stypy.reporting.localization.Localization(__file__, 161, 49), _numeric_methods_2207, *[floor_divide_2209, str_2210], **kwargs_2211)

# Obtaining the member '__getitem__' of a type (line 161)
getitem___2213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), _numeric_methods_call_result_2212, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 161)
subscript_call_result_2214 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), getitem___2213, int_2206)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1844' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2215, 'tuple_var_assignment_1844', subscript_call_result_2214)

# Assigning a Subscript to a Name (line 161):

# Obtaining the type of the subscript
int_2216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')

# Call to _numeric_methods(...): (line 161)
# Processing the call arguments (line 161)
# Getting the type of 'um' (line 162)
um_2218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'um', False)
# Obtaining the member 'floor_divide' of a type (line 162)
floor_divide_2219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), um_2218, 'floor_divide')
str_2220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 25), 'str', 'floordiv')
# Processing the call keyword arguments (line 161)
kwargs_2221 = {}
# Getting the type of '_numeric_methods' (line 161)
_numeric_methods_2217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 49), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 161)
_numeric_methods_call_result_2222 = invoke(stypy.reporting.localization.Localization(__file__, 161, 49), _numeric_methods_2217, *[floor_divide_2219, str_2220], **kwargs_2221)

# Obtaining the member '__getitem__' of a type (line 161)
getitem___2223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), _numeric_methods_call_result_2222, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 161)
subscript_call_result_2224 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), getitem___2223, int_2216)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1845' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2225, 'tuple_var_assignment_1845', subscript_call_result_2224)

# Assigning a Subscript to a Name (line 161):

# Obtaining the type of the subscript
int_2226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 4), 'int')

# Call to _numeric_methods(...): (line 161)
# Processing the call arguments (line 161)
# Getting the type of 'um' (line 162)
um_2228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'um', False)
# Obtaining the member 'floor_divide' of a type (line 162)
floor_divide_2229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 8), um_2228, 'floor_divide')
str_2230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 25), 'str', 'floordiv')
# Processing the call keyword arguments (line 161)
kwargs_2231 = {}
# Getting the type of '_numeric_methods' (line 161)
_numeric_methods_2227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 49), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 161)
_numeric_methods_call_result_2232 = invoke(stypy.reporting.localization.Localization(__file__, 161, 49), _numeric_methods_2227, *[floor_divide_2229, str_2230], **kwargs_2231)

# Obtaining the member '__getitem__' of a type (line 161)
getitem___2233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 4), _numeric_methods_call_result_2232, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 161)
subscript_call_result_2234 = invoke(stypy.reporting.localization.Localization(__file__, 161, 4), getitem___2233, int_2226)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1846' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2235, 'tuple_var_assignment_1846', subscript_call_result_2234)

# Assigning a Name to a Name (line 161):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1844' of a type
tuple_var_assignment_1844_2237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2236, 'tuple_var_assignment_1844')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__floordiv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2238, '__floordiv__', tuple_var_assignment_1844_2237)

# Assigning a Name to a Name (line 161):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1845' of a type
tuple_var_assignment_1845_2240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2239, 'tuple_var_assignment_1845')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rfloordiv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2241, '__rfloordiv__', tuple_var_assignment_1845_2240)

# Assigning a Name to a Name (line 161):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1846' of a type
tuple_var_assignment_1846_2243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2242, 'tuple_var_assignment_1846')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__ifloordiv__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2244, '__ifloordiv__', tuple_var_assignment_1846_2243)

# Assigning a Subscript to a Name (line 163):

# Obtaining the type of the subscript
int_2245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 4), 'int')

# Call to _numeric_methods(...): (line 163)
# Processing the call arguments (line 163)
# Getting the type of 'um' (line 163)
um_2247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'um', False)
# Obtaining the member 'remainder' of a type (line 163)
remainder_2248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 51), um_2247, 'remainder')
str_2249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 65), 'str', 'mod')
# Processing the call keyword arguments (line 163)
kwargs_2250 = {}
# Getting the type of '_numeric_methods' (line 163)
_numeric_methods_2246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 163)
_numeric_methods_call_result_2251 = invoke(stypy.reporting.localization.Localization(__file__, 163, 34), _numeric_methods_2246, *[remainder_2248, str_2249], **kwargs_2250)

# Obtaining the member '__getitem__' of a type (line 163)
getitem___2252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 4), _numeric_methods_call_result_2251, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 163)
subscript_call_result_2253 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), getitem___2252, int_2245)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1847' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2254, 'tuple_var_assignment_1847', subscript_call_result_2253)

# Assigning a Subscript to a Name (line 163):

# Obtaining the type of the subscript
int_2255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 4), 'int')

# Call to _numeric_methods(...): (line 163)
# Processing the call arguments (line 163)
# Getting the type of 'um' (line 163)
um_2257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'um', False)
# Obtaining the member 'remainder' of a type (line 163)
remainder_2258 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 51), um_2257, 'remainder')
str_2259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 65), 'str', 'mod')
# Processing the call keyword arguments (line 163)
kwargs_2260 = {}
# Getting the type of '_numeric_methods' (line 163)
_numeric_methods_2256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 163)
_numeric_methods_call_result_2261 = invoke(stypy.reporting.localization.Localization(__file__, 163, 34), _numeric_methods_2256, *[remainder_2258, str_2259], **kwargs_2260)

# Obtaining the member '__getitem__' of a type (line 163)
getitem___2262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 4), _numeric_methods_call_result_2261, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 163)
subscript_call_result_2263 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), getitem___2262, int_2255)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1848' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2264, 'tuple_var_assignment_1848', subscript_call_result_2263)

# Assigning a Subscript to a Name (line 163):

# Obtaining the type of the subscript
int_2265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 4), 'int')

# Call to _numeric_methods(...): (line 163)
# Processing the call arguments (line 163)
# Getting the type of 'um' (line 163)
um_2267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'um', False)
# Obtaining the member 'remainder' of a type (line 163)
remainder_2268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 51), um_2267, 'remainder')
str_2269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 65), 'str', 'mod')
# Processing the call keyword arguments (line 163)
kwargs_2270 = {}
# Getting the type of '_numeric_methods' (line 163)
_numeric_methods_2266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 163)
_numeric_methods_call_result_2271 = invoke(stypy.reporting.localization.Localization(__file__, 163, 34), _numeric_methods_2266, *[remainder_2268, str_2269], **kwargs_2270)

# Obtaining the member '__getitem__' of a type (line 163)
getitem___2272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 4), _numeric_methods_call_result_2271, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 163)
subscript_call_result_2273 = invoke(stypy.reporting.localization.Localization(__file__, 163, 4), getitem___2272, int_2265)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1849' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2274, 'tuple_var_assignment_1849', subscript_call_result_2273)

# Assigning a Name to a Name (line 163):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1847' of a type
tuple_var_assignment_1847_2276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2275, 'tuple_var_assignment_1847')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__mod__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2277, '__mod__', tuple_var_assignment_1847_2276)

# Assigning a Name to a Name (line 163):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1848' of a type
tuple_var_assignment_1848_2279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2278, 'tuple_var_assignment_1848')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rmod__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2280, '__rmod__', tuple_var_assignment_1848_2279)

# Assigning a Name to a Name (line 163):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1849' of a type
tuple_var_assignment_1849_2282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2281, 'tuple_var_assignment_1849')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__imod__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2283, '__imod__', tuple_var_assignment_1849_2282)

# Assigning a Call to a Name (line 164):

# Call to _binary_method(...): (line 164)
# Processing the call arguments (line 164)
# Getting the type of 'um' (line 164)
um_2285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 32), 'um', False)
# Obtaining the member 'divmod' of a type (line 164)
divmod_2286 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 32), um_2285, 'divmod')
str_2287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 43), 'str', 'divmod')
# Processing the call keyword arguments (line 164)
kwargs_2288 = {}
# Getting the type of '_binary_method' (line 164)
_binary_method_2284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 17), '_binary_method', False)
# Calling _binary_method(args, kwargs) (line 164)
_binary_method_call_result_2289 = invoke(stypy.reporting.localization.Localization(__file__, 164, 17), _binary_method_2284, *[divmod_2286, str_2287], **kwargs_2288)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__divmod__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2290, '__divmod__', _binary_method_call_result_2289)

# Assigning a Call to a Name (line 165):

# Call to _reflected_binary_method(...): (line 165)
# Processing the call arguments (line 165)
# Getting the type of 'um' (line 165)
um_2292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 43), 'um', False)
# Obtaining the member 'divmod' of a type (line 165)
divmod_2293 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 43), um_2292, 'divmod')
str_2294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 54), 'str', 'divmod')
# Processing the call keyword arguments (line 165)
kwargs_2295 = {}
# Getting the type of '_reflected_binary_method' (line 165)
_reflected_binary_method_2291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 18), '_reflected_binary_method', False)
# Calling _reflected_binary_method(args, kwargs) (line 165)
_reflected_binary_method_call_result_2296 = invoke(stypy.reporting.localization.Localization(__file__, 165, 18), _reflected_binary_method_2291, *[divmod_2293, str_2294], **kwargs_2295)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rdivmod__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2297, '__rdivmod__', _reflected_binary_method_call_result_2296)

# Assigning a Subscript to a Name (line 168):

# Obtaining the type of the subscript
int_2298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 4), 'int')

# Call to _numeric_methods(...): (line 168)
# Processing the call arguments (line 168)
# Getting the type of 'um' (line 168)
um_2300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 51), 'um', False)
# Obtaining the member 'power' of a type (line 168)
power_2301 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 51), um_2300, 'power')
str_2302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 61), 'str', 'pow')
# Processing the call keyword arguments (line 168)
kwargs_2303 = {}
# Getting the type of '_numeric_methods' (line 168)
_numeric_methods_2299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 168)
_numeric_methods_call_result_2304 = invoke(stypy.reporting.localization.Localization(__file__, 168, 34), _numeric_methods_2299, *[power_2301, str_2302], **kwargs_2303)

# Obtaining the member '__getitem__' of a type (line 168)
getitem___2305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), _numeric_methods_call_result_2304, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 168)
subscript_call_result_2306 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), getitem___2305, int_2298)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1850' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2307, 'tuple_var_assignment_1850', subscript_call_result_2306)

# Assigning a Subscript to a Name (line 168):

# Obtaining the type of the subscript
int_2308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 4), 'int')

# Call to _numeric_methods(...): (line 168)
# Processing the call arguments (line 168)
# Getting the type of 'um' (line 168)
um_2310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 51), 'um', False)
# Obtaining the member 'power' of a type (line 168)
power_2311 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 51), um_2310, 'power')
str_2312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 61), 'str', 'pow')
# Processing the call keyword arguments (line 168)
kwargs_2313 = {}
# Getting the type of '_numeric_methods' (line 168)
_numeric_methods_2309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 168)
_numeric_methods_call_result_2314 = invoke(stypy.reporting.localization.Localization(__file__, 168, 34), _numeric_methods_2309, *[power_2311, str_2312], **kwargs_2313)

# Obtaining the member '__getitem__' of a type (line 168)
getitem___2315 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), _numeric_methods_call_result_2314, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 168)
subscript_call_result_2316 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), getitem___2315, int_2308)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1851' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2317, 'tuple_var_assignment_1851', subscript_call_result_2316)

# Assigning a Subscript to a Name (line 168):

# Obtaining the type of the subscript
int_2318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 4), 'int')

# Call to _numeric_methods(...): (line 168)
# Processing the call arguments (line 168)
# Getting the type of 'um' (line 168)
um_2320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 51), 'um', False)
# Obtaining the member 'power' of a type (line 168)
power_2321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 51), um_2320, 'power')
str_2322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 168, 61), 'str', 'pow')
# Processing the call keyword arguments (line 168)
kwargs_2323 = {}
# Getting the type of '_numeric_methods' (line 168)
_numeric_methods_2319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 168)
_numeric_methods_call_result_2324 = invoke(stypy.reporting.localization.Localization(__file__, 168, 34), _numeric_methods_2319, *[power_2321, str_2322], **kwargs_2323)

# Obtaining the member '__getitem__' of a type (line 168)
getitem___2325 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 4), _numeric_methods_call_result_2324, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 168)
subscript_call_result_2326 = invoke(stypy.reporting.localization.Localization(__file__, 168, 4), getitem___2325, int_2318)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1852' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2327, 'tuple_var_assignment_1852', subscript_call_result_2326)

# Assigning a Name to a Name (line 168):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1850' of a type
tuple_var_assignment_1850_2329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2328, 'tuple_var_assignment_1850')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__pow__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2330, '__pow__', tuple_var_assignment_1850_2329)

# Assigning a Name to a Name (line 168):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1851' of a type
tuple_var_assignment_1851_2332 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2331, 'tuple_var_assignment_1851')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rpow__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2333, '__rpow__', tuple_var_assignment_1851_2332)

# Assigning a Name to a Name (line 168):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1852' of a type
tuple_var_assignment_1852_2335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2334, 'tuple_var_assignment_1852')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__ipow__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2336, '__ipow__', tuple_var_assignment_1852_2335)

# Assigning a Subscript to a Name (line 169):

# Obtaining the type of the subscript
int_2337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 4), 'int')

# Call to _numeric_methods(...): (line 169)
# Processing the call arguments (line 169)
# Getting the type of 'um' (line 170)
um_2339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'um', False)
# Obtaining the member 'left_shift' of a type (line 170)
left_shift_2340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), um_2339, 'left_shift')
str_2341 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'str', 'lshift')
# Processing the call keyword arguments (line 169)
kwargs_2342 = {}
# Getting the type of '_numeric_methods' (line 169)
_numeric_methods_2338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 169)
_numeric_methods_call_result_2343 = invoke(stypy.reporting.localization.Localization(__file__, 169, 43), _numeric_methods_2338, *[left_shift_2340, str_2341], **kwargs_2342)

# Obtaining the member '__getitem__' of a type (line 169)
getitem___2344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 4), _numeric_methods_call_result_2343, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 169)
subscript_call_result_2345 = invoke(stypy.reporting.localization.Localization(__file__, 169, 4), getitem___2344, int_2337)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1853' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2346, 'tuple_var_assignment_1853', subscript_call_result_2345)

# Assigning a Subscript to a Name (line 169):

# Obtaining the type of the subscript
int_2347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 4), 'int')

# Call to _numeric_methods(...): (line 169)
# Processing the call arguments (line 169)
# Getting the type of 'um' (line 170)
um_2349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'um', False)
# Obtaining the member 'left_shift' of a type (line 170)
left_shift_2350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), um_2349, 'left_shift')
str_2351 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'str', 'lshift')
# Processing the call keyword arguments (line 169)
kwargs_2352 = {}
# Getting the type of '_numeric_methods' (line 169)
_numeric_methods_2348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 169)
_numeric_methods_call_result_2353 = invoke(stypy.reporting.localization.Localization(__file__, 169, 43), _numeric_methods_2348, *[left_shift_2350, str_2351], **kwargs_2352)

# Obtaining the member '__getitem__' of a type (line 169)
getitem___2354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 4), _numeric_methods_call_result_2353, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 169)
subscript_call_result_2355 = invoke(stypy.reporting.localization.Localization(__file__, 169, 4), getitem___2354, int_2347)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1854' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2356, 'tuple_var_assignment_1854', subscript_call_result_2355)

# Assigning a Subscript to a Name (line 169):

# Obtaining the type of the subscript
int_2357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 4), 'int')

# Call to _numeric_methods(...): (line 169)
# Processing the call arguments (line 169)
# Getting the type of 'um' (line 170)
um_2359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'um', False)
# Obtaining the member 'left_shift' of a type (line 170)
left_shift_2360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 8), um_2359, 'left_shift')
str_2361 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 23), 'str', 'lshift')
# Processing the call keyword arguments (line 169)
kwargs_2362 = {}
# Getting the type of '_numeric_methods' (line 169)
_numeric_methods_2358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 43), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 169)
_numeric_methods_call_result_2363 = invoke(stypy.reporting.localization.Localization(__file__, 169, 43), _numeric_methods_2358, *[left_shift_2360, str_2361], **kwargs_2362)

# Obtaining the member '__getitem__' of a type (line 169)
getitem___2364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 4), _numeric_methods_call_result_2363, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 169)
subscript_call_result_2365 = invoke(stypy.reporting.localization.Localization(__file__, 169, 4), getitem___2364, int_2357)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1855' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2366, 'tuple_var_assignment_1855', subscript_call_result_2365)

# Assigning a Name to a Name (line 169):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1853' of a type
tuple_var_assignment_1853_2368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2367, 'tuple_var_assignment_1853')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__lshift__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2369, '__lshift__', tuple_var_assignment_1853_2368)

# Assigning a Name to a Name (line 169):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1854' of a type
tuple_var_assignment_1854_2371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2370, 'tuple_var_assignment_1854')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rlshift__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2372, '__rlshift__', tuple_var_assignment_1854_2371)

# Assigning a Name to a Name (line 169):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1855' of a type
tuple_var_assignment_1855_2374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2373, 'tuple_var_assignment_1855')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__ilshift__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2375, '__ilshift__', tuple_var_assignment_1855_2374)

# Assigning a Subscript to a Name (line 171):

# Obtaining the type of the subscript
int_2376 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 4), 'int')

# Call to _numeric_methods(...): (line 171)
# Processing the call arguments (line 171)
# Getting the type of 'um' (line 172)
um_2378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'um', False)
# Obtaining the member 'right_shift' of a type (line 172)
right_shift_2379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), um_2378, 'right_shift')
str_2380 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'str', 'rshift')
# Processing the call keyword arguments (line 171)
kwargs_2381 = {}
# Getting the type of '_numeric_methods' (line 171)
_numeric_methods_2377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 43), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 171)
_numeric_methods_call_result_2382 = invoke(stypy.reporting.localization.Localization(__file__, 171, 43), _numeric_methods_2377, *[right_shift_2379, str_2380], **kwargs_2381)

# Obtaining the member '__getitem__' of a type (line 171)
getitem___2383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 4), _numeric_methods_call_result_2382, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 171)
subscript_call_result_2384 = invoke(stypy.reporting.localization.Localization(__file__, 171, 4), getitem___2383, int_2376)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1856' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2385, 'tuple_var_assignment_1856', subscript_call_result_2384)

# Assigning a Subscript to a Name (line 171):

# Obtaining the type of the subscript
int_2386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 4), 'int')

# Call to _numeric_methods(...): (line 171)
# Processing the call arguments (line 171)
# Getting the type of 'um' (line 172)
um_2388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'um', False)
# Obtaining the member 'right_shift' of a type (line 172)
right_shift_2389 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), um_2388, 'right_shift')
str_2390 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'str', 'rshift')
# Processing the call keyword arguments (line 171)
kwargs_2391 = {}
# Getting the type of '_numeric_methods' (line 171)
_numeric_methods_2387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 43), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 171)
_numeric_methods_call_result_2392 = invoke(stypy.reporting.localization.Localization(__file__, 171, 43), _numeric_methods_2387, *[right_shift_2389, str_2390], **kwargs_2391)

# Obtaining the member '__getitem__' of a type (line 171)
getitem___2393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 4), _numeric_methods_call_result_2392, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 171)
subscript_call_result_2394 = invoke(stypy.reporting.localization.Localization(__file__, 171, 4), getitem___2393, int_2386)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1857' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2395, 'tuple_var_assignment_1857', subscript_call_result_2394)

# Assigning a Subscript to a Name (line 171):

# Obtaining the type of the subscript
int_2396 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 4), 'int')

# Call to _numeric_methods(...): (line 171)
# Processing the call arguments (line 171)
# Getting the type of 'um' (line 172)
um_2398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'um', False)
# Obtaining the member 'right_shift' of a type (line 172)
right_shift_2399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 8), um_2398, 'right_shift')
str_2400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 24), 'str', 'rshift')
# Processing the call keyword arguments (line 171)
kwargs_2401 = {}
# Getting the type of '_numeric_methods' (line 171)
_numeric_methods_2397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 43), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 171)
_numeric_methods_call_result_2402 = invoke(stypy.reporting.localization.Localization(__file__, 171, 43), _numeric_methods_2397, *[right_shift_2399, str_2400], **kwargs_2401)

# Obtaining the member '__getitem__' of a type (line 171)
getitem___2403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 4), _numeric_methods_call_result_2402, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 171)
subscript_call_result_2404 = invoke(stypy.reporting.localization.Localization(__file__, 171, 4), getitem___2403, int_2396)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1858' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2405, 'tuple_var_assignment_1858', subscript_call_result_2404)

# Assigning a Name to a Name (line 171):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1856' of a type
tuple_var_assignment_1856_2407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2406, 'tuple_var_assignment_1856')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rshift__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2408, '__rshift__', tuple_var_assignment_1856_2407)

# Assigning a Name to a Name (line 171):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1857' of a type
tuple_var_assignment_1857_2410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2409, 'tuple_var_assignment_1857')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rrshift__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2411, '__rrshift__', tuple_var_assignment_1857_2410)

# Assigning a Name to a Name (line 171):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1858' of a type
tuple_var_assignment_1858_2413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2412, 'tuple_var_assignment_1858')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__irshift__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2414, '__irshift__', tuple_var_assignment_1858_2413)

# Assigning a Subscript to a Name (line 173):

# Obtaining the type of the subscript
int_2415 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'int')

# Call to _numeric_methods(...): (line 173)
# Processing the call arguments (line 173)
# Getting the type of 'um' (line 173)
um_2417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 51), 'um', False)
# Obtaining the member 'bitwise_and' of a type (line 173)
bitwise_and_2418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 51), um_2417, 'bitwise_and')
str_2419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 67), 'str', 'and')
# Processing the call keyword arguments (line 173)
kwargs_2420 = {}
# Getting the type of '_numeric_methods' (line 173)
_numeric_methods_2416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 173)
_numeric_methods_call_result_2421 = invoke(stypy.reporting.localization.Localization(__file__, 173, 34), _numeric_methods_2416, *[bitwise_and_2418, str_2419], **kwargs_2420)

# Obtaining the member '__getitem__' of a type (line 173)
getitem___2422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), _numeric_methods_call_result_2421, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 173)
subscript_call_result_2423 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), getitem___2422, int_2415)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1859' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2424, 'tuple_var_assignment_1859', subscript_call_result_2423)

# Assigning a Subscript to a Name (line 173):

# Obtaining the type of the subscript
int_2425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'int')

# Call to _numeric_methods(...): (line 173)
# Processing the call arguments (line 173)
# Getting the type of 'um' (line 173)
um_2427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 51), 'um', False)
# Obtaining the member 'bitwise_and' of a type (line 173)
bitwise_and_2428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 51), um_2427, 'bitwise_and')
str_2429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 67), 'str', 'and')
# Processing the call keyword arguments (line 173)
kwargs_2430 = {}
# Getting the type of '_numeric_methods' (line 173)
_numeric_methods_2426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 173)
_numeric_methods_call_result_2431 = invoke(stypy.reporting.localization.Localization(__file__, 173, 34), _numeric_methods_2426, *[bitwise_and_2428, str_2429], **kwargs_2430)

# Obtaining the member '__getitem__' of a type (line 173)
getitem___2432 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), _numeric_methods_call_result_2431, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 173)
subscript_call_result_2433 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), getitem___2432, int_2425)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1860' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2434, 'tuple_var_assignment_1860', subscript_call_result_2433)

# Assigning a Subscript to a Name (line 173):

# Obtaining the type of the subscript
int_2435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'int')

# Call to _numeric_methods(...): (line 173)
# Processing the call arguments (line 173)
# Getting the type of 'um' (line 173)
um_2437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 51), 'um', False)
# Obtaining the member 'bitwise_and' of a type (line 173)
bitwise_and_2438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 51), um_2437, 'bitwise_and')
str_2439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 67), 'str', 'and')
# Processing the call keyword arguments (line 173)
kwargs_2440 = {}
# Getting the type of '_numeric_methods' (line 173)
_numeric_methods_2436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 173)
_numeric_methods_call_result_2441 = invoke(stypy.reporting.localization.Localization(__file__, 173, 34), _numeric_methods_2436, *[bitwise_and_2438, str_2439], **kwargs_2440)

# Obtaining the member '__getitem__' of a type (line 173)
getitem___2442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 4), _numeric_methods_call_result_2441, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 173)
subscript_call_result_2443 = invoke(stypy.reporting.localization.Localization(__file__, 173, 4), getitem___2442, int_2435)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1861' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2444, 'tuple_var_assignment_1861', subscript_call_result_2443)

# Assigning a Name to a Name (line 173):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1859' of a type
tuple_var_assignment_1859_2446 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2445, 'tuple_var_assignment_1859')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__and__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2447, '__and__', tuple_var_assignment_1859_2446)

# Assigning a Name to a Name (line 173):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1860' of a type
tuple_var_assignment_1860_2449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2448, 'tuple_var_assignment_1860')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rand__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2450, '__rand__', tuple_var_assignment_1860_2449)

# Assigning a Name to a Name (line 173):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1861' of a type
tuple_var_assignment_1861_2452 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2451, 'tuple_var_assignment_1861')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__iand__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2453, '__iand__', tuple_var_assignment_1861_2452)

# Assigning a Subscript to a Name (line 174):

# Obtaining the type of the subscript
int_2454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 4), 'int')

# Call to _numeric_methods(...): (line 174)
# Processing the call arguments (line 174)
# Getting the type of 'um' (line 174)
um_2456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 51), 'um', False)
# Obtaining the member 'bitwise_xor' of a type (line 174)
bitwise_xor_2457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 51), um_2456, 'bitwise_xor')
str_2458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 67), 'str', 'xor')
# Processing the call keyword arguments (line 174)
kwargs_2459 = {}
# Getting the type of '_numeric_methods' (line 174)
_numeric_methods_2455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 174)
_numeric_methods_call_result_2460 = invoke(stypy.reporting.localization.Localization(__file__, 174, 34), _numeric_methods_2455, *[bitwise_xor_2457, str_2458], **kwargs_2459)

# Obtaining the member '__getitem__' of a type (line 174)
getitem___2461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), _numeric_methods_call_result_2460, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 174)
subscript_call_result_2462 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), getitem___2461, int_2454)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1862' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2463, 'tuple_var_assignment_1862', subscript_call_result_2462)

# Assigning a Subscript to a Name (line 174):

# Obtaining the type of the subscript
int_2464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 4), 'int')

# Call to _numeric_methods(...): (line 174)
# Processing the call arguments (line 174)
# Getting the type of 'um' (line 174)
um_2466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 51), 'um', False)
# Obtaining the member 'bitwise_xor' of a type (line 174)
bitwise_xor_2467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 51), um_2466, 'bitwise_xor')
str_2468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 67), 'str', 'xor')
# Processing the call keyword arguments (line 174)
kwargs_2469 = {}
# Getting the type of '_numeric_methods' (line 174)
_numeric_methods_2465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 174)
_numeric_methods_call_result_2470 = invoke(stypy.reporting.localization.Localization(__file__, 174, 34), _numeric_methods_2465, *[bitwise_xor_2467, str_2468], **kwargs_2469)

# Obtaining the member '__getitem__' of a type (line 174)
getitem___2471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), _numeric_methods_call_result_2470, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 174)
subscript_call_result_2472 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), getitem___2471, int_2464)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1863' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2473, 'tuple_var_assignment_1863', subscript_call_result_2472)

# Assigning a Subscript to a Name (line 174):

# Obtaining the type of the subscript
int_2474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 4), 'int')

# Call to _numeric_methods(...): (line 174)
# Processing the call arguments (line 174)
# Getting the type of 'um' (line 174)
um_2476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 51), 'um', False)
# Obtaining the member 'bitwise_xor' of a type (line 174)
bitwise_xor_2477 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 51), um_2476, 'bitwise_xor')
str_2478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 67), 'str', 'xor')
# Processing the call keyword arguments (line 174)
kwargs_2479 = {}
# Getting the type of '_numeric_methods' (line 174)
_numeric_methods_2475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 34), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 174)
_numeric_methods_call_result_2480 = invoke(stypy.reporting.localization.Localization(__file__, 174, 34), _numeric_methods_2475, *[bitwise_xor_2477, str_2478], **kwargs_2479)

# Obtaining the member '__getitem__' of a type (line 174)
getitem___2481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 4), _numeric_methods_call_result_2480, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 174)
subscript_call_result_2482 = invoke(stypy.reporting.localization.Localization(__file__, 174, 4), getitem___2481, int_2474)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1864' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2483, 'tuple_var_assignment_1864', subscript_call_result_2482)

# Assigning a Name to a Name (line 174):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1862' of a type
tuple_var_assignment_1862_2485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2484, 'tuple_var_assignment_1862')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__xor__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2486, '__xor__', tuple_var_assignment_1862_2485)

# Assigning a Name to a Name (line 174):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1863' of a type
tuple_var_assignment_1863_2488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2487, 'tuple_var_assignment_1863')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__rxor__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2489, '__rxor__', tuple_var_assignment_1863_2488)

# Assigning a Name to a Name (line 174):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1864' of a type
tuple_var_assignment_1864_2491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2490, 'tuple_var_assignment_1864')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__ixor__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2492, '__ixor__', tuple_var_assignment_1864_2491)

# Assigning a Subscript to a Name (line 175):

# Obtaining the type of the subscript
int_2493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 4), 'int')

# Call to _numeric_methods(...): (line 175)
# Processing the call arguments (line 175)
# Getting the type of 'um' (line 175)
um_2495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 48), 'um', False)
# Obtaining the member 'bitwise_or' of a type (line 175)
bitwise_or_2496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 48), um_2495, 'bitwise_or')
str_2497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 63), 'str', 'or')
# Processing the call keyword arguments (line 175)
kwargs_2498 = {}
# Getting the type of '_numeric_methods' (line 175)
_numeric_methods_2494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 31), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 175)
_numeric_methods_call_result_2499 = invoke(stypy.reporting.localization.Localization(__file__, 175, 31), _numeric_methods_2494, *[bitwise_or_2496, str_2497], **kwargs_2498)

# Obtaining the member '__getitem__' of a type (line 175)
getitem___2500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 4), _numeric_methods_call_result_2499, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 175)
subscript_call_result_2501 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), getitem___2500, int_2493)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1865' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2502, 'tuple_var_assignment_1865', subscript_call_result_2501)

# Assigning a Subscript to a Name (line 175):

# Obtaining the type of the subscript
int_2503 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 4), 'int')

# Call to _numeric_methods(...): (line 175)
# Processing the call arguments (line 175)
# Getting the type of 'um' (line 175)
um_2505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 48), 'um', False)
# Obtaining the member 'bitwise_or' of a type (line 175)
bitwise_or_2506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 48), um_2505, 'bitwise_or')
str_2507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 63), 'str', 'or')
# Processing the call keyword arguments (line 175)
kwargs_2508 = {}
# Getting the type of '_numeric_methods' (line 175)
_numeric_methods_2504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 31), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 175)
_numeric_methods_call_result_2509 = invoke(stypy.reporting.localization.Localization(__file__, 175, 31), _numeric_methods_2504, *[bitwise_or_2506, str_2507], **kwargs_2508)

# Obtaining the member '__getitem__' of a type (line 175)
getitem___2510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 4), _numeric_methods_call_result_2509, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 175)
subscript_call_result_2511 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), getitem___2510, int_2503)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1866' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2512, 'tuple_var_assignment_1866', subscript_call_result_2511)

# Assigning a Subscript to a Name (line 175):

# Obtaining the type of the subscript
int_2513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 4), 'int')

# Call to _numeric_methods(...): (line 175)
# Processing the call arguments (line 175)
# Getting the type of 'um' (line 175)
um_2515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 48), 'um', False)
# Obtaining the member 'bitwise_or' of a type (line 175)
bitwise_or_2516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 48), um_2515, 'bitwise_or')
str_2517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 63), 'str', 'or')
# Processing the call keyword arguments (line 175)
kwargs_2518 = {}
# Getting the type of '_numeric_methods' (line 175)
_numeric_methods_2514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 31), '_numeric_methods', False)
# Calling _numeric_methods(args, kwargs) (line 175)
_numeric_methods_call_result_2519 = invoke(stypy.reporting.localization.Localization(__file__, 175, 31), _numeric_methods_2514, *[bitwise_or_2516, str_2517], **kwargs_2518)

# Obtaining the member '__getitem__' of a type (line 175)
getitem___2520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 4), _numeric_methods_call_result_2519, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 175)
subscript_call_result_2521 = invoke(stypy.reporting.localization.Localization(__file__, 175, 4), getitem___2520, int_2513)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member 'tuple_var_assignment_1867' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2522, 'tuple_var_assignment_1867', subscript_call_result_2521)

# Assigning a Name to a Name (line 175):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1865' of a type
tuple_var_assignment_1865_2524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2523, 'tuple_var_assignment_1865')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__or__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2525, '__or__', tuple_var_assignment_1865_2524)

# Assigning a Name to a Name (line 175):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1866' of a type
tuple_var_assignment_1866_2527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2526, 'tuple_var_assignment_1866')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__ror__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2528, '__ror__', tuple_var_assignment_1866_2527)

# Assigning a Name to a Name (line 175):
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Obtaining the member 'tuple_var_assignment_1867' of a type
tuple_var_assignment_1867_2530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2529, 'tuple_var_assignment_1867')
# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__ior__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2531, '__ior__', tuple_var_assignment_1867_2530)

# Assigning a Call to a Name (line 178):

# Call to _unary_method(...): (line 178)
# Processing the call arguments (line 178)
# Getting the type of 'um' (line 178)
um_2533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 28), 'um', False)
# Obtaining the member 'negative' of a type (line 178)
negative_2534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 28), um_2533, 'negative')
str_2535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 41), 'str', 'neg')
# Processing the call keyword arguments (line 178)
kwargs_2536 = {}
# Getting the type of '_unary_method' (line 178)
_unary_method_2532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 14), '_unary_method', False)
# Calling _unary_method(args, kwargs) (line 178)
_unary_method_call_result_2537 = invoke(stypy.reporting.localization.Localization(__file__, 178, 14), _unary_method_2532, *[negative_2534, str_2535], **kwargs_2536)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__neg__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2538, '__neg__', _unary_method_call_result_2537)

# Assigning a Call to a Name (line 179):

# Call to _unary_method(...): (line 179)
# Processing the call arguments (line 179)
# Getting the type of 'um' (line 179)
um_2540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 'um', False)
# Obtaining the member 'positive' of a type (line 179)
positive_2541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 28), um_2540, 'positive')
str_2542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 41), 'str', 'pos')
# Processing the call keyword arguments (line 179)
kwargs_2543 = {}
# Getting the type of '_unary_method' (line 179)
_unary_method_2539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), '_unary_method', False)
# Calling _unary_method(args, kwargs) (line 179)
_unary_method_call_result_2544 = invoke(stypy.reporting.localization.Localization(__file__, 179, 14), _unary_method_2539, *[positive_2541, str_2542], **kwargs_2543)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__pos__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2545, '__pos__', _unary_method_call_result_2544)

# Assigning a Call to a Name (line 180):

# Call to _unary_method(...): (line 180)
# Processing the call arguments (line 180)
# Getting the type of 'um' (line 180)
um_2547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 28), 'um', False)
# Obtaining the member 'absolute' of a type (line 180)
absolute_2548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 28), um_2547, 'absolute')
str_2549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 41), 'str', 'abs')
# Processing the call keyword arguments (line 180)
kwargs_2550 = {}
# Getting the type of '_unary_method' (line 180)
_unary_method_2546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 14), '_unary_method', False)
# Calling _unary_method(args, kwargs) (line 180)
_unary_method_call_result_2551 = invoke(stypy.reporting.localization.Localization(__file__, 180, 14), _unary_method_2546, *[absolute_2548, str_2549], **kwargs_2550)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__abs__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2552, '__abs__', _unary_method_call_result_2551)

# Assigning a Call to a Name (line 181):

# Call to _unary_method(...): (line 181)
# Processing the call arguments (line 181)
# Getting the type of 'um' (line 181)
um_2554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 31), 'um', False)
# Obtaining the member 'invert' of a type (line 181)
invert_2555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 181, 31), um_2554, 'invert')
str_2556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 181, 42), 'str', 'invert')
# Processing the call keyword arguments (line 181)
kwargs_2557 = {}
# Getting the type of '_unary_method' (line 181)
_unary_method_2553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 17), '_unary_method', False)
# Calling _unary_method(args, kwargs) (line 181)
_unary_method_call_result_2558 = invoke(stypy.reporting.localization.Localization(__file__, 181, 17), _unary_method_2553, *[invert_2555, str_2556], **kwargs_2557)

# Getting the type of 'NDArrayOperatorsMixin'
NDArrayOperatorsMixin_2559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NDArrayOperatorsMixin')
# Setting the type of the member '__invert__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NDArrayOperatorsMixin_2559, '__invert__', _unary_method_call_result_2558)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
