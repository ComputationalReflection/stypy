
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from . import _ccallback_c
2: 
3: import ctypes
4: 
5: PyCFuncPtr = ctypes.CFUNCTYPE(ctypes.c_void_p).__bases__[0]
6: 
7: ffi = None
8: 
9: class CData(object):
10:     pass
11: 
12: def _import_cffi():
13:     global ffi, CData
14: 
15:     if ffi is not None:
16:         return
17: 
18:     try:
19:         import cffi
20:         ffi = cffi.FFI()
21:         CData = ffi.CData
22:     except ImportError:
23:         ffi = False
24: 
25: 
26: class LowLevelCallable(tuple):
27:     '''
28:     Low-level callback function.
29: 
30:     Parameters
31:     ----------
32:     function : {PyCapsule, ctypes function pointer, cffi function pointer}
33:         Low-level callback function.
34:     user_data : {PyCapsule, ctypes void pointer, cffi void pointer}
35:         User data to pass on to the callback function.
36:     signature : str, optional
37:         Signature of the function. If omitted, determined from *function*,
38:         if possible.
39: 
40:     Attributes
41:     ----------
42:     function
43:         Callback function given
44:     user_data
45:         User data given
46:     signature
47:         Signature of the function.
48: 
49:     Methods
50:     -------
51:     from_cython
52:         Class method for constructing callables from Cython C-exported
53:         functions.
54: 
55:     Notes
56:     -----
57:     The argument ``function`` can be one of:
58: 
59:     - PyCapsule, whose name contains the C function signature
60:     - ctypes function pointer
61:     - cffi function pointer
62: 
63:     The signature of the low-level callback must match one of  those expected 
64:     by the routine it is passed to.
65: 
66:     If constructing low-level functions from a PyCapsule, the name of the 
67:     capsule must be the corresponding signature, in the format::
68: 
69:         return_type (arg1_type, arg2_type, ...)
70: 
71:     For example::
72: 
73:         "void (double)"
74:         "double (double, int *, void *)"
75: 
76:     The context of a PyCapsule passed in as ``function`` is used as ``user_data``, 
77:     if an explicit value for `user_data` was not given.
78: 
79:     '''
80: 
81:     # Make the class immutable
82:     __slots__ = ()
83: 
84:     def __new__(cls, function, user_data=None, signature=None):
85:         # We need to hold a reference to the function & user data,
86:         # to prevent them going out of scope
87:         item = cls._parse_callback(function, user_data, signature)
88:         return tuple.__new__(cls, (item, function, user_data))
89: 
90:     def __repr__(self):
91:         return "LowLevelCallable({!r}, {!r})".format(self.function, self.user_data)
92: 
93:     @property
94:     def function(self):
95:         return tuple.__getitem__(self, 1)
96: 
97:     @property
98:     def user_data(self):
99:         return tuple.__getitem__(self, 2)
100: 
101:     @property
102:     def signature(self):
103:         return _ccallback_c.get_capsule_signature(tuple.__getitem__(self, 0))
104: 
105:     def __getitem__(self, idx):
106:         raise ValueError()
107: 
108:     @classmethod
109:     def from_cython(cls, module, name, user_data=None, signature=None):
110:         '''
111:         Create a low-level callback function from an exported Cython function.
112: 
113:         Parameters
114:         ----------
115:         module : module
116:             Cython module where the exported function resides
117:         name : str
118:             Name of the exported function
119:         user_data : {PyCapsule, ctypes void pointer, cffi void pointer}, optional
120:             User data to pass on to the callback function.
121:         signature : str, optional
122:             Signature of the function. If omitted, determined from *function*.
123: 
124:         '''
125:         try:
126:             function = module.__pyx_capi__[name]
127:         except AttributeError:
128:             raise ValueError("Given module is not a Cython module with __pyx_capi__ attribute")
129:         except KeyError:
130:             raise ValueError("No function {!r} found in __pyx_capi__ of the module".format(name))
131:         return cls(function, user_data, signature)
132: 
133:     @classmethod
134:     def _parse_callback(cls, obj, user_data=None, signature=None):
135:         _import_cffi()
136: 
137:         if isinstance(obj, LowLevelCallable):
138:             func = tuple.__getitem__(obj, 0)
139:         elif isinstance(obj, PyCFuncPtr):
140:             func, signature = _get_ctypes_func(obj, signature)
141:         elif isinstance(obj, CData):
142:             func, signature = _get_cffi_func(obj, signature)
143:         elif _ccallback_c.check_capsule(obj):
144:             func = obj
145:         else:
146:             raise ValueError("Given input is not a callable or a low-level callable (pycapsule/ctypes/cffi)")
147: 
148:         if isinstance(user_data, ctypes.c_void_p):
149:             context = _get_ctypes_data(user_data)
150:         elif isinstance(user_data, CData):
151:             context = _get_cffi_data(user_data)
152:         elif user_data is None:
153:             context = 0
154:         elif _ccallback_c.check_capsule(user_data):
155:             context = user_data
156:         else:
157:             raise ValueError("Given user data is not a valid low-level void* pointer (pycapsule/ctypes/cffi)")
158: 
159:         return _ccallback_c.get_raw_capsule(func, signature, context)
160: 
161: 
162: #
163: # ctypes helpers
164: #
165: 
166: def _get_ctypes_func(func, signature=None):
167:     # Get function pointer
168:     func_ptr = ctypes.cast(func, ctypes.c_void_p).value
169: 
170:     # Construct function signature
171:     if signature is None:
172:         signature = _typename_from_ctypes(func.restype) + " ("
173:         for j, arg in enumerate(func.argtypes):
174:             if j == 0:
175:                 signature += _typename_from_ctypes(arg)
176:             else:
177:                 signature += ", " + _typename_from_ctypes(arg)
178:         signature += ")"
179: 
180:     return func_ptr, signature
181: 
182: 
183: def _typename_from_ctypes(item):
184:     if item is None:
185:         return "void"
186:     elif item is ctypes.c_void_p:
187:         return "void *"
188: 
189:     name = item.__name__
190: 
191:     pointer_level = 0
192:     while name.startswith("LP_"):
193:         pointer_level += 1
194:         name = name[3:]
195: 
196:     if name.startswith('c_'):
197:         name = name[2:]
198: 
199:     if pointer_level > 0:
200:         name += " " + "*"*pointer_level
201: 
202:     return name
203: 
204: 
205: def _get_ctypes_data(data):
206:     # Get voidp pointer
207:     return ctypes.cast(data, ctypes.c_void_p).value
208: 
209: 
210: #
211: # CFFI helpers
212: #
213: 
214: def _get_cffi_func(func, signature=None):
215:     # Get function pointer
216:     func_ptr = ffi.cast('uintptr_t', func)
217: 
218:     # Get signature
219:     if signature is None:
220:         signature = ffi.getctype(ffi.typeof(func)).replace('(*)', ' ')
221: 
222:     return func_ptr, signature
223: 
224: 
225: def _get_cffi_data(data):
226:     # Get pointer
227:     return ffi.cast('uintptr_t', data)
228: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from scipy._lib import _ccallback_c' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
import_708180 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy._lib')

if (type(import_708180) is not StypyTypeError):

    if (import_708180 != 'pyd_module'):
        __import__(import_708180)
        sys_modules_708181 = sys.modules[import_708180]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy._lib', sys_modules_708181.module_type_store, module_type_store, ['_ccallback_c'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_708181, sys_modules_708181.module_type_store, module_type_store)
    else:
        from scipy._lib import _ccallback_c

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy._lib', None, module_type_store, ['_ccallback_c'], [_ccallback_c])

else:
    # Assigning a type to the variable 'scipy._lib' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'scipy._lib', import_708180)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import ctypes' statement (line 3)
import ctypes

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'ctypes', ctypes, module_type_store)


# Assigning a Subscript to a Name (line 5):

# Assigning a Subscript to a Name (line 5):

# Obtaining the type of the subscript
int_708182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 57), 'int')

# Call to CFUNCTYPE(...): (line 5)
# Processing the call arguments (line 5)
# Getting the type of 'ctypes' (line 5)
ctypes_708185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 30), 'ctypes', False)
# Obtaining the member 'c_void_p' of a type (line 5)
c_void_p_708186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 30), ctypes_708185, 'c_void_p')
# Processing the call keyword arguments (line 5)
kwargs_708187 = {}
# Getting the type of 'ctypes' (line 5)
ctypes_708183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 13), 'ctypes', False)
# Obtaining the member 'CFUNCTYPE' of a type (line 5)
CFUNCTYPE_708184 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 13), ctypes_708183, 'CFUNCTYPE')
# Calling CFUNCTYPE(args, kwargs) (line 5)
CFUNCTYPE_call_result_708188 = invoke(stypy.reporting.localization.Localization(__file__, 5, 13), CFUNCTYPE_708184, *[c_void_p_708186], **kwargs_708187)

# Obtaining the member '__bases__' of a type (line 5)
bases___708189 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 13), CFUNCTYPE_call_result_708188, '__bases__')
# Obtaining the member '__getitem__' of a type (line 5)
getitem___708190 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 13), bases___708189, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 5)
subscript_call_result_708191 = invoke(stypy.reporting.localization.Localization(__file__, 5, 13), getitem___708190, int_708182)

# Assigning a type to the variable 'PyCFuncPtr' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'PyCFuncPtr', subscript_call_result_708191)

# Assigning a Name to a Name (line 7):

# Assigning a Name to a Name (line 7):
# Getting the type of 'None' (line 7)
None_708192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 6), 'None')
# Assigning a type to the variable 'ffi' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'ffi', None_708192)
# Declaration of the 'CData' class

class CData(object, ):
    pass

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CData.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CData' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'CData', CData)

@norecursion
def _import_cffi(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_import_cffi'
    module_type_store = module_type_store.open_function_context('_import_cffi', 12, 0, False)
    
    # Passed parameters checking function
    _import_cffi.stypy_localization = localization
    _import_cffi.stypy_type_of_self = None
    _import_cffi.stypy_type_store = module_type_store
    _import_cffi.stypy_function_name = '_import_cffi'
    _import_cffi.stypy_param_names_list = []
    _import_cffi.stypy_varargs_param_name = None
    _import_cffi.stypy_kwargs_param_name = None
    _import_cffi.stypy_call_defaults = defaults
    _import_cffi.stypy_call_varargs = varargs
    _import_cffi.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_import_cffi', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_import_cffi', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_import_cffi(...)' code ##################

    # Marking variables as global (line 13)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 13, 4), 'ffi')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 13, 4), 'CData')
    
    # Type idiom detected: calculating its left and rigth part (line 15)
    # Getting the type of 'ffi' (line 15)
    ffi_708193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'ffi')
    # Getting the type of 'None' (line 15)
    None_708194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'None')
    
    (may_be_708195, more_types_in_union_708196) = may_not_be_none(ffi_708193, None_708194)

    if may_be_708195:

        if more_types_in_union_708196:
            # Runtime conditional SSA (line 15)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'stypy_return_type' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'stypy_return_type', types.NoneType)

        if more_types_in_union_708196:
            # SSA join for if statement (line 15)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # SSA begins for try-except statement (line 18)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 8))
    
    # 'import cffi' statement (line 19)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/scipy/_lib/')
    import_708197 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 8), 'cffi')

    if (type(import_708197) is not StypyTypeError):

        if (import_708197 != 'pyd_module'):
            __import__(import_708197)
            sys_modules_708198 = sys.modules[import_708197]
            import_module(stypy.reporting.localization.Localization(__file__, 19, 8), 'cffi', sys_modules_708198.module_type_store, module_type_store)
        else:
            import cffi

            import_module(stypy.reporting.localization.Localization(__file__, 19, 8), 'cffi', cffi, module_type_store)

    else:
        # Assigning a type to the variable 'cffi' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'cffi', import_708197)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/scipy/_lib/')
    
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 20):
    
    # Call to FFI(...): (line 20)
    # Processing the call keyword arguments (line 20)
    kwargs_708201 = {}
    # Getting the type of 'cffi' (line 20)
    cffi_708199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 14), 'cffi', False)
    # Obtaining the member 'FFI' of a type (line 20)
    FFI_708200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 14), cffi_708199, 'FFI')
    # Calling FFI(args, kwargs) (line 20)
    FFI_call_result_708202 = invoke(stypy.reporting.localization.Localization(__file__, 20, 14), FFI_708200, *[], **kwargs_708201)
    
    # Assigning a type to the variable 'ffi' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'ffi', FFI_call_result_708202)
    
    # Assigning a Attribute to a Name (line 21):
    
    # Assigning a Attribute to a Name (line 21):
    # Getting the type of 'ffi' (line 21)
    ffi_708203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 16), 'ffi')
    # Obtaining the member 'CData' of a type (line 21)
    CData_708204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 16), ffi_708203, 'CData')
    # Assigning a type to the variable 'CData' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'CData', CData_708204)
    # SSA branch for the except part of a try statement (line 18)
    # SSA branch for the except 'ImportError' branch of a try statement (line 18)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 23):
    
    # Assigning a Name to a Name (line 23):
    # Getting the type of 'False' (line 23)
    False_708205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 14), 'False')
    # Assigning a type to the variable 'ffi' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'ffi', False_708205)
    # SSA join for try-except statement (line 18)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_import_cffi(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_import_cffi' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_708206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708206)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_import_cffi'
    return stypy_return_type_708206

# Assigning a type to the variable '_import_cffi' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), '_import_cffi', _import_cffi)
# Declaration of the 'LowLevelCallable' class
# Getting the type of 'tuple' (line 26)
tuple_708207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 23), 'tuple')

class LowLevelCallable(tuple_708207, ):
    str_708208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, (-1)), 'str', '\n    Low-level callback function.\n\n    Parameters\n    ----------\n    function : {PyCapsule, ctypes function pointer, cffi function pointer}\n        Low-level callback function.\n    user_data : {PyCapsule, ctypes void pointer, cffi void pointer}\n        User data to pass on to the callback function.\n    signature : str, optional\n        Signature of the function. If omitted, determined from *function*,\n        if possible.\n\n    Attributes\n    ----------\n    function\n        Callback function given\n    user_data\n        User data given\n    signature\n        Signature of the function.\n\n    Methods\n    -------\n    from_cython\n        Class method for constructing callables from Cython C-exported\n        functions.\n\n    Notes\n    -----\n    The argument ``function`` can be one of:\n\n    - PyCapsule, whose name contains the C function signature\n    - ctypes function pointer\n    - cffi function pointer\n\n    The signature of the low-level callback must match one of  those expected \n    by the routine it is passed to.\n\n    If constructing low-level functions from a PyCapsule, the name of the \n    capsule must be the corresponding signature, in the format::\n\n        return_type (arg1_type, arg2_type, ...)\n\n    For example::\n\n        "void (double)"\n        "double (double, int *, void *)"\n\n    The context of a PyCapsule passed in as ``function`` is used as ``user_data``, \n    if an explicit value for `user_data` was not given.\n\n    ')
    
    # Assigning a Tuple to a Name (line 82):

    @norecursion
    def __new__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 84)
        None_708209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 41), 'None')
        # Getting the type of 'None' (line 84)
        None_708210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 57), 'None')
        defaults = [None_708209, None_708210]
        # Create a new context for function '__new__'
        module_type_store = module_type_store.open_function_context('__new__', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_localization', localization)
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_function_name', 'LowLevelCallable.__new__')
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_param_names_list', ['function', 'user_data', 'signature'])
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LowLevelCallable.__new__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable.__new__', ['function', 'user_data', 'signature'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__new__', localization, ['function', 'user_data', 'signature'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__new__(...)' code ##################

        
        # Assigning a Call to a Name (line 87):
        
        # Assigning a Call to a Name (line 87):
        
        # Call to _parse_callback(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'function' (line 87)
        function_708213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 35), 'function', False)
        # Getting the type of 'user_data' (line 87)
        user_data_708214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 45), 'user_data', False)
        # Getting the type of 'signature' (line 87)
        signature_708215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 56), 'signature', False)
        # Processing the call keyword arguments (line 87)
        kwargs_708216 = {}
        # Getting the type of 'cls' (line 87)
        cls_708211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 15), 'cls', False)
        # Obtaining the member '_parse_callback' of a type (line 87)
        _parse_callback_708212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 15), cls_708211, '_parse_callback')
        # Calling _parse_callback(args, kwargs) (line 87)
        _parse_callback_call_result_708217 = invoke(stypy.reporting.localization.Localization(__file__, 87, 15), _parse_callback_708212, *[function_708213, user_data_708214, signature_708215], **kwargs_708216)
        
        # Assigning a type to the variable 'item' (line 87)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'item', _parse_callback_call_result_708217)
        
        # Call to __new__(...): (line 88)
        # Processing the call arguments (line 88)
        # Getting the type of 'cls' (line 88)
        cls_708220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 29), 'cls', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 88)
        tuple_708221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 35), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 88)
        # Adding element type (line 88)
        # Getting the type of 'item' (line 88)
        item_708222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 35), 'item', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 35), tuple_708221, item_708222)
        # Adding element type (line 88)
        # Getting the type of 'function' (line 88)
        function_708223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 41), 'function', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 35), tuple_708221, function_708223)
        # Adding element type (line 88)
        # Getting the type of 'user_data' (line 88)
        user_data_708224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 51), 'user_data', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 88, 35), tuple_708221, user_data_708224)
        
        # Processing the call keyword arguments (line 88)
        kwargs_708225 = {}
        # Getting the type of 'tuple' (line 88)
        tuple_708218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'tuple', False)
        # Obtaining the member '__new__' of a type (line 88)
        new___708219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 15), tuple_708218, '__new__')
        # Calling __new__(args, kwargs) (line 88)
        new___call_result_708226 = invoke(stypy.reporting.localization.Localization(__file__, 88, 15), new___708219, *[cls_708220, tuple_708221], **kwargs_708225)
        
        # Assigning a type to the variable 'stypy_return_type' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'stypy_return_type', new___call_result_708226)
        
        # ################# End of '__new__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__new__' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_708227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708227)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__new__'
        return stypy_return_type_708227


    @norecursion
    def stypy__repr__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__repr__'
        module_type_store = module_type_store.open_function_context('__repr__', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_localization', localization)
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_function_name', 'LowLevelCallable.stypy__repr__')
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_param_names_list', [])
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LowLevelCallable.stypy__repr__.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable.stypy__repr__', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to format(...): (line 91)
        # Processing the call arguments (line 91)
        # Getting the type of 'self' (line 91)
        self_708230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 53), 'self', False)
        # Obtaining the member 'function' of a type (line 91)
        function_708231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 53), self_708230, 'function')
        # Getting the type of 'self' (line 91)
        self_708232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 68), 'self', False)
        # Obtaining the member 'user_data' of a type (line 91)
        user_data_708233 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 68), self_708232, 'user_data')
        # Processing the call keyword arguments (line 91)
        kwargs_708234 = {}
        str_708228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, 15), 'str', 'LowLevelCallable({!r}, {!r})')
        # Obtaining the member 'format' of a type (line 91)
        format_708229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 15), str_708228, 'format')
        # Calling format(args, kwargs) (line 91)
        format_call_result_708235 = invoke(stypy.reporting.localization.Localization(__file__, 91, 15), format_708229, *[function_708231, user_data_708233], **kwargs_708234)
        
        # Assigning a type to the variable 'stypy_return_type' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'stypy_return_type', format_call_result_708235)
        
        # ################# End of '__repr__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__repr__' in the type store
        # Getting the type of 'stypy_return_type' (line 90)
        stypy_return_type_708236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708236)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__repr__'
        return stypy_return_type_708236


    @norecursion
    def function(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'function'
        module_type_store = module_type_store.open_function_context('function', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LowLevelCallable.function.__dict__.__setitem__('stypy_localization', localization)
        LowLevelCallable.function.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LowLevelCallable.function.__dict__.__setitem__('stypy_type_store', module_type_store)
        LowLevelCallable.function.__dict__.__setitem__('stypy_function_name', 'LowLevelCallable.function')
        LowLevelCallable.function.__dict__.__setitem__('stypy_param_names_list', [])
        LowLevelCallable.function.__dict__.__setitem__('stypy_varargs_param_name', None)
        LowLevelCallable.function.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LowLevelCallable.function.__dict__.__setitem__('stypy_call_defaults', defaults)
        LowLevelCallable.function.__dict__.__setitem__('stypy_call_varargs', varargs)
        LowLevelCallable.function.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LowLevelCallable.function.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable.function', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'function', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'function(...)' code ##################

        
        # Call to __getitem__(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'self' (line 95)
        self_708239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 33), 'self', False)
        int_708240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 39), 'int')
        # Processing the call keyword arguments (line 95)
        kwargs_708241 = {}
        # Getting the type of 'tuple' (line 95)
        tuple_708237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 15), 'tuple', False)
        # Obtaining the member '__getitem__' of a type (line 95)
        getitem___708238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 15), tuple_708237, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 95)
        getitem___call_result_708242 = invoke(stypy.reporting.localization.Localization(__file__, 95, 15), getitem___708238, *[self_708239, int_708240], **kwargs_708241)
        
        # Assigning a type to the variable 'stypy_return_type' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'stypy_return_type', getitem___call_result_708242)
        
        # ################# End of 'function(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'function' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_708243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708243)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'function'
        return stypy_return_type_708243


    @norecursion
    def user_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'user_data'
        module_type_store = module_type_store.open_function_context('user_data', 97, 4, False)
        # Assigning a type to the variable 'self' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_localization', localization)
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_function_name', 'LowLevelCallable.user_data')
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_param_names_list', [])
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LowLevelCallable.user_data.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable.user_data', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'user_data', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'user_data(...)' code ##################

        
        # Call to __getitem__(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'self' (line 99)
        self_708246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'self', False)
        int_708247 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 39), 'int')
        # Processing the call keyword arguments (line 99)
        kwargs_708248 = {}
        # Getting the type of 'tuple' (line 99)
        tuple_708244 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 15), 'tuple', False)
        # Obtaining the member '__getitem__' of a type (line 99)
        getitem___708245 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 15), tuple_708244, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 99)
        getitem___call_result_708249 = invoke(stypy.reporting.localization.Localization(__file__, 99, 15), getitem___708245, *[self_708246, int_708247], **kwargs_708248)
        
        # Assigning a type to the variable 'stypy_return_type' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'stypy_return_type', getitem___call_result_708249)
        
        # ################# End of 'user_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'user_data' in the type store
        # Getting the type of 'stypy_return_type' (line 97)
        stypy_return_type_708250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708250)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'user_data'
        return stypy_return_type_708250


    @norecursion
    def signature(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'signature'
        module_type_store = module_type_store.open_function_context('signature', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LowLevelCallable.signature.__dict__.__setitem__('stypy_localization', localization)
        LowLevelCallable.signature.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LowLevelCallable.signature.__dict__.__setitem__('stypy_type_store', module_type_store)
        LowLevelCallable.signature.__dict__.__setitem__('stypy_function_name', 'LowLevelCallable.signature')
        LowLevelCallable.signature.__dict__.__setitem__('stypy_param_names_list', [])
        LowLevelCallable.signature.__dict__.__setitem__('stypy_varargs_param_name', None)
        LowLevelCallable.signature.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LowLevelCallable.signature.__dict__.__setitem__('stypy_call_defaults', defaults)
        LowLevelCallable.signature.__dict__.__setitem__('stypy_call_varargs', varargs)
        LowLevelCallable.signature.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LowLevelCallable.signature.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable.signature', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'signature', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'signature(...)' code ##################

        
        # Call to get_capsule_signature(...): (line 103)
        # Processing the call arguments (line 103)
        
        # Call to __getitem__(...): (line 103)
        # Processing the call arguments (line 103)
        # Getting the type of 'self' (line 103)
        self_708255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 68), 'self', False)
        int_708256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 74), 'int')
        # Processing the call keyword arguments (line 103)
        kwargs_708257 = {}
        # Getting the type of 'tuple' (line 103)
        tuple_708253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 50), 'tuple', False)
        # Obtaining the member '__getitem__' of a type (line 103)
        getitem___708254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 50), tuple_708253, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 103)
        getitem___call_result_708258 = invoke(stypy.reporting.localization.Localization(__file__, 103, 50), getitem___708254, *[self_708255, int_708256], **kwargs_708257)
        
        # Processing the call keyword arguments (line 103)
        kwargs_708259 = {}
        # Getting the type of '_ccallback_c' (line 103)
        _ccallback_c_708251 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), '_ccallback_c', False)
        # Obtaining the member 'get_capsule_signature' of a type (line 103)
        get_capsule_signature_708252 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 15), _ccallback_c_708251, 'get_capsule_signature')
        # Calling get_capsule_signature(args, kwargs) (line 103)
        get_capsule_signature_call_result_708260 = invoke(stypy.reporting.localization.Localization(__file__, 103, 15), get_capsule_signature_708252, *[getitem___call_result_708258], **kwargs_708259)
        
        # Assigning a type to the variable 'stypy_return_type' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'stypy_return_type', get_capsule_signature_call_result_708260)
        
        # ################# End of 'signature(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'signature' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_708261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708261)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'signature'
        return stypy_return_type_708261


    @norecursion
    def __getitem__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__getitem__'
        module_type_store = module_type_store.open_function_context('__getitem__', 105, 4, False)
        # Assigning a type to the variable 'self' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_localization', localization)
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_type_store', module_type_store)
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_function_name', 'LowLevelCallable.__getitem__')
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_param_names_list', ['idx'])
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_varargs_param_name', None)
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_call_defaults', defaults)
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_call_varargs', varargs)
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LowLevelCallable.__getitem__.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable.__getitem__', ['idx'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__getitem__', localization, ['idx'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__getitem__(...)' code ##################

        
        # Call to ValueError(...): (line 106)
        # Processing the call keyword arguments (line 106)
        kwargs_708263 = {}
        # Getting the type of 'ValueError' (line 106)
        ValueError_708262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 14), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 106)
        ValueError_call_result_708264 = invoke(stypy.reporting.localization.Localization(__file__, 106, 14), ValueError_708262, *[], **kwargs_708263)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 106, 8), ValueError_call_result_708264, 'raise parameter', BaseException)
        
        # ################# End of '__getitem__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__getitem__' in the type store
        # Getting the type of 'stypy_return_type' (line 105)
        stypy_return_type_708265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__getitem__'
        return stypy_return_type_708265


    @norecursion
    def from_cython(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 109)
        None_708266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 49), 'None')
        # Getting the type of 'None' (line 109)
        None_708267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 65), 'None')
        defaults = [None_708266, None_708267]
        # Create a new context for function 'from_cython'
        module_type_store = module_type_store.open_function_context('from_cython', 108, 4, False)
        # Assigning a type to the variable 'self' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_localization', localization)
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_type_store', module_type_store)
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_function_name', 'LowLevelCallable.from_cython')
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_param_names_list', ['module', 'name', 'user_data', 'signature'])
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_varargs_param_name', None)
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_call_defaults', defaults)
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_call_varargs', varargs)
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LowLevelCallable.from_cython.__dict__.__setitem__('stypy_declared_arg_number', 5)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable.from_cython', ['module', 'name', 'user_data', 'signature'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'from_cython', localization, ['module', 'name', 'user_data', 'signature'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'from_cython(...)' code ##################

        str_708268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, (-1)), 'str', '\n        Create a low-level callback function from an exported Cython function.\n\n        Parameters\n        ----------\n        module : module\n            Cython module where the exported function resides\n        name : str\n            Name of the exported function\n        user_data : {PyCapsule, ctypes void pointer, cffi void pointer}, optional\n            User data to pass on to the callback function.\n        signature : str, optional\n            Signature of the function. If omitted, determined from *function*.\n\n        ')
        
        
        # SSA begins for try-except statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Subscript to a Name (line 126):
        
        # Assigning a Subscript to a Name (line 126):
        
        # Obtaining the type of the subscript
        # Getting the type of 'name' (line 126)
        name_708269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 43), 'name')
        # Getting the type of 'module' (line 126)
        module_708270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 23), 'module')
        # Obtaining the member '__pyx_capi__' of a type (line 126)
        pyx_capi___708271 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), module_708270, '__pyx_capi__')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___708272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 23), pyx_capi___708271, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_708273 = invoke(stypy.reporting.localization.Localization(__file__, 126, 23), getitem___708272, name_708269)
        
        # Assigning a type to the variable 'function' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), 'function', subscript_call_result_708273)
        # SSA branch for the except part of a try statement (line 125)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 125)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 128)
        # Processing the call arguments (line 128)
        str_708275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 29), 'str', 'Given module is not a Cython module with __pyx_capi__ attribute')
        # Processing the call keyword arguments (line 128)
        kwargs_708276 = {}
        # Getting the type of 'ValueError' (line 128)
        ValueError_708274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 128)
        ValueError_call_result_708277 = invoke(stypy.reporting.localization.Localization(__file__, 128, 18), ValueError_708274, *[str_708275], **kwargs_708276)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 128, 12), ValueError_call_result_708277, 'raise parameter', BaseException)
        # SSA branch for the except 'KeyError' branch of a try statement (line 125)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 130)
        # Processing the call arguments (line 130)
        
        # Call to format(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'name' (line 130)
        name_708281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 91), 'name', False)
        # Processing the call keyword arguments (line 130)
        kwargs_708282 = {}
        str_708279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 29), 'str', 'No function {!r} found in __pyx_capi__ of the module')
        # Obtaining the member 'format' of a type (line 130)
        format_708280 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 29), str_708279, 'format')
        # Calling format(args, kwargs) (line 130)
        format_call_result_708283 = invoke(stypy.reporting.localization.Localization(__file__, 130, 29), format_708280, *[name_708281], **kwargs_708282)
        
        # Processing the call keyword arguments (line 130)
        kwargs_708284 = {}
        # Getting the type of 'ValueError' (line 130)
        ValueError_708278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 130)
        ValueError_call_result_708285 = invoke(stypy.reporting.localization.Localization(__file__, 130, 18), ValueError_708278, *[format_call_result_708283], **kwargs_708284)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 130, 12), ValueError_call_result_708285, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to cls(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'function' (line 131)
        function_708287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 19), 'function', False)
        # Getting the type of 'user_data' (line 131)
        user_data_708288 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 29), 'user_data', False)
        # Getting the type of 'signature' (line 131)
        signature_708289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 40), 'signature', False)
        # Processing the call keyword arguments (line 131)
        kwargs_708290 = {}
        # Getting the type of 'cls' (line 131)
        cls_708286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'cls', False)
        # Calling cls(args, kwargs) (line 131)
        cls_call_result_708291 = invoke(stypy.reporting.localization.Localization(__file__, 131, 15), cls_708286, *[function_708287, user_data_708288, signature_708289], **kwargs_708290)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', cls_call_result_708291)
        
        # ################# End of 'from_cython(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'from_cython' in the type store
        # Getting the type of 'stypy_return_type' (line 108)
        stypy_return_type_708292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708292)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'from_cython'
        return stypy_return_type_708292


    @norecursion
    def _parse_callback(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 134)
        None_708293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 44), 'None')
        # Getting the type of 'None' (line 134)
        None_708294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 60), 'None')
        defaults = [None_708293, None_708294]
        # Create a new context for function '_parse_callback'
        module_type_store = module_type_store.open_function_context('_parse_callback', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_localization', localization)
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_type_store', module_type_store)
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_function_name', 'LowLevelCallable._parse_callback')
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_param_names_list', ['obj', 'user_data', 'signature'])
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_varargs_param_name', None)
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_call_defaults', defaults)
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_call_varargs', varargs)
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LowLevelCallable._parse_callback.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable._parse_callback', ['obj', 'user_data', 'signature'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_parse_callback', localization, ['obj', 'user_data', 'signature'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_parse_callback(...)' code ##################

        
        # Call to _import_cffi(...): (line 135)
        # Processing the call keyword arguments (line 135)
        kwargs_708296 = {}
        # Getting the type of '_import_cffi' (line 135)
        _import_cffi_708295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), '_import_cffi', False)
        # Calling _import_cffi(args, kwargs) (line 135)
        _import_cffi_call_result_708297 = invoke(stypy.reporting.localization.Localization(__file__, 135, 8), _import_cffi_708295, *[], **kwargs_708296)
        
        
        
        # Call to isinstance(...): (line 137)
        # Processing the call arguments (line 137)
        # Getting the type of 'obj' (line 137)
        obj_708299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 22), 'obj', False)
        # Getting the type of 'LowLevelCallable' (line 137)
        LowLevelCallable_708300 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 27), 'LowLevelCallable', False)
        # Processing the call keyword arguments (line 137)
        kwargs_708301 = {}
        # Getting the type of 'isinstance' (line 137)
        isinstance_708298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 137)
        isinstance_call_result_708302 = invoke(stypy.reporting.localization.Localization(__file__, 137, 11), isinstance_708298, *[obj_708299, LowLevelCallable_708300], **kwargs_708301)
        
        # Testing the type of an if condition (line 137)
        if_condition_708303 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 137, 8), isinstance_call_result_708302)
        # Assigning a type to the variable 'if_condition_708303' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'if_condition_708303', if_condition_708303)
        # SSA begins for if statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 138):
        
        # Assigning a Call to a Name (line 138):
        
        # Call to __getitem__(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'obj' (line 138)
        obj_708306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 37), 'obj', False)
        int_708307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 42), 'int')
        # Processing the call keyword arguments (line 138)
        kwargs_708308 = {}
        # Getting the type of 'tuple' (line 138)
        tuple_708304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 19), 'tuple', False)
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___708305 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 19), tuple_708304, '__getitem__')
        # Calling __getitem__(args, kwargs) (line 138)
        getitem___call_result_708309 = invoke(stypy.reporting.localization.Localization(__file__, 138, 19), getitem___708305, *[obj_708306, int_708307], **kwargs_708308)
        
        # Assigning a type to the variable 'func' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'func', getitem___call_result_708309)
        # SSA branch for the else part of an if statement (line 137)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 139)
        # Processing the call arguments (line 139)
        # Getting the type of 'obj' (line 139)
        obj_708311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 24), 'obj', False)
        # Getting the type of 'PyCFuncPtr' (line 139)
        PyCFuncPtr_708312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 29), 'PyCFuncPtr', False)
        # Processing the call keyword arguments (line 139)
        kwargs_708313 = {}
        # Getting the type of 'isinstance' (line 139)
        isinstance_708310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 139)
        isinstance_call_result_708314 = invoke(stypy.reporting.localization.Localization(__file__, 139, 13), isinstance_708310, *[obj_708311, PyCFuncPtr_708312], **kwargs_708313)
        
        # Testing the type of an if condition (line 139)
        if_condition_708315 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 13), isinstance_call_result_708314)
        # Assigning a type to the variable 'if_condition_708315' (line 139)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 13), 'if_condition_708315', if_condition_708315)
        # SSA begins for if statement (line 139)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 140):
        
        # Assigning a Subscript to a Name (line 140):
        
        # Obtaining the type of the subscript
        int_708316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 12), 'int')
        
        # Call to _get_ctypes_func(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'obj' (line 140)
        obj_708318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 47), 'obj', False)
        # Getting the type of 'signature' (line 140)
        signature_708319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 52), 'signature', False)
        # Processing the call keyword arguments (line 140)
        kwargs_708320 = {}
        # Getting the type of '_get_ctypes_func' (line 140)
        _get_ctypes_func_708317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), '_get_ctypes_func', False)
        # Calling _get_ctypes_func(args, kwargs) (line 140)
        _get_ctypes_func_call_result_708321 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), _get_ctypes_func_708317, *[obj_708318, signature_708319], **kwargs_708320)
        
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___708322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), _get_ctypes_func_call_result_708321, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_708323 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), getitem___708322, int_708316)
        
        # Assigning a type to the variable 'tuple_var_assignment_708176' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'tuple_var_assignment_708176', subscript_call_result_708323)
        
        # Assigning a Subscript to a Name (line 140):
        
        # Obtaining the type of the subscript
        int_708324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 12), 'int')
        
        # Call to _get_ctypes_func(...): (line 140)
        # Processing the call arguments (line 140)
        # Getting the type of 'obj' (line 140)
        obj_708326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 47), 'obj', False)
        # Getting the type of 'signature' (line 140)
        signature_708327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 52), 'signature', False)
        # Processing the call keyword arguments (line 140)
        kwargs_708328 = {}
        # Getting the type of '_get_ctypes_func' (line 140)
        _get_ctypes_func_708325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), '_get_ctypes_func', False)
        # Calling _get_ctypes_func(args, kwargs) (line 140)
        _get_ctypes_func_call_result_708329 = invoke(stypy.reporting.localization.Localization(__file__, 140, 30), _get_ctypes_func_708325, *[obj_708326, signature_708327], **kwargs_708328)
        
        # Obtaining the member '__getitem__' of a type (line 140)
        getitem___708330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 12), _get_ctypes_func_call_result_708329, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 140)
        subscript_call_result_708331 = invoke(stypy.reporting.localization.Localization(__file__, 140, 12), getitem___708330, int_708324)
        
        # Assigning a type to the variable 'tuple_var_assignment_708177' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'tuple_var_assignment_708177', subscript_call_result_708331)
        
        # Assigning a Name to a Name (line 140):
        # Getting the type of 'tuple_var_assignment_708176' (line 140)
        tuple_var_assignment_708176_708332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'tuple_var_assignment_708176')
        # Assigning a type to the variable 'func' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'func', tuple_var_assignment_708176_708332)
        
        # Assigning a Name to a Name (line 140):
        # Getting the type of 'tuple_var_assignment_708177' (line 140)
        tuple_var_assignment_708177_708333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'tuple_var_assignment_708177')
        # Assigning a type to the variable 'signature' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'signature', tuple_var_assignment_708177_708333)
        # SSA branch for the else part of an if statement (line 139)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'obj' (line 141)
        obj_708335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 24), 'obj', False)
        # Getting the type of 'CData' (line 141)
        CData_708336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 29), 'CData', False)
        # Processing the call keyword arguments (line 141)
        kwargs_708337 = {}
        # Getting the type of 'isinstance' (line 141)
        isinstance_708334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 141)
        isinstance_call_result_708338 = invoke(stypy.reporting.localization.Localization(__file__, 141, 13), isinstance_708334, *[obj_708335, CData_708336], **kwargs_708337)
        
        # Testing the type of an if condition (line 141)
        if_condition_708339 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 13), isinstance_call_result_708338)
        # Assigning a type to the variable 'if_condition_708339' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 13), 'if_condition_708339', if_condition_708339)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Tuple (line 142):
        
        # Assigning a Subscript to a Name (line 142):
        
        # Obtaining the type of the subscript
        int_708340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 12), 'int')
        
        # Call to _get_cffi_func(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'obj' (line 142)
        obj_708342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'obj', False)
        # Getting the type of 'signature' (line 142)
        signature_708343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 50), 'signature', False)
        # Processing the call keyword arguments (line 142)
        kwargs_708344 = {}
        # Getting the type of '_get_cffi_func' (line 142)
        _get_cffi_func_708341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), '_get_cffi_func', False)
        # Calling _get_cffi_func(args, kwargs) (line 142)
        _get_cffi_func_call_result_708345 = invoke(stypy.reporting.localization.Localization(__file__, 142, 30), _get_cffi_func_708341, *[obj_708342, signature_708343], **kwargs_708344)
        
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___708346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), _get_cffi_func_call_result_708345, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_708347 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), getitem___708346, int_708340)
        
        # Assigning a type to the variable 'tuple_var_assignment_708178' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'tuple_var_assignment_708178', subscript_call_result_708347)
        
        # Assigning a Subscript to a Name (line 142):
        
        # Obtaining the type of the subscript
        int_708348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 12), 'int')
        
        # Call to _get_cffi_func(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'obj' (line 142)
        obj_708350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 45), 'obj', False)
        # Getting the type of 'signature' (line 142)
        signature_708351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 50), 'signature', False)
        # Processing the call keyword arguments (line 142)
        kwargs_708352 = {}
        # Getting the type of '_get_cffi_func' (line 142)
        _get_cffi_func_708349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 30), '_get_cffi_func', False)
        # Calling _get_cffi_func(args, kwargs) (line 142)
        _get_cffi_func_call_result_708353 = invoke(stypy.reporting.localization.Localization(__file__, 142, 30), _get_cffi_func_708349, *[obj_708350, signature_708351], **kwargs_708352)
        
        # Obtaining the member '__getitem__' of a type (line 142)
        getitem___708354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 12), _get_cffi_func_call_result_708353, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 142)
        subscript_call_result_708355 = invoke(stypy.reporting.localization.Localization(__file__, 142, 12), getitem___708354, int_708348)
        
        # Assigning a type to the variable 'tuple_var_assignment_708179' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'tuple_var_assignment_708179', subscript_call_result_708355)
        
        # Assigning a Name to a Name (line 142):
        # Getting the type of 'tuple_var_assignment_708178' (line 142)
        tuple_var_assignment_708178_708356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'tuple_var_assignment_708178')
        # Assigning a type to the variable 'func' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'func', tuple_var_assignment_708178_708356)
        
        # Assigning a Name to a Name (line 142):
        # Getting the type of 'tuple_var_assignment_708179' (line 142)
        tuple_var_assignment_708179_708357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'tuple_var_assignment_708179')
        # Assigning a type to the variable 'signature' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 18), 'signature', tuple_var_assignment_708179_708357)
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to check_capsule(...): (line 143)
        # Processing the call arguments (line 143)
        # Getting the type of 'obj' (line 143)
        obj_708360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 40), 'obj', False)
        # Processing the call keyword arguments (line 143)
        kwargs_708361 = {}
        # Getting the type of '_ccallback_c' (line 143)
        _ccallback_c_708358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 13), '_ccallback_c', False)
        # Obtaining the member 'check_capsule' of a type (line 143)
        check_capsule_708359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 13), _ccallback_c_708358, 'check_capsule')
        # Calling check_capsule(args, kwargs) (line 143)
        check_capsule_call_result_708362 = invoke(stypy.reporting.localization.Localization(__file__, 143, 13), check_capsule_708359, *[obj_708360], **kwargs_708361)
        
        # Testing the type of an if condition (line 143)
        if_condition_708363 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 143, 13), check_capsule_call_result_708362)
        # Assigning a type to the variable 'if_condition_708363' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 13), 'if_condition_708363', if_condition_708363)
        # SSA begins for if statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 144):
        
        # Assigning a Name to a Name (line 144):
        # Getting the type of 'obj' (line 144)
        obj_708364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'obj')
        # Assigning a type to the variable 'func' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'func', obj_708364)
        # SSA branch for the else part of an if statement (line 143)
        module_type_store.open_ssa_branch('else')
        
        # Call to ValueError(...): (line 146)
        # Processing the call arguments (line 146)
        str_708366 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 29), 'str', 'Given input is not a callable or a low-level callable (pycapsule/ctypes/cffi)')
        # Processing the call keyword arguments (line 146)
        kwargs_708367 = {}
        # Getting the type of 'ValueError' (line 146)
        ValueError_708365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 146)
        ValueError_call_result_708368 = invoke(stypy.reporting.localization.Localization(__file__, 146, 18), ValueError_708365, *[str_708366], **kwargs_708367)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 146, 12), ValueError_call_result_708368, 'raise parameter', BaseException)
        # SSA join for if statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 139)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 137)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to isinstance(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'user_data' (line 148)
        user_data_708370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 22), 'user_data', False)
        # Getting the type of 'ctypes' (line 148)
        ctypes_708371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 33), 'ctypes', False)
        # Obtaining the member 'c_void_p' of a type (line 148)
        c_void_p_708372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 33), ctypes_708371, 'c_void_p')
        # Processing the call keyword arguments (line 148)
        kwargs_708373 = {}
        # Getting the type of 'isinstance' (line 148)
        isinstance_708369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 148)
        isinstance_call_result_708374 = invoke(stypy.reporting.localization.Localization(__file__, 148, 11), isinstance_708369, *[user_data_708370, c_void_p_708372], **kwargs_708373)
        
        # Testing the type of an if condition (line 148)
        if_condition_708375 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 8), isinstance_call_result_708374)
        # Assigning a type to the variable 'if_condition_708375' (line 148)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 8), 'if_condition_708375', if_condition_708375)
        # SSA begins for if statement (line 148)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 149):
        
        # Assigning a Call to a Name (line 149):
        
        # Call to _get_ctypes_data(...): (line 149)
        # Processing the call arguments (line 149)
        # Getting the type of 'user_data' (line 149)
        user_data_708377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 39), 'user_data', False)
        # Processing the call keyword arguments (line 149)
        kwargs_708378 = {}
        # Getting the type of '_get_ctypes_data' (line 149)
        _get_ctypes_data_708376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 22), '_get_ctypes_data', False)
        # Calling _get_ctypes_data(args, kwargs) (line 149)
        _get_ctypes_data_call_result_708379 = invoke(stypy.reporting.localization.Localization(__file__, 149, 22), _get_ctypes_data_708376, *[user_data_708377], **kwargs_708378)
        
        # Assigning a type to the variable 'context' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 12), 'context', _get_ctypes_data_call_result_708379)
        # SSA branch for the else part of an if statement (line 148)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to isinstance(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'user_data' (line 150)
        user_data_708381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'user_data', False)
        # Getting the type of 'CData' (line 150)
        CData_708382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 35), 'CData', False)
        # Processing the call keyword arguments (line 150)
        kwargs_708383 = {}
        # Getting the type of 'isinstance' (line 150)
        isinstance_708380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 150)
        isinstance_call_result_708384 = invoke(stypy.reporting.localization.Localization(__file__, 150, 13), isinstance_708380, *[user_data_708381, CData_708382], **kwargs_708383)
        
        # Testing the type of an if condition (line 150)
        if_condition_708385 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 13), isinstance_call_result_708384)
        # Assigning a type to the variable 'if_condition_708385' (line 150)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 13), 'if_condition_708385', if_condition_708385)
        # SSA begins for if statement (line 150)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 151):
        
        # Assigning a Call to a Name (line 151):
        
        # Call to _get_cffi_data(...): (line 151)
        # Processing the call arguments (line 151)
        # Getting the type of 'user_data' (line 151)
        user_data_708387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 37), 'user_data', False)
        # Processing the call keyword arguments (line 151)
        kwargs_708388 = {}
        # Getting the type of '_get_cffi_data' (line 151)
        _get_cffi_data_708386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), '_get_cffi_data', False)
        # Calling _get_cffi_data(args, kwargs) (line 151)
        _get_cffi_data_call_result_708389 = invoke(stypy.reporting.localization.Localization(__file__, 151, 22), _get_cffi_data_708386, *[user_data_708387], **kwargs_708388)
        
        # Assigning a type to the variable 'context' (line 151)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'context', _get_cffi_data_call_result_708389)
        # SSA branch for the else part of an if statement (line 150)
        module_type_store.open_ssa_branch('else')
        
        # Type idiom detected: calculating its left and rigth part (line 152)
        # Getting the type of 'user_data' (line 152)
        user_data_708390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 13), 'user_data')
        # Getting the type of 'None' (line 152)
        None_708391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 26), 'None')
        
        (may_be_708392, more_types_in_union_708393) = may_be_none(user_data_708390, None_708391)

        if may_be_708392:

            if more_types_in_union_708393:
                # Runtime conditional SSA (line 152)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Num to a Name (line 153):
            
            # Assigning a Num to a Name (line 153):
            int_708394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 22), 'int')
            # Assigning a type to the variable 'context' (line 153)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'context', int_708394)

            if more_types_in_union_708393:
                # Runtime conditional SSA for else branch (line 152)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_708392) or more_types_in_union_708393):
            
            
            # Call to check_capsule(...): (line 154)
            # Processing the call arguments (line 154)
            # Getting the type of 'user_data' (line 154)
            user_data_708397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 40), 'user_data', False)
            # Processing the call keyword arguments (line 154)
            kwargs_708398 = {}
            # Getting the type of '_ccallback_c' (line 154)
            _ccallback_c_708395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), '_ccallback_c', False)
            # Obtaining the member 'check_capsule' of a type (line 154)
            check_capsule_708396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 13), _ccallback_c_708395, 'check_capsule')
            # Calling check_capsule(args, kwargs) (line 154)
            check_capsule_call_result_708399 = invoke(stypy.reporting.localization.Localization(__file__, 154, 13), check_capsule_708396, *[user_data_708397], **kwargs_708398)
            
            # Testing the type of an if condition (line 154)
            if_condition_708400 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 154, 13), check_capsule_call_result_708399)
            # Assigning a type to the variable 'if_condition_708400' (line 154)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), 'if_condition_708400', if_condition_708400)
            # SSA begins for if statement (line 154)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 155):
            
            # Assigning a Name to a Name (line 155):
            # Getting the type of 'user_data' (line 155)
            user_data_708401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 22), 'user_data')
            # Assigning a type to the variable 'context' (line 155)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'context', user_data_708401)
            # SSA branch for the else part of an if statement (line 154)
            module_type_store.open_ssa_branch('else')
            
            # Call to ValueError(...): (line 157)
            # Processing the call arguments (line 157)
            str_708403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 157, 29), 'str', 'Given user data is not a valid low-level void* pointer (pycapsule/ctypes/cffi)')
            # Processing the call keyword arguments (line 157)
            kwargs_708404 = {}
            # Getting the type of 'ValueError' (line 157)
            ValueError_708402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 18), 'ValueError', False)
            # Calling ValueError(args, kwargs) (line 157)
            ValueError_call_result_708405 = invoke(stypy.reporting.localization.Localization(__file__, 157, 18), ValueError_708402, *[str_708403], **kwargs_708404)
            
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 157, 12), ValueError_call_result_708405, 'raise parameter', BaseException)
            # SSA join for if statement (line 154)
            module_type_store = module_type_store.join_ssa_context()
            

            if (may_be_708392 and more_types_in_union_708393):
                # SSA join for if statement (line 152)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 150)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 148)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to get_raw_capsule(...): (line 159)
        # Processing the call arguments (line 159)
        # Getting the type of 'func' (line 159)
        func_708408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 44), 'func', False)
        # Getting the type of 'signature' (line 159)
        signature_708409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 50), 'signature', False)
        # Getting the type of 'context' (line 159)
        context_708410 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 61), 'context', False)
        # Processing the call keyword arguments (line 159)
        kwargs_708411 = {}
        # Getting the type of '_ccallback_c' (line 159)
        _ccallback_c_708406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 15), '_ccallback_c', False)
        # Obtaining the member 'get_raw_capsule' of a type (line 159)
        get_raw_capsule_708407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 15), _ccallback_c_708406, 'get_raw_capsule')
        # Calling get_raw_capsule(args, kwargs) (line 159)
        get_raw_capsule_call_result_708412 = invoke(stypy.reporting.localization.Localization(__file__, 159, 15), get_raw_capsule_708407, *[func_708408, signature_708409, context_708410], **kwargs_708411)
        
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 8), 'stypy_return_type', get_raw_capsule_call_result_708412)
        
        # ################# End of '_parse_callback(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_parse_callback' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_708413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_708413)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_parse_callback'
        return stypy_return_type_708413


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 0, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LowLevelCallable.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LowLevelCallable' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'LowLevelCallable', LowLevelCallable)

# Assigning a Tuple to a Name (line 82):

# Obtaining an instance of the builtin type 'tuple' (line 82)
tuple_708414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 16), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 82)

# Getting the type of 'LowLevelCallable'
LowLevelCallable_708415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'LowLevelCallable')
# Setting the type of the member '__slots__' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), LowLevelCallable_708415, '__slots__', tuple_708414)

@norecursion
def _get_ctypes_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 166)
    None_708416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 37), 'None')
    defaults = [None_708416]
    # Create a new context for function '_get_ctypes_func'
    module_type_store = module_type_store.open_function_context('_get_ctypes_func', 166, 0, False)
    
    # Passed parameters checking function
    _get_ctypes_func.stypy_localization = localization
    _get_ctypes_func.stypy_type_of_self = None
    _get_ctypes_func.stypy_type_store = module_type_store
    _get_ctypes_func.stypy_function_name = '_get_ctypes_func'
    _get_ctypes_func.stypy_param_names_list = ['func', 'signature']
    _get_ctypes_func.stypy_varargs_param_name = None
    _get_ctypes_func.stypy_kwargs_param_name = None
    _get_ctypes_func.stypy_call_defaults = defaults
    _get_ctypes_func.stypy_call_varargs = varargs
    _get_ctypes_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_ctypes_func', ['func', 'signature'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_ctypes_func', localization, ['func', 'signature'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_ctypes_func(...)' code ##################

    
    # Assigning a Attribute to a Name (line 168):
    
    # Assigning a Attribute to a Name (line 168):
    
    # Call to cast(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'func' (line 168)
    func_708419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'func', False)
    # Getting the type of 'ctypes' (line 168)
    ctypes_708420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 33), 'ctypes', False)
    # Obtaining the member 'c_void_p' of a type (line 168)
    c_void_p_708421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 33), ctypes_708420, 'c_void_p')
    # Processing the call keyword arguments (line 168)
    kwargs_708422 = {}
    # Getting the type of 'ctypes' (line 168)
    ctypes_708417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 15), 'ctypes', False)
    # Obtaining the member 'cast' of a type (line 168)
    cast_708418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), ctypes_708417, 'cast')
    # Calling cast(args, kwargs) (line 168)
    cast_call_result_708423 = invoke(stypy.reporting.localization.Localization(__file__, 168, 15), cast_708418, *[func_708419, c_void_p_708421], **kwargs_708422)
    
    # Obtaining the member 'value' of a type (line 168)
    value_708424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 15), cast_call_result_708423, 'value')
    # Assigning a type to the variable 'func_ptr' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'func_ptr', value_708424)
    
    # Type idiom detected: calculating its left and rigth part (line 171)
    # Getting the type of 'signature' (line 171)
    signature_708425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 7), 'signature')
    # Getting the type of 'None' (line 171)
    None_708426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 20), 'None')
    
    (may_be_708427, more_types_in_union_708428) = may_be_none(signature_708425, None_708426)

    if may_be_708427:

        if more_types_in_union_708428:
            # Runtime conditional SSA (line 171)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BinOp to a Name (line 172):
        
        # Assigning a BinOp to a Name (line 172):
        
        # Call to _typename_from_ctypes(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'func' (line 172)
        func_708430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 42), 'func', False)
        # Obtaining the member 'restype' of a type (line 172)
        restype_708431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 172, 42), func_708430, 'restype')
        # Processing the call keyword arguments (line 172)
        kwargs_708432 = {}
        # Getting the type of '_typename_from_ctypes' (line 172)
        _typename_from_ctypes_708429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 20), '_typename_from_ctypes', False)
        # Calling _typename_from_ctypes(args, kwargs) (line 172)
        _typename_from_ctypes_call_result_708433 = invoke(stypy.reporting.localization.Localization(__file__, 172, 20), _typename_from_ctypes_708429, *[restype_708431], **kwargs_708432)
        
        str_708434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 58), 'str', ' (')
        # Applying the binary operator '+' (line 172)
        result_add_708435 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 20), '+', _typename_from_ctypes_call_result_708433, str_708434)
        
        # Assigning a type to the variable 'signature' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 8), 'signature', result_add_708435)
        
        
        # Call to enumerate(...): (line 173)
        # Processing the call arguments (line 173)
        # Getting the type of 'func' (line 173)
        func_708437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 32), 'func', False)
        # Obtaining the member 'argtypes' of a type (line 173)
        argtypes_708438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 32), func_708437, 'argtypes')
        # Processing the call keyword arguments (line 173)
        kwargs_708439 = {}
        # Getting the type of 'enumerate' (line 173)
        enumerate_708436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'enumerate', False)
        # Calling enumerate(args, kwargs) (line 173)
        enumerate_call_result_708440 = invoke(stypy.reporting.localization.Localization(__file__, 173, 22), enumerate_708436, *[argtypes_708438], **kwargs_708439)
        
        # Testing the type of a for loop iterable (line 173)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 173, 8), enumerate_call_result_708440)
        # Getting the type of the for loop variable (line 173)
        for_loop_var_708441 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 173, 8), enumerate_call_result_708440)
        # Assigning a type to the variable 'j' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'j', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 8), for_loop_var_708441))
        # Assigning a type to the variable 'arg' (line 173)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 173, 8), for_loop_var_708441))
        # SSA begins for a for statement (line 173)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'j' (line 174)
        j_708442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 15), 'j')
        int_708443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 174, 20), 'int')
        # Applying the binary operator '==' (line 174)
        result_eq_708444 = python_operator(stypy.reporting.localization.Localization(__file__, 174, 15), '==', j_708442, int_708443)
        
        # Testing the type of an if condition (line 174)
        if_condition_708445 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 174, 12), result_eq_708444)
        # Assigning a type to the variable 'if_condition_708445' (line 174)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 12), 'if_condition_708445', if_condition_708445)
        # SSA begins for if statement (line 174)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'signature' (line 175)
        signature_708446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'signature')
        
        # Call to _typename_from_ctypes(...): (line 175)
        # Processing the call arguments (line 175)
        # Getting the type of 'arg' (line 175)
        arg_708448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 51), 'arg', False)
        # Processing the call keyword arguments (line 175)
        kwargs_708449 = {}
        # Getting the type of '_typename_from_ctypes' (line 175)
        _typename_from_ctypes_708447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 29), '_typename_from_ctypes', False)
        # Calling _typename_from_ctypes(args, kwargs) (line 175)
        _typename_from_ctypes_call_result_708450 = invoke(stypy.reporting.localization.Localization(__file__, 175, 29), _typename_from_ctypes_708447, *[arg_708448], **kwargs_708449)
        
        # Applying the binary operator '+=' (line 175)
        result_iadd_708451 = python_operator(stypy.reporting.localization.Localization(__file__, 175, 16), '+=', signature_708446, _typename_from_ctypes_call_result_708450)
        # Assigning a type to the variable 'signature' (line 175)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 16), 'signature', result_iadd_708451)
        
        # SSA branch for the else part of an if statement (line 174)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'signature' (line 177)
        signature_708452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'signature')
        str_708453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 29), 'str', ', ')
        
        # Call to _typename_from_ctypes(...): (line 177)
        # Processing the call arguments (line 177)
        # Getting the type of 'arg' (line 177)
        arg_708455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 58), 'arg', False)
        # Processing the call keyword arguments (line 177)
        kwargs_708456 = {}
        # Getting the type of '_typename_from_ctypes' (line 177)
        _typename_from_ctypes_708454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 36), '_typename_from_ctypes', False)
        # Calling _typename_from_ctypes(args, kwargs) (line 177)
        _typename_from_ctypes_call_result_708457 = invoke(stypy.reporting.localization.Localization(__file__, 177, 36), _typename_from_ctypes_708454, *[arg_708455], **kwargs_708456)
        
        # Applying the binary operator '+' (line 177)
        result_add_708458 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 29), '+', str_708453, _typename_from_ctypes_call_result_708457)
        
        # Applying the binary operator '+=' (line 177)
        result_iadd_708459 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 16), '+=', signature_708452, result_add_708458)
        # Assigning a type to the variable 'signature' (line 177)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'signature', result_iadd_708459)
        
        # SSA join for if statement (line 174)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'signature' (line 178)
        signature_708460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'signature')
        str_708461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 21), 'str', ')')
        # Applying the binary operator '+=' (line 178)
        result_iadd_708462 = python_operator(stypy.reporting.localization.Localization(__file__, 178, 8), '+=', signature_708460, str_708461)
        # Assigning a type to the variable 'signature' (line 178)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 8), 'signature', result_iadd_708462)
        

        if more_types_in_union_708428:
            # SSA join for if statement (line 171)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 180)
    tuple_708463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 180)
    # Adding element type (line 180)
    # Getting the type of 'func_ptr' (line 180)
    func_ptr_708464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 11), 'func_ptr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 11), tuple_708463, func_ptr_708464)
    # Adding element type (line 180)
    # Getting the type of 'signature' (line 180)
    signature_708465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 21), 'signature')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 11), tuple_708463, signature_708465)
    
    # Assigning a type to the variable 'stypy_return_type' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'stypy_return_type', tuple_708463)
    
    # ################# End of '_get_ctypes_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_ctypes_func' in the type store
    # Getting the type of 'stypy_return_type' (line 166)
    stypy_return_type_708466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708466)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_ctypes_func'
    return stypy_return_type_708466

# Assigning a type to the variable '_get_ctypes_func' (line 166)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 0), '_get_ctypes_func', _get_ctypes_func)

@norecursion
def _typename_from_ctypes(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_typename_from_ctypes'
    module_type_store = module_type_store.open_function_context('_typename_from_ctypes', 183, 0, False)
    
    # Passed parameters checking function
    _typename_from_ctypes.stypy_localization = localization
    _typename_from_ctypes.stypy_type_of_self = None
    _typename_from_ctypes.stypy_type_store = module_type_store
    _typename_from_ctypes.stypy_function_name = '_typename_from_ctypes'
    _typename_from_ctypes.stypy_param_names_list = ['item']
    _typename_from_ctypes.stypy_varargs_param_name = None
    _typename_from_ctypes.stypy_kwargs_param_name = None
    _typename_from_ctypes.stypy_call_defaults = defaults
    _typename_from_ctypes.stypy_call_varargs = varargs
    _typename_from_ctypes.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_typename_from_ctypes', ['item'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_typename_from_ctypes', localization, ['item'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_typename_from_ctypes(...)' code ##################

    
    # Type idiom detected: calculating its left and rigth part (line 184)
    # Getting the type of 'item' (line 184)
    item_708467 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 7), 'item')
    # Getting the type of 'None' (line 184)
    None_708468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'None')
    
    (may_be_708469, more_types_in_union_708470) = may_be_none(item_708467, None_708468)

    if may_be_708469:

        if more_types_in_union_708470:
            # Runtime conditional SSA (line 184)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        str_708471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 15), 'str', 'void')
        # Assigning a type to the variable 'stypy_return_type' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 8), 'stypy_return_type', str_708471)

        if more_types_in_union_708470:
            # Runtime conditional SSA for else branch (line 184)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_708469) or more_types_in_union_708470):
        
        
        # Getting the type of 'item' (line 186)
        item_708472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 9), 'item')
        # Getting the type of 'ctypes' (line 186)
        ctypes_708473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 17), 'ctypes')
        # Obtaining the member 'c_void_p' of a type (line 186)
        c_void_p_708474 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 186, 17), ctypes_708473, 'c_void_p')
        # Applying the binary operator 'is' (line 186)
        result_is__708475 = python_operator(stypy.reporting.localization.Localization(__file__, 186, 9), 'is', item_708472, c_void_p_708474)
        
        # Testing the type of an if condition (line 186)
        if_condition_708476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 9), result_is__708475)
        # Assigning a type to the variable 'if_condition_708476' (line 186)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 9), 'if_condition_708476', if_condition_708476)
        # SSA begins for if statement (line 186)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        str_708477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, 15), 'str', 'void *')
        # Assigning a type to the variable 'stypy_return_type' (line 187)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', str_708477)
        # SSA join for if statement (line 186)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_708469 and more_types_in_union_708470):
            # SSA join for if statement (line 184)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Attribute to a Name (line 189):
    
    # Assigning a Attribute to a Name (line 189):
    # Getting the type of 'item' (line 189)
    item_708478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'item')
    # Obtaining the member '__name__' of a type (line 189)
    name___708479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 11), item_708478, '__name__')
    # Assigning a type to the variable 'name' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'name', name___708479)
    
    # Assigning a Num to a Name (line 191):
    
    # Assigning a Num to a Name (line 191):
    int_708480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 20), 'int')
    # Assigning a type to the variable 'pointer_level' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'pointer_level', int_708480)
    
    
    # Call to startswith(...): (line 192)
    # Processing the call arguments (line 192)
    str_708483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 192, 26), 'str', 'LP_')
    # Processing the call keyword arguments (line 192)
    kwargs_708484 = {}
    # Getting the type of 'name' (line 192)
    name_708481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 10), 'name', False)
    # Obtaining the member 'startswith' of a type (line 192)
    startswith_708482 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 192, 10), name_708481, 'startswith')
    # Calling startswith(args, kwargs) (line 192)
    startswith_call_result_708485 = invoke(stypy.reporting.localization.Localization(__file__, 192, 10), startswith_708482, *[str_708483], **kwargs_708484)
    
    # Testing the type of an if condition (line 192)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 4), startswith_call_result_708485)
    # SSA begins for while statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Getting the type of 'pointer_level' (line 193)
    pointer_level_708486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'pointer_level')
    int_708487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 25), 'int')
    # Applying the binary operator '+=' (line 193)
    result_iadd_708488 = python_operator(stypy.reporting.localization.Localization(__file__, 193, 8), '+=', pointer_level_708486, int_708487)
    # Assigning a type to the variable 'pointer_level' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'pointer_level', result_iadd_708488)
    
    
    # Assigning a Subscript to a Name (line 194):
    
    # Assigning a Subscript to a Name (line 194):
    
    # Obtaining the type of the subscript
    int_708489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 20), 'int')
    slice_708490 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 194, 15), int_708489, None, None)
    # Getting the type of 'name' (line 194)
    name_708491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 15), 'name')
    # Obtaining the member '__getitem__' of a type (line 194)
    getitem___708492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 15), name_708491, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 194)
    subscript_call_result_708493 = invoke(stypy.reporting.localization.Localization(__file__, 194, 15), getitem___708492, slice_708490)
    
    # Assigning a type to the variable 'name' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 8), 'name', subscript_call_result_708493)
    # SSA join for while statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to startswith(...): (line 196)
    # Processing the call arguments (line 196)
    str_708496 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 23), 'str', 'c_')
    # Processing the call keyword arguments (line 196)
    kwargs_708497 = {}
    # Getting the type of 'name' (line 196)
    name_708494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 7), 'name', False)
    # Obtaining the member 'startswith' of a type (line 196)
    startswith_708495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 7), name_708494, 'startswith')
    # Calling startswith(args, kwargs) (line 196)
    startswith_call_result_708498 = invoke(stypy.reporting.localization.Localization(__file__, 196, 7), startswith_708495, *[str_708496], **kwargs_708497)
    
    # Testing the type of an if condition (line 196)
    if_condition_708499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 4), startswith_call_result_708498)
    # Assigning a type to the variable 'if_condition_708499' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'if_condition_708499', if_condition_708499)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Name (line 197):
    
    # Assigning a Subscript to a Name (line 197):
    
    # Obtaining the type of the subscript
    int_708500 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 20), 'int')
    slice_708501 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 197, 15), int_708500, None, None)
    # Getting the type of 'name' (line 197)
    name_708502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'name')
    # Obtaining the member '__getitem__' of a type (line 197)
    getitem___708503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 15), name_708502, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 197)
    subscript_call_result_708504 = invoke(stypy.reporting.localization.Localization(__file__, 197, 15), getitem___708503, slice_708501)
    
    # Assigning a type to the variable 'name' (line 197)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 8), 'name', subscript_call_result_708504)
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'pointer_level' (line 199)
    pointer_level_708505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 7), 'pointer_level')
    int_708506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 23), 'int')
    # Applying the binary operator '>' (line 199)
    result_gt_708507 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 7), '>', pointer_level_708505, int_708506)
    
    # Testing the type of an if condition (line 199)
    if_condition_708508 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 4), result_gt_708507)
    # Assigning a type to the variable 'if_condition_708508' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'if_condition_708508', if_condition_708508)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'name' (line 200)
    name_708509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'name')
    str_708510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 16), 'str', ' ')
    str_708511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 22), 'str', '*')
    # Getting the type of 'pointer_level' (line 200)
    pointer_level_708512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 26), 'pointer_level')
    # Applying the binary operator '*' (line 200)
    result_mul_708513 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 22), '*', str_708511, pointer_level_708512)
    
    # Applying the binary operator '+' (line 200)
    result_add_708514 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 16), '+', str_708510, result_mul_708513)
    
    # Applying the binary operator '+=' (line 200)
    result_iadd_708515 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 8), '+=', name_708509, result_add_708514)
    # Assigning a type to the variable 'name' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 8), 'name', result_iadd_708515)
    
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'name' (line 202)
    name_708516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 11), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'stypy_return_type', name_708516)
    
    # ################# End of '_typename_from_ctypes(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_typename_from_ctypes' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_708517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708517)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_typename_from_ctypes'
    return stypy_return_type_708517

# Assigning a type to the variable '_typename_from_ctypes' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), '_typename_from_ctypes', _typename_from_ctypes)

@norecursion
def _get_ctypes_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_ctypes_data'
    module_type_store = module_type_store.open_function_context('_get_ctypes_data', 205, 0, False)
    
    # Passed parameters checking function
    _get_ctypes_data.stypy_localization = localization
    _get_ctypes_data.stypy_type_of_self = None
    _get_ctypes_data.stypy_type_store = module_type_store
    _get_ctypes_data.stypy_function_name = '_get_ctypes_data'
    _get_ctypes_data.stypy_param_names_list = ['data']
    _get_ctypes_data.stypy_varargs_param_name = None
    _get_ctypes_data.stypy_kwargs_param_name = None
    _get_ctypes_data.stypy_call_defaults = defaults
    _get_ctypes_data.stypy_call_varargs = varargs
    _get_ctypes_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_ctypes_data', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_ctypes_data', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_ctypes_data(...)' code ##################

    
    # Call to cast(...): (line 207)
    # Processing the call arguments (line 207)
    # Getting the type of 'data' (line 207)
    data_708520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 23), 'data', False)
    # Getting the type of 'ctypes' (line 207)
    ctypes_708521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 29), 'ctypes', False)
    # Obtaining the member 'c_void_p' of a type (line 207)
    c_void_p_708522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 29), ctypes_708521, 'c_void_p')
    # Processing the call keyword arguments (line 207)
    kwargs_708523 = {}
    # Getting the type of 'ctypes' (line 207)
    ctypes_708518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'ctypes', False)
    # Obtaining the member 'cast' of a type (line 207)
    cast_708519 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 11), ctypes_708518, 'cast')
    # Calling cast(args, kwargs) (line 207)
    cast_call_result_708524 = invoke(stypy.reporting.localization.Localization(__file__, 207, 11), cast_708519, *[data_708520, c_void_p_708522], **kwargs_708523)
    
    # Obtaining the member 'value' of a type (line 207)
    value_708525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 11), cast_call_result_708524, 'value')
    # Assigning a type to the variable 'stypy_return_type' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'stypy_return_type', value_708525)
    
    # ################# End of '_get_ctypes_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_ctypes_data' in the type store
    # Getting the type of 'stypy_return_type' (line 205)
    stypy_return_type_708526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708526)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_ctypes_data'
    return stypy_return_type_708526

# Assigning a type to the variable '_get_ctypes_data' (line 205)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 0), '_get_ctypes_data', _get_ctypes_data)

@norecursion
def _get_cffi_func(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 214)
    None_708527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 35), 'None')
    defaults = [None_708527]
    # Create a new context for function '_get_cffi_func'
    module_type_store = module_type_store.open_function_context('_get_cffi_func', 214, 0, False)
    
    # Passed parameters checking function
    _get_cffi_func.stypy_localization = localization
    _get_cffi_func.stypy_type_of_self = None
    _get_cffi_func.stypy_type_store = module_type_store
    _get_cffi_func.stypy_function_name = '_get_cffi_func'
    _get_cffi_func.stypy_param_names_list = ['func', 'signature']
    _get_cffi_func.stypy_varargs_param_name = None
    _get_cffi_func.stypy_kwargs_param_name = None
    _get_cffi_func.stypy_call_defaults = defaults
    _get_cffi_func.stypy_call_varargs = varargs
    _get_cffi_func.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_cffi_func', ['func', 'signature'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_cffi_func', localization, ['func', 'signature'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_cffi_func(...)' code ##################

    
    # Assigning a Call to a Name (line 216):
    
    # Assigning a Call to a Name (line 216):
    
    # Call to cast(...): (line 216)
    # Processing the call arguments (line 216)
    str_708530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 24), 'str', 'uintptr_t')
    # Getting the type of 'func' (line 216)
    func_708531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 37), 'func', False)
    # Processing the call keyword arguments (line 216)
    kwargs_708532 = {}
    # Getting the type of 'ffi' (line 216)
    ffi_708528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 15), 'ffi', False)
    # Obtaining the member 'cast' of a type (line 216)
    cast_708529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 15), ffi_708528, 'cast')
    # Calling cast(args, kwargs) (line 216)
    cast_call_result_708533 = invoke(stypy.reporting.localization.Localization(__file__, 216, 15), cast_708529, *[str_708530, func_708531], **kwargs_708532)
    
    # Assigning a type to the variable 'func_ptr' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 4), 'func_ptr', cast_call_result_708533)
    
    # Type idiom detected: calculating its left and rigth part (line 219)
    # Getting the type of 'signature' (line 219)
    signature_708534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 7), 'signature')
    # Getting the type of 'None' (line 219)
    None_708535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 20), 'None')
    
    (may_be_708536, more_types_in_union_708537) = may_be_none(signature_708534, None_708535)

    if may_be_708536:

        if more_types_in_union_708537:
            # Runtime conditional SSA (line 219)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 220):
        
        # Assigning a Call to a Name (line 220):
        
        # Call to replace(...): (line 220)
        # Processing the call arguments (line 220)
        str_708548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 59), 'str', '(*)')
        str_708549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 66), 'str', ' ')
        # Processing the call keyword arguments (line 220)
        kwargs_708550 = {}
        
        # Call to getctype(...): (line 220)
        # Processing the call arguments (line 220)
        
        # Call to typeof(...): (line 220)
        # Processing the call arguments (line 220)
        # Getting the type of 'func' (line 220)
        func_708542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 44), 'func', False)
        # Processing the call keyword arguments (line 220)
        kwargs_708543 = {}
        # Getting the type of 'ffi' (line 220)
        ffi_708540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 33), 'ffi', False)
        # Obtaining the member 'typeof' of a type (line 220)
        typeof_708541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 33), ffi_708540, 'typeof')
        # Calling typeof(args, kwargs) (line 220)
        typeof_call_result_708544 = invoke(stypy.reporting.localization.Localization(__file__, 220, 33), typeof_708541, *[func_708542], **kwargs_708543)
        
        # Processing the call keyword arguments (line 220)
        kwargs_708545 = {}
        # Getting the type of 'ffi' (line 220)
        ffi_708538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'ffi', False)
        # Obtaining the member 'getctype' of a type (line 220)
        getctype_708539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 20), ffi_708538, 'getctype')
        # Calling getctype(args, kwargs) (line 220)
        getctype_call_result_708546 = invoke(stypy.reporting.localization.Localization(__file__, 220, 20), getctype_708539, *[typeof_call_result_708544], **kwargs_708545)
        
        # Obtaining the member 'replace' of a type (line 220)
        replace_708547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 20), getctype_call_result_708546, 'replace')
        # Calling replace(args, kwargs) (line 220)
        replace_call_result_708551 = invoke(stypy.reporting.localization.Localization(__file__, 220, 20), replace_708547, *[str_708548, str_708549], **kwargs_708550)
        
        # Assigning a type to the variable 'signature' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 8), 'signature', replace_call_result_708551)

        if more_types_in_union_708537:
            # SSA join for if statement (line 219)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Obtaining an instance of the builtin type 'tuple' (line 222)
    tuple_708552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 222, 11), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 222)
    # Adding element type (line 222)
    # Getting the type of 'func_ptr' (line 222)
    func_ptr_708553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 11), 'func_ptr')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 11), tuple_708552, func_ptr_708553)
    # Adding element type (line 222)
    # Getting the type of 'signature' (line 222)
    signature_708554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 21), 'signature')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 222, 11), tuple_708552, signature_708554)
    
    # Assigning a type to the variable 'stypy_return_type' (line 222)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 222, 4), 'stypy_return_type', tuple_708552)
    
    # ################# End of '_get_cffi_func(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_cffi_func' in the type store
    # Getting the type of 'stypy_return_type' (line 214)
    stypy_return_type_708555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708555)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_cffi_func'
    return stypy_return_type_708555

# Assigning a type to the variable '_get_cffi_func' (line 214)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 0), '_get_cffi_func', _get_cffi_func)

@norecursion
def _get_cffi_data(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_cffi_data'
    module_type_store = module_type_store.open_function_context('_get_cffi_data', 225, 0, False)
    
    # Passed parameters checking function
    _get_cffi_data.stypy_localization = localization
    _get_cffi_data.stypy_type_of_self = None
    _get_cffi_data.stypy_type_store = module_type_store
    _get_cffi_data.stypy_function_name = '_get_cffi_data'
    _get_cffi_data.stypy_param_names_list = ['data']
    _get_cffi_data.stypy_varargs_param_name = None
    _get_cffi_data.stypy_kwargs_param_name = None
    _get_cffi_data.stypy_call_defaults = defaults
    _get_cffi_data.stypy_call_varargs = varargs
    _get_cffi_data.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_cffi_data', ['data'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_cffi_data', localization, ['data'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_cffi_data(...)' code ##################

    
    # Call to cast(...): (line 227)
    # Processing the call arguments (line 227)
    str_708558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 20), 'str', 'uintptr_t')
    # Getting the type of 'data' (line 227)
    data_708559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 33), 'data', False)
    # Processing the call keyword arguments (line 227)
    kwargs_708560 = {}
    # Getting the type of 'ffi' (line 227)
    ffi_708556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'ffi', False)
    # Obtaining the member 'cast' of a type (line 227)
    cast_708557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 227, 11), ffi_708556, 'cast')
    # Calling cast(args, kwargs) (line 227)
    cast_call_result_708561 = invoke(stypy.reporting.localization.Localization(__file__, 227, 11), cast_708557, *[str_708558, data_708559], **kwargs_708560)
    
    # Assigning a type to the variable 'stypy_return_type' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type', cast_call_result_708561)
    
    # ################# End of '_get_cffi_data(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_cffi_data' in the type store
    # Getting the type of 'stypy_return_type' (line 225)
    stypy_return_type_708562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_708562)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_cffi_data'
    return stypy_return_type_708562

# Assigning a type to the variable '_get_cffi_data' (line 225)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 0), '_get_cffi_data', _get_cffi_data)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
