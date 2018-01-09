
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import absolute_import
2: 
3: import os
4: import sys
5: 
6: import numpy as np
7: 
8: 
9: # Copy-pasted from Python 3.4's shutil.
10: def which(cmd, mode=os.F_OK | os.X_OK, path=None):
11:     '''Given a command, mode, and a PATH string, return the path which
12:     conforms to the given mode on the PATH, or None if there is no such
13:     file.
14: 
15:     `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result
16:     of os.environ.get("PATH"), or can be overridden with a custom search
17:     path.
18: 
19:     '''
20:     # Check that a given file can be accessed with the correct mode.
21:     # Additionally check that `file` is not a directory, as on Windows
22:     # directories pass the os.access check.
23:     def _access_check(fn, mode):
24:         return (os.path.exists(fn) and os.access(fn, mode)
25:                 and not os.path.isdir(fn))
26: 
27:     # If we're given a path with a directory part, look it up directly rather
28:     # than referring to PATH directories. This includes checking relative to the
29:     # current directory, e.g. ./script
30:     if os.path.dirname(cmd):
31:         if _access_check(cmd, mode):
32:             return cmd
33:         return None
34: 
35:     if path is None:
36:         path = os.environ.get("PATH", os.defpath)
37:     if not path:
38:         return None
39:     path = path.split(os.pathsep)
40: 
41:     if sys.platform == "win32":
42:         # The current directory takes precedence on Windows.
43:         if not os.curdir in path:
44:             path.insert(0, os.curdir)
45: 
46:         # PATHEXT is necessary to check on Windows.
47:         pathext = os.environ.get("PATHEXT", "").split(os.pathsep)
48:         # See if the given file matches any of the expected path extensions.
49:         # This will allow us to short circuit when given "python.exe".
50:         # If it does match, only test that one, otherwise we have to try
51:         # others.
52:         if any(cmd.lower().endswith(ext.lower()) for ext in pathext):
53:             files = [cmd]
54:         else:
55:             files = [cmd + ext for ext in pathext]
56:     else:
57:         # On other platforms you don't have things like PATHEXT to tell you
58:         # what file suffixes are executable, so just pass on cmd as-is.
59:         files = [cmd]
60: 
61:     seen = set()
62:     for dir in path:
63:         normdir = os.path.normcase(dir)
64:         if not normdir in seen:
65:             seen.add(normdir)
66:             for thefile in files:
67:                 name = os.path.join(dir, thefile)
68:                 if _access_check(name, mode):
69:                     return name
70:     return None
71: 
72: 
73: # Copy-pasted from numpy.lib.stride_tricks 1.11.2.
74: def _maybe_view_as_subclass(original_array, new_array):
75:     if type(original_array) is not type(new_array):
76:         # if input was an ndarray subclass and subclasses were OK,
77:         # then view the result as that subclass.
78:         new_array = new_array.view(type=type(original_array))
79:         # Since we have done something akin to a view from original_array, we
80:         # should let the subclass finalize (if it has it implemented, i.e., is
81:         # not None).
82:         if new_array.__array_finalize__:
83:             new_array.__array_finalize__(original_array)
84:     return new_array
85: 
86: 
87: # Copy-pasted from numpy.lib.stride_tricks 1.11.2.
88: def _broadcast_to(array, shape, subok, readonly):
89:     shape = tuple(shape) if np.iterable(shape) else (shape,)
90:     array = np.array(array, copy=False, subok=subok)
91:     if not shape and array.shape:
92:         raise ValueError('cannot broadcast a non-scalar to a scalar array')
93:     if any(size < 0 for size in shape):
94:         raise ValueError('all elements of broadcast shape must be non-'
95:                          'negative')
96:     needs_writeable = not readonly and array.flags.writeable
97:     extras = ['reduce_ok'] if needs_writeable else []
98:     op_flag = 'readwrite' if needs_writeable else 'readonly'
99:     broadcast = np.nditer(
100:         (array,), flags=['multi_index', 'refs_ok', 'zerosize_ok'] + extras,
101:         op_flags=[op_flag], itershape=shape, order='C').itviews[0]
102:     result = _maybe_view_as_subclass(array, broadcast)
103:     if needs_writeable and not result.flags.writeable:
104:         result.flags.writeable = True
105:     return result
106: 
107: 
108: # Copy-pasted from numpy.lib.stride_tricks 1.11.2.
109: def broadcast_to(array, shape, subok=False):
110:     '''Broadcast an array to a new shape.
111: 
112:     Parameters
113:     ----------
114:     array : array_like
115:         The array to broadcast.
116:     shape : tuple
117:         The shape of the desired array.
118:     subok : bool, optional
119:         If True, then sub-classes will be passed-through, otherwise
120:         the returned array will be forced to be a base-class array (default).
121: 
122:     Returns
123:     -------
124:     broadcast : array
125:         A readonly view on the original array with the given shape. It is
126:         typically not contiguous. Furthermore, more than one element of a
127:         broadcasted array may refer to a single memory location.
128: 
129:     Raises
130:     ------
131:     ValueError
132:         If the array is not compatible with the new shape according to NumPy's
133:         broadcasting rules.
134: 
135:     Notes
136:     -----
137:     .. versionadded:: 1.10.0
138: 
139:     Examples
140:     --------
141:     >>> x = np.array([1, 2, 3])
142:     >>> np.broadcast_to(x, (3, 3))
143:     array([[1, 2, 3],
144:            [1, 2, 3],
145:            [1, 2, 3]])
146:     '''
147:     return _broadcast_to(array, shape, subok=subok, readonly=True)
148: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import numpy' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/matplotlib/cbook/')
import_273355 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy')

if (type(import_273355) is not StypyTypeError):

    if (import_273355 != 'pyd_module'):
        __import__(import_273355)
        sys_modules_273356 = sys.modules[import_273355]
        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', sys_modules_273356.module_type_store, module_type_store)
    else:
        import numpy as np

        import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'np', numpy, module_type_store)

else:
    # Assigning a type to the variable 'numpy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'numpy', import_273355)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/matplotlib/cbook/')


@norecursion
def which(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'os' (line 10)
    os_273357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'os')
    # Obtaining the member 'F_OK' of a type (line 10)
    F_OK_273358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 20), os_273357, 'F_OK')
    # Getting the type of 'os' (line 10)
    os_273359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 30), 'os')
    # Obtaining the member 'X_OK' of a type (line 10)
    X_OK_273360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 30), os_273359, 'X_OK')
    # Applying the binary operator '|' (line 10)
    result_or__273361 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 20), '|', F_OK_273358, X_OK_273360)
    
    # Getting the type of 'None' (line 10)
    None_273362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 44), 'None')
    defaults = [result_or__273361, None_273362]
    # Create a new context for function 'which'
    module_type_store = module_type_store.open_function_context('which', 10, 0, False)
    
    # Passed parameters checking function
    which.stypy_localization = localization
    which.stypy_type_of_self = None
    which.stypy_type_store = module_type_store
    which.stypy_function_name = 'which'
    which.stypy_param_names_list = ['cmd', 'mode', 'path']
    which.stypy_varargs_param_name = None
    which.stypy_kwargs_param_name = None
    which.stypy_call_defaults = defaults
    which.stypy_call_varargs = varargs
    which.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'which', ['cmd', 'mode', 'path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'which', localization, ['cmd', 'mode', 'path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'which(...)' code ##################

    str_273363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, (-1)), 'str', 'Given a command, mode, and a PATH string, return the path which\n    conforms to the given mode on the PATH, or None if there is no such\n    file.\n\n    `mode` defaults to os.F_OK | os.X_OK. `path` defaults to the result\n    of os.environ.get("PATH"), or can be overridden with a custom search\n    path.\n\n    ')

    @norecursion
    def _access_check(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_access_check'
        module_type_store = module_type_store.open_function_context('_access_check', 23, 4, False)
        
        # Passed parameters checking function
        _access_check.stypy_localization = localization
        _access_check.stypy_type_of_self = None
        _access_check.stypy_type_store = module_type_store
        _access_check.stypy_function_name = '_access_check'
        _access_check.stypy_param_names_list = ['fn', 'mode']
        _access_check.stypy_varargs_param_name = None
        _access_check.stypy_kwargs_param_name = None
        _access_check.stypy_call_defaults = defaults
        _access_check.stypy_call_varargs = varargs
        _access_check.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_access_check', ['fn', 'mode'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_access_check', localization, ['fn', 'mode'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_access_check(...)' code ##################

        
        # Evaluating a boolean operation
        
        # Call to exists(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'fn' (line 24)
        fn_273367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 31), 'fn', False)
        # Processing the call keyword arguments (line 24)
        kwargs_273368 = {}
        # Getting the type of 'os' (line 24)
        os_273364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 16), 'os', False)
        # Obtaining the member 'path' of a type (line 24)
        path_273365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), os_273364, 'path')
        # Obtaining the member 'exists' of a type (line 24)
        exists_273366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 16), path_273365, 'exists')
        # Calling exists(args, kwargs) (line 24)
        exists_call_result_273369 = invoke(stypy.reporting.localization.Localization(__file__, 24, 16), exists_273366, *[fn_273367], **kwargs_273368)
        
        
        # Call to access(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'fn' (line 24)
        fn_273372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 49), 'fn', False)
        # Getting the type of 'mode' (line 24)
        mode_273373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 53), 'mode', False)
        # Processing the call keyword arguments (line 24)
        kwargs_273374 = {}
        # Getting the type of 'os' (line 24)
        os_273370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 39), 'os', False)
        # Obtaining the member 'access' of a type (line 24)
        access_273371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 39), os_273370, 'access')
        # Calling access(args, kwargs) (line 24)
        access_call_result_273375 = invoke(stypy.reporting.localization.Localization(__file__, 24, 39), access_273371, *[fn_273372, mode_273373], **kwargs_273374)
        
        # Applying the binary operator 'and' (line 24)
        result_and_keyword_273376 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 16), 'and', exists_call_result_273369, access_call_result_273375)
        
        
        # Call to isdir(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of 'fn' (line 25)
        fn_273380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 38), 'fn', False)
        # Processing the call keyword arguments (line 25)
        kwargs_273381 = {}
        # Getting the type of 'os' (line 25)
        os_273377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'os', False)
        # Obtaining the member 'path' of a type (line 25)
        path_273378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), os_273377, 'path')
        # Obtaining the member 'isdir' of a type (line 25)
        isdir_273379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 24), path_273378, 'isdir')
        # Calling isdir(args, kwargs) (line 25)
        isdir_call_result_273382 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), isdir_273379, *[fn_273380], **kwargs_273381)
        
        # Applying the 'not' unary operator (line 25)
        result_not__273383 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 20), 'not', isdir_call_result_273382)
        
        # Applying the binary operator 'and' (line 24)
        result_and_keyword_273384 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 16), 'and', result_and_keyword_273376, result_not__273383)
        
        # Assigning a type to the variable 'stypy_return_type' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'stypy_return_type', result_and_keyword_273384)
        
        # ################# End of '_access_check(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_access_check' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_273385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_273385)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_access_check'
        return stypy_return_type_273385

    # Assigning a type to the variable '_access_check' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), '_access_check', _access_check)
    
    
    # Call to dirname(...): (line 30)
    # Processing the call arguments (line 30)
    # Getting the type of 'cmd' (line 30)
    cmd_273389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 23), 'cmd', False)
    # Processing the call keyword arguments (line 30)
    kwargs_273390 = {}
    # Getting the type of 'os' (line 30)
    os_273386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 30)
    path_273387 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 7), os_273386, 'path')
    # Obtaining the member 'dirname' of a type (line 30)
    dirname_273388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 7), path_273387, 'dirname')
    # Calling dirname(args, kwargs) (line 30)
    dirname_call_result_273391 = invoke(stypy.reporting.localization.Localization(__file__, 30, 7), dirname_273388, *[cmd_273389], **kwargs_273390)
    
    # Testing the type of an if condition (line 30)
    if_condition_273392 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 30, 4), dirname_call_result_273391)
    # Assigning a type to the variable 'if_condition_273392' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'if_condition_273392', if_condition_273392)
    # SSA begins for if statement (line 30)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Call to _access_check(...): (line 31)
    # Processing the call arguments (line 31)
    # Getting the type of 'cmd' (line 31)
    cmd_273394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 25), 'cmd', False)
    # Getting the type of 'mode' (line 31)
    mode_273395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 30), 'mode', False)
    # Processing the call keyword arguments (line 31)
    kwargs_273396 = {}
    # Getting the type of '_access_check' (line 31)
    _access_check_273393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 11), '_access_check', False)
    # Calling _access_check(args, kwargs) (line 31)
    _access_check_call_result_273397 = invoke(stypy.reporting.localization.Localization(__file__, 31, 11), _access_check_273393, *[cmd_273394, mode_273395], **kwargs_273396)
    
    # Testing the type of an if condition (line 31)
    if_condition_273398 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 31, 8), _access_check_call_result_273397)
    # Assigning a type to the variable 'if_condition_273398' (line 31)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'if_condition_273398', if_condition_273398)
    # SSA begins for if statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'cmd' (line 32)
    cmd_273399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'cmd')
    # Assigning a type to the variable 'stypy_return_type' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'stypy_return_type', cmd_273399)
    # SSA join for if statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 33)
    None_273400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', None_273400)
    # SSA join for if statement (line 30)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 35)
    # Getting the type of 'path' (line 35)
    path_273401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 7), 'path')
    # Getting the type of 'None' (line 35)
    None_273402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'None')
    
    (may_be_273403, more_types_in_union_273404) = may_be_none(path_273401, None_273402)

    if may_be_273403:

        if more_types_in_union_273404:
            # Runtime conditional SSA (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Call to a Name (line 36):
        
        # Call to get(...): (line 36)
        # Processing the call arguments (line 36)
        str_273408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 30), 'str', 'PATH')
        # Getting the type of 'os' (line 36)
        os_273409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 38), 'os', False)
        # Obtaining the member 'defpath' of a type (line 36)
        defpath_273410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 38), os_273409, 'defpath')
        # Processing the call keyword arguments (line 36)
        kwargs_273411 = {}
        # Getting the type of 'os' (line 36)
        os_273405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'os', False)
        # Obtaining the member 'environ' of a type (line 36)
        environ_273406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), os_273405, 'environ')
        # Obtaining the member 'get' of a type (line 36)
        get_273407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), environ_273406, 'get')
        # Calling get(args, kwargs) (line 36)
        get_call_result_273412 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), get_273407, *[str_273408, defpath_273410], **kwargs_273411)
        
        # Assigning a type to the variable 'path' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'path', get_call_result_273412)

        if more_types_in_union_273404:
            # SSA join for if statement (line 35)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Getting the type of 'path' (line 37)
    path_273413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'path')
    # Applying the 'not' unary operator (line 37)
    result_not__273414 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), 'not', path_273413)
    
    # Testing the type of an if condition (line 37)
    if_condition_273415 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), result_not__273414)
    # Assigning a type to the variable 'if_condition_273415' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_273415', if_condition_273415)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 38)
    None_273416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', None_273416)
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 39):
    
    # Call to split(...): (line 39)
    # Processing the call arguments (line 39)
    # Getting the type of 'os' (line 39)
    os_273419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 22), 'os', False)
    # Obtaining the member 'pathsep' of a type (line 39)
    pathsep_273420 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 22), os_273419, 'pathsep')
    # Processing the call keyword arguments (line 39)
    kwargs_273421 = {}
    # Getting the type of 'path' (line 39)
    path_273417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 11), 'path', False)
    # Obtaining the member 'split' of a type (line 39)
    split_273418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 11), path_273417, 'split')
    # Calling split(args, kwargs) (line 39)
    split_call_result_273422 = invoke(stypy.reporting.localization.Localization(__file__, 39, 11), split_273418, *[pathsep_273420], **kwargs_273421)
    
    # Assigning a type to the variable 'path' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'path', split_call_result_273422)
    
    
    # Getting the type of 'sys' (line 41)
    sys_273423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'sys')
    # Obtaining the member 'platform' of a type (line 41)
    platform_273424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 7), sys_273423, 'platform')
    str_273425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 23), 'str', 'win32')
    # Applying the binary operator '==' (line 41)
    result_eq_273426 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), '==', platform_273424, str_273425)
    
    # Testing the type of an if condition (line 41)
    if_condition_273427 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_eq_273426)
    # Assigning a type to the variable 'if_condition_273427' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_273427', if_condition_273427)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Getting the type of 'os' (line 43)
    os_273428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'os')
    # Obtaining the member 'curdir' of a type (line 43)
    curdir_273429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 15), os_273428, 'curdir')
    # Getting the type of 'path' (line 43)
    path_273430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 28), 'path')
    # Applying the binary operator 'in' (line 43)
    result_contains_273431 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 15), 'in', curdir_273429, path_273430)
    
    # Applying the 'not' unary operator (line 43)
    result_not__273432 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 11), 'not', result_contains_273431)
    
    # Testing the type of an if condition (line 43)
    if_condition_273433 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), result_not__273432)
    # Assigning a type to the variable 'if_condition_273433' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_273433', if_condition_273433)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to insert(...): (line 44)
    # Processing the call arguments (line 44)
    int_273436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 24), 'int')
    # Getting the type of 'os' (line 44)
    os_273437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 27), 'os', False)
    # Obtaining the member 'curdir' of a type (line 44)
    curdir_273438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 27), os_273437, 'curdir')
    # Processing the call keyword arguments (line 44)
    kwargs_273439 = {}
    # Getting the type of 'path' (line 44)
    path_273434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'path', False)
    # Obtaining the member 'insert' of a type (line 44)
    insert_273435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), path_273434, 'insert')
    # Calling insert(args, kwargs) (line 44)
    insert_call_result_273440 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), insert_273435, *[int_273436, curdir_273438], **kwargs_273439)
    
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 47):
    
    # Call to split(...): (line 47)
    # Processing the call arguments (line 47)
    # Getting the type of 'os' (line 47)
    os_273449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 54), 'os', False)
    # Obtaining the member 'pathsep' of a type (line 47)
    pathsep_273450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 54), os_273449, 'pathsep')
    # Processing the call keyword arguments (line 47)
    kwargs_273451 = {}
    
    # Call to get(...): (line 47)
    # Processing the call arguments (line 47)
    str_273444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 33), 'str', 'PATHEXT')
    str_273445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 44), 'str', '')
    # Processing the call keyword arguments (line 47)
    kwargs_273446 = {}
    # Getting the type of 'os' (line 47)
    os_273441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 18), 'os', False)
    # Obtaining the member 'environ' of a type (line 47)
    environ_273442 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 18), os_273441, 'environ')
    # Obtaining the member 'get' of a type (line 47)
    get_273443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 18), environ_273442, 'get')
    # Calling get(args, kwargs) (line 47)
    get_call_result_273447 = invoke(stypy.reporting.localization.Localization(__file__, 47, 18), get_273443, *[str_273444, str_273445], **kwargs_273446)
    
    # Obtaining the member 'split' of a type (line 47)
    split_273448 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 18), get_call_result_273447, 'split')
    # Calling split(args, kwargs) (line 47)
    split_call_result_273452 = invoke(stypy.reporting.localization.Localization(__file__, 47, 18), split_273448, *[pathsep_273450], **kwargs_273451)
    
    # Assigning a type to the variable 'pathext' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'pathext', split_call_result_273452)
    
    
    # Call to any(...): (line 52)
    # Processing the call arguments (line 52)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 52, 15, True)
    # Calculating comprehension expression
    # Getting the type of 'pathext' (line 52)
    pathext_273465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 60), 'pathext', False)
    comprehension_273466 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 15), pathext_273465)
    # Assigning a type to the variable 'ext' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'ext', comprehension_273466)
    
    # Call to endswith(...): (line 52)
    # Processing the call arguments (line 52)
    
    # Call to lower(...): (line 52)
    # Processing the call keyword arguments (line 52)
    kwargs_273461 = {}
    # Getting the type of 'ext' (line 52)
    ext_273459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 36), 'ext', False)
    # Obtaining the member 'lower' of a type (line 52)
    lower_273460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 36), ext_273459, 'lower')
    # Calling lower(args, kwargs) (line 52)
    lower_call_result_273462 = invoke(stypy.reporting.localization.Localization(__file__, 52, 36), lower_273460, *[], **kwargs_273461)
    
    # Processing the call keyword arguments (line 52)
    kwargs_273463 = {}
    
    # Call to lower(...): (line 52)
    # Processing the call keyword arguments (line 52)
    kwargs_273456 = {}
    # Getting the type of 'cmd' (line 52)
    cmd_273454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 15), 'cmd', False)
    # Obtaining the member 'lower' of a type (line 52)
    lower_273455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), cmd_273454, 'lower')
    # Calling lower(args, kwargs) (line 52)
    lower_call_result_273457 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), lower_273455, *[], **kwargs_273456)
    
    # Obtaining the member 'endswith' of a type (line 52)
    endswith_273458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 15), lower_call_result_273457, 'endswith')
    # Calling endswith(args, kwargs) (line 52)
    endswith_call_result_273464 = invoke(stypy.reporting.localization.Localization(__file__, 52, 15), endswith_273458, *[lower_call_result_273462], **kwargs_273463)
    
    list_273467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 15), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 52, 15), list_273467, endswith_call_result_273464)
    # Processing the call keyword arguments (line 52)
    kwargs_273468 = {}
    # Getting the type of 'any' (line 52)
    any_273453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 11), 'any', False)
    # Calling any(args, kwargs) (line 52)
    any_call_result_273469 = invoke(stypy.reporting.localization.Localization(__file__, 52, 11), any_273453, *[list_273467], **kwargs_273468)
    
    # Testing the type of an if condition (line 52)
    if_condition_273470 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 8), any_call_result_273469)
    # Assigning a type to the variable 'if_condition_273470' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'if_condition_273470', if_condition_273470)
    # SSA begins for if statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 53):
    
    # Obtaining an instance of the builtin type 'list' (line 53)
    list_273471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 53)
    # Adding element type (line 53)
    # Getting the type of 'cmd' (line 53)
    cmd_273472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 21), 'cmd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 20), list_273471, cmd_273472)
    
    # Assigning a type to the variable 'files' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 12), 'files', list_273471)
    # SSA branch for the else part of an if statement (line 52)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a ListComp to a Name (line 55):
    # Calculating list comprehension
    # Calculating comprehension expression
    # Getting the type of 'pathext' (line 55)
    pathext_273476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 42), 'pathext')
    comprehension_273477 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), pathext_273476)
    # Assigning a type to the variable 'ext' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'ext', comprehension_273477)
    # Getting the type of 'cmd' (line 55)
    cmd_273473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'cmd')
    # Getting the type of 'ext' (line 55)
    ext_273474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 27), 'ext')
    # Applying the binary operator '+' (line 55)
    result_add_273475 = python_operator(stypy.reporting.localization.Localization(__file__, 55, 21), '+', cmd_273473, ext_273474)
    
    list_273478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 21), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 55, 21), list_273478, result_add_273475)
    # Assigning a type to the variable 'files' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'files', list_273478)
    # SSA join for if statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 59):
    
    # Obtaining an instance of the builtin type 'list' (line 59)
    list_273479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 59)
    # Adding element type (line 59)
    # Getting the type of 'cmd' (line 59)
    cmd_273480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 17), 'cmd')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 16), list_273479, cmd_273480)
    
    # Assigning a type to the variable 'files' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'files', list_273479)
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 61):
    
    # Call to set(...): (line 61)
    # Processing the call keyword arguments (line 61)
    kwargs_273482 = {}
    # Getting the type of 'set' (line 61)
    set_273481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'set', False)
    # Calling set(args, kwargs) (line 61)
    set_call_result_273483 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), set_273481, *[], **kwargs_273482)
    
    # Assigning a type to the variable 'seen' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'seen', set_call_result_273483)
    
    # Getting the type of 'path' (line 62)
    path_273484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 15), 'path')
    # Testing the type of a for loop iterable (line 62)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 4), path_273484)
    # Getting the type of the for loop variable (line 62)
    for_loop_var_273485 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 4), path_273484)
    # Assigning a type to the variable 'dir' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'dir', for_loop_var_273485)
    # SSA begins for a for statement (line 62)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 63):
    
    # Call to normcase(...): (line 63)
    # Processing the call arguments (line 63)
    # Getting the type of 'dir' (line 63)
    dir_273489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 35), 'dir', False)
    # Processing the call keyword arguments (line 63)
    kwargs_273490 = {}
    # Getting the type of 'os' (line 63)
    os_273486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 63)
    path_273487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 18), os_273486, 'path')
    # Obtaining the member 'normcase' of a type (line 63)
    normcase_273488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 18), path_273487, 'normcase')
    # Calling normcase(args, kwargs) (line 63)
    normcase_call_result_273491 = invoke(stypy.reporting.localization.Localization(__file__, 63, 18), normcase_273488, *[dir_273489], **kwargs_273490)
    
    # Assigning a type to the variable 'normdir' (line 63)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'normdir', normcase_call_result_273491)
    
    
    
    # Getting the type of 'normdir' (line 64)
    normdir_273492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'normdir')
    # Getting the type of 'seen' (line 64)
    seen_273493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'seen')
    # Applying the binary operator 'in' (line 64)
    result_contains_273494 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 15), 'in', normdir_273492, seen_273493)
    
    # Applying the 'not' unary operator (line 64)
    result_not__273495 = python_operator(stypy.reporting.localization.Localization(__file__, 64, 11), 'not', result_contains_273494)
    
    # Testing the type of an if condition (line 64)
    if_condition_273496 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), result_not__273495)
    # Assigning a type to the variable 'if_condition_273496' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_273496', if_condition_273496)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to add(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'normdir' (line 65)
    normdir_273499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'normdir', False)
    # Processing the call keyword arguments (line 65)
    kwargs_273500 = {}
    # Getting the type of 'seen' (line 65)
    seen_273497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'seen', False)
    # Obtaining the member 'add' of a type (line 65)
    add_273498 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 12), seen_273497, 'add')
    # Calling add(args, kwargs) (line 65)
    add_call_result_273501 = invoke(stypy.reporting.localization.Localization(__file__, 65, 12), add_273498, *[normdir_273499], **kwargs_273500)
    
    
    # Getting the type of 'files' (line 66)
    files_273502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 27), 'files')
    # Testing the type of a for loop iterable (line 66)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 66, 12), files_273502)
    # Getting the type of the for loop variable (line 66)
    for_loop_var_273503 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 66, 12), files_273502)
    # Assigning a type to the variable 'thefile' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'thefile', for_loop_var_273503)
    # SSA begins for a for statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 67):
    
    # Call to join(...): (line 67)
    # Processing the call arguments (line 67)
    # Getting the type of 'dir' (line 67)
    dir_273507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 36), 'dir', False)
    # Getting the type of 'thefile' (line 67)
    thefile_273508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 41), 'thefile', False)
    # Processing the call keyword arguments (line 67)
    kwargs_273509 = {}
    # Getting the type of 'os' (line 67)
    os_273504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 67)
    path_273505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 23), os_273504, 'path')
    # Obtaining the member 'join' of a type (line 67)
    join_273506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 23), path_273505, 'join')
    # Calling join(args, kwargs) (line 67)
    join_call_result_273510 = invoke(stypy.reporting.localization.Localization(__file__, 67, 23), join_273506, *[dir_273507, thefile_273508], **kwargs_273509)
    
    # Assigning a type to the variable 'name' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'name', join_call_result_273510)
    
    
    # Call to _access_check(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'name' (line 68)
    name_273512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 33), 'name', False)
    # Getting the type of 'mode' (line 68)
    mode_273513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 39), 'mode', False)
    # Processing the call keyword arguments (line 68)
    kwargs_273514 = {}
    # Getting the type of '_access_check' (line 68)
    _access_check_273511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), '_access_check', False)
    # Calling _access_check(args, kwargs) (line 68)
    _access_check_call_result_273515 = invoke(stypy.reporting.localization.Localization(__file__, 68, 19), _access_check_273511, *[name_273512, mode_273513], **kwargs_273514)
    
    # Testing the type of an if condition (line 68)
    if_condition_273516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 16), _access_check_call_result_273515)
    # Assigning a type to the variable 'if_condition_273516' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'if_condition_273516', if_condition_273516)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'name' (line 69)
    name_273517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'name')
    # Assigning a type to the variable 'stypy_return_type' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 20), 'stypy_return_type', name_273517)
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 70)
    None_273518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', None_273518)
    
    # ################# End of 'which(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'which' in the type store
    # Getting the type of 'stypy_return_type' (line 10)
    stypy_return_type_273519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273519)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'which'
    return stypy_return_type_273519

# Assigning a type to the variable 'which' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'which', which)

@norecursion
def _maybe_view_as_subclass(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_maybe_view_as_subclass'
    module_type_store = module_type_store.open_function_context('_maybe_view_as_subclass', 74, 0, False)
    
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

    
    # Type idiom detected: calculating its left and rigth part (line 75)
    # Getting the type of 'original_array' (line 75)
    original_array_273520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'original_array')
    
    # Call to type(...): (line 75)
    # Processing the call arguments (line 75)
    # Getting the type of 'new_array' (line 75)
    new_array_273522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 40), 'new_array', False)
    # Processing the call keyword arguments (line 75)
    kwargs_273523 = {}
    # Getting the type of 'type' (line 75)
    type_273521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 35), 'type', False)
    # Calling type(args, kwargs) (line 75)
    type_call_result_273524 = invoke(stypy.reporting.localization.Localization(__file__, 75, 35), type_273521, *[new_array_273522], **kwargs_273523)
    
    
    (may_be_273525, more_types_in_union_273526) = may_not_be_type(original_array_273520, type_call_result_273524)

    if may_be_273525:

        if more_types_in_union_273526:
            # Runtime conditional SSA (line 75)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'original_array' (line 75)
        original_array_273527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'original_array')
        # Assigning a type to the variable 'original_array' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 4), 'original_array', remove_type_from_union(original_array_273527, type_call_result_273524))
        
        # Assigning a Call to a Name (line 78):
        
        # Call to view(...): (line 78)
        # Processing the call keyword arguments (line 78)
        
        # Call to type(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'original_array' (line 78)
        original_array_273531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 45), 'original_array', False)
        # Processing the call keyword arguments (line 78)
        kwargs_273532 = {}
        # Getting the type of 'type' (line 78)
        type_273530 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 40), 'type', False)
        # Calling type(args, kwargs) (line 78)
        type_call_result_273533 = invoke(stypy.reporting.localization.Localization(__file__, 78, 40), type_273530, *[original_array_273531], **kwargs_273532)
        
        keyword_273534 = type_call_result_273533
        kwargs_273535 = {'type': keyword_273534}
        # Getting the type of 'new_array' (line 78)
        new_array_273528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'new_array', False)
        # Obtaining the member 'view' of a type (line 78)
        view_273529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), new_array_273528, 'view')
        # Calling view(args, kwargs) (line 78)
        view_call_result_273536 = invoke(stypy.reporting.localization.Localization(__file__, 78, 20), view_273529, *[], **kwargs_273535)
        
        # Assigning a type to the variable 'new_array' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'new_array', view_call_result_273536)
        
        # Getting the type of 'new_array' (line 82)
        new_array_273537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 11), 'new_array')
        # Obtaining the member '__array_finalize__' of a type (line 82)
        array_finalize___273538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 11), new_array_273537, '__array_finalize__')
        # Testing the type of an if condition (line 82)
        if_condition_273539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 8), array_finalize___273538)
        # Assigning a type to the variable 'if_condition_273539' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'if_condition_273539', if_condition_273539)
        # SSA begins for if statement (line 82)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to __array_finalize__(...): (line 83)
        # Processing the call arguments (line 83)
        # Getting the type of 'original_array' (line 83)
        original_array_273542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 41), 'original_array', False)
        # Processing the call keyword arguments (line 83)
        kwargs_273543 = {}
        # Getting the type of 'new_array' (line 83)
        new_array_273540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'new_array', False)
        # Obtaining the member '__array_finalize__' of a type (line 83)
        array_finalize___273541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), new_array_273540, '__array_finalize__')
        # Calling __array_finalize__(args, kwargs) (line 83)
        array_finalize___call_result_273544 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), array_finalize___273541, *[original_array_273542], **kwargs_273543)
        
        # SSA join for if statement (line 82)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_273526:
            # SSA join for if statement (line 75)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'new_array' (line 84)
    new_array_273545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'new_array')
    # Assigning a type to the variable 'stypy_return_type' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type', new_array_273545)
    
    # ################# End of '_maybe_view_as_subclass(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_maybe_view_as_subclass' in the type store
    # Getting the type of 'stypy_return_type' (line 74)
    stypy_return_type_273546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273546)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_maybe_view_as_subclass'
    return stypy_return_type_273546

# Assigning a type to the variable '_maybe_view_as_subclass' (line 74)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), '_maybe_view_as_subclass', _maybe_view_as_subclass)

@norecursion
def _broadcast_to(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_broadcast_to'
    module_type_store = module_type_store.open_function_context('_broadcast_to', 88, 0, False)
    
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

    
    # Assigning a IfExp to a Name (line 89):
    
    
    # Call to iterable(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'shape' (line 89)
    shape_273549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 40), 'shape', False)
    # Processing the call keyword arguments (line 89)
    kwargs_273550 = {}
    # Getting the type of 'np' (line 89)
    np_273547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 28), 'np', False)
    # Obtaining the member 'iterable' of a type (line 89)
    iterable_273548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 28), np_273547, 'iterable')
    # Calling iterable(args, kwargs) (line 89)
    iterable_call_result_273551 = invoke(stypy.reporting.localization.Localization(__file__, 89, 28), iterable_273548, *[shape_273549], **kwargs_273550)
    
    # Testing the type of an if expression (line 89)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 12), iterable_call_result_273551)
    # SSA begins for if expression (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Call to tuple(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'shape' (line 89)
    shape_273553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 18), 'shape', False)
    # Processing the call keyword arguments (line 89)
    kwargs_273554 = {}
    # Getting the type of 'tuple' (line 89)
    tuple_273552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 12), 'tuple', False)
    # Calling tuple(args, kwargs) (line 89)
    tuple_call_result_273555 = invoke(stypy.reporting.localization.Localization(__file__, 89, 12), tuple_273552, *[shape_273553], **kwargs_273554)
    
    # SSA branch for the else part of an if expression (line 89)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining an instance of the builtin type 'tuple' (line 89)
    tuple_273556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 89)
    # Adding element type (line 89)
    # Getting the type of 'shape' (line 89)
    shape_273557 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 53), 'shape')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 89, 53), tuple_273556, shape_273557)
    
    # SSA join for if expression (line 89)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_273558 = union_type.UnionType.add(tuple_call_result_273555, tuple_273556)
    
    # Assigning a type to the variable 'shape' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'shape', if_exp_273558)
    
    # Assigning a Call to a Name (line 90):
    
    # Call to array(...): (line 90)
    # Processing the call arguments (line 90)
    # Getting the type of 'array' (line 90)
    array_273561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'array', False)
    # Processing the call keyword arguments (line 90)
    # Getting the type of 'False' (line 90)
    False_273562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 33), 'False', False)
    keyword_273563 = False_273562
    # Getting the type of 'subok' (line 90)
    subok_273564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 46), 'subok', False)
    keyword_273565 = subok_273564
    kwargs_273566 = {'subok': keyword_273565, 'copy': keyword_273563}
    # Getting the type of 'np' (line 90)
    np_273559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'np', False)
    # Obtaining the member 'array' of a type (line 90)
    array_273560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), np_273559, 'array')
    # Calling array(args, kwargs) (line 90)
    array_call_result_273567 = invoke(stypy.reporting.localization.Localization(__file__, 90, 12), array_273560, *[array_273561], **kwargs_273566)
    
    # Assigning a type to the variable 'array' (line 90)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 4), 'array', array_call_result_273567)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'shape' (line 91)
    shape_273568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 11), 'shape')
    # Applying the 'not' unary operator (line 91)
    result_not__273569 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 7), 'not', shape_273568)
    
    # Getting the type of 'array' (line 91)
    array_273570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'array')
    # Obtaining the member 'shape' of a type (line 91)
    shape_273571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 21), array_273570, 'shape')
    # Applying the binary operator 'and' (line 91)
    result_and_keyword_273572 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 7), 'and', result_not__273569, shape_273571)
    
    # Testing the type of an if condition (line 91)
    if_condition_273573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 91, 4), result_and_keyword_273572)
    # Assigning a type to the variable 'if_condition_273573' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'if_condition_273573', if_condition_273573)
    # SSA begins for if statement (line 91)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 92)
    # Processing the call arguments (line 92)
    str_273575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 25), 'str', 'cannot broadcast a non-scalar to a scalar array')
    # Processing the call keyword arguments (line 92)
    kwargs_273576 = {}
    # Getting the type of 'ValueError' (line 92)
    ValueError_273574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 92)
    ValueError_call_result_273577 = invoke(stypy.reporting.localization.Localization(__file__, 92, 14), ValueError_273574, *[str_273575], **kwargs_273576)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 92, 8), ValueError_call_result_273577, 'raise parameter', BaseException)
    # SSA join for if statement (line 91)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to any(...): (line 93)
    # Processing the call arguments (line 93)
    # Calculating generator expression
    module_type_store = module_type_store.open_function_context('list comprehension expression', 93, 11, True)
    # Calculating comprehension expression
    # Getting the type of 'shape' (line 93)
    shape_273582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 32), 'shape', False)
    comprehension_273583 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 11), shape_273582)
    # Assigning a type to the variable 'size' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'size', comprehension_273583)
    
    # Getting the type of 'size' (line 93)
    size_273579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'size', False)
    int_273580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 18), 'int')
    # Applying the binary operator '<' (line 93)
    result_lt_273581 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 11), '<', size_273579, int_273580)
    
    list_273584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 11), 'list')
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 11), list_273584, result_lt_273581)
    # Processing the call keyword arguments (line 93)
    kwargs_273585 = {}
    # Getting the type of 'any' (line 93)
    any_273578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 7), 'any', False)
    # Calling any(args, kwargs) (line 93)
    any_call_result_273586 = invoke(stypy.reporting.localization.Localization(__file__, 93, 7), any_273578, *[list_273584], **kwargs_273585)
    
    # Testing the type of an if condition (line 93)
    if_condition_273587 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 4), any_call_result_273586)
    # Assigning a type to the variable 'if_condition_273587' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'if_condition_273587', if_condition_273587)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to ValueError(...): (line 94)
    # Processing the call arguments (line 94)
    str_273589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 94, 25), 'str', 'all elements of broadcast shape must be non-negative')
    # Processing the call keyword arguments (line 94)
    kwargs_273590 = {}
    # Getting the type of 'ValueError' (line 94)
    ValueError_273588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 94)
    ValueError_call_result_273591 = invoke(stypy.reporting.localization.Localization(__file__, 94, 14), ValueError_273588, *[str_273589], **kwargs_273590)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 94, 8), ValueError_call_result_273591, 'raise parameter', BaseException)
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BoolOp to a Name (line 96):
    
    # Evaluating a boolean operation
    
    # Getting the type of 'readonly' (line 96)
    readonly_273592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'readonly')
    # Applying the 'not' unary operator (line 96)
    result_not__273593 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 22), 'not', readonly_273592)
    
    # Getting the type of 'array' (line 96)
    array_273594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 39), 'array')
    # Obtaining the member 'flags' of a type (line 96)
    flags_273595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 39), array_273594, 'flags')
    # Obtaining the member 'writeable' of a type (line 96)
    writeable_273596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 39), flags_273595, 'writeable')
    # Applying the binary operator 'and' (line 96)
    result_and_keyword_273597 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 22), 'and', result_not__273593, writeable_273596)
    
    # Assigning a type to the variable 'needs_writeable' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'needs_writeable', result_and_keyword_273597)
    
    # Assigning a IfExp to a Name (line 97):
    
    # Getting the type of 'needs_writeable' (line 97)
    needs_writeable_273598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 30), 'needs_writeable')
    # Testing the type of an if expression (line 97)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 13), needs_writeable_273598)
    # SSA begins for if expression (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_273599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 13), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    # Adding element type (line 97)
    str_273600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 14), 'str', 'reduce_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 97, 13), list_273599, str_273600)
    
    # SSA branch for the else part of an if expression (line 97)
    module_type_store.open_ssa_branch('if expression else')
    
    # Obtaining an instance of the builtin type 'list' (line 97)
    list_273601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 51), 'list')
    # Adding type elements to the builtin type 'list' instance (line 97)
    
    # SSA join for if expression (line 97)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_273602 = union_type.UnionType.add(list_273599, list_273601)
    
    # Assigning a type to the variable 'extras' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'extras', if_exp_273602)
    
    # Assigning a IfExp to a Name (line 98):
    
    # Getting the type of 'needs_writeable' (line 98)
    needs_writeable_273603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 29), 'needs_writeable')
    # Testing the type of an if expression (line 98)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 14), needs_writeable_273603)
    # SSA begins for if expression (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if expression')
    str_273604 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 14), 'str', 'readwrite')
    # SSA branch for the else part of an if expression (line 98)
    module_type_store.open_ssa_branch('if expression else')
    str_273605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 50), 'str', 'readonly')
    # SSA join for if expression (line 98)
    module_type_store = module_type_store.join_ssa_context()
    if_exp_273606 = union_type.UnionType.add(str_273604, str_273605)
    
    # Assigning a type to the variable 'op_flag' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'op_flag', if_exp_273606)
    
    # Assigning a Subscript to a Name (line 99):
    
    # Obtaining the type of the subscript
    int_273607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 64), 'int')
    
    # Call to nditer(...): (line 99)
    # Processing the call arguments (line 99)
    
    # Obtaining an instance of the builtin type 'tuple' (line 100)
    tuple_273610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 9), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 100)
    # Adding element type (line 100)
    # Getting the type of 'array' (line 100)
    array_273611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'array', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 9), tuple_273610, array_273611)
    
    # Processing the call keyword arguments (line 99)
    
    # Obtaining an instance of the builtin type 'list' (line 100)
    list_273612 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 24), 'list')
    # Adding type elements to the builtin type 'list' instance (line 100)
    # Adding element type (line 100)
    str_273613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 25), 'str', 'multi_index')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 24), list_273612, str_273613)
    # Adding element type (line 100)
    str_273614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 40), 'str', 'refs_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 24), list_273612, str_273614)
    # Adding element type (line 100)
    str_273615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 51), 'str', 'zerosize_ok')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 100, 24), list_273612, str_273615)
    
    # Getting the type of 'extras' (line 100)
    extras_273616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 68), 'extras', False)
    # Applying the binary operator '+' (line 100)
    result_add_273617 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 24), '+', list_273612, extras_273616)
    
    keyword_273618 = result_add_273617
    
    # Obtaining an instance of the builtin type 'list' (line 101)
    list_273619 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 101)
    # Adding element type (line 101)
    # Getting the type of 'op_flag' (line 101)
    op_flag_273620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'op_flag', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 101, 17), list_273619, op_flag_273620)
    
    keyword_273621 = list_273619
    # Getting the type of 'shape' (line 101)
    shape_273622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 38), 'shape', False)
    keyword_273623 = shape_273622
    str_273624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 51), 'str', 'C')
    keyword_273625 = str_273624
    kwargs_273626 = {'itershape': keyword_273623, 'op_flags': keyword_273621, 'flags': keyword_273618, 'order': keyword_273625}
    # Getting the type of 'np' (line 99)
    np_273608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'np', False)
    # Obtaining the member 'nditer' of a type (line 99)
    nditer_273609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), np_273608, 'nditer')
    # Calling nditer(args, kwargs) (line 99)
    nditer_call_result_273627 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), nditer_273609, *[tuple_273610], **kwargs_273626)
    
    # Obtaining the member 'itviews' of a type (line 99)
    itviews_273628 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), nditer_call_result_273627, 'itviews')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___273629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), itviews_273628, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_273630 = invoke(stypy.reporting.localization.Localization(__file__, 99, 16), getitem___273629, int_273607)
    
    # Assigning a type to the variable 'broadcast' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'broadcast', subscript_call_result_273630)
    
    # Assigning a Call to a Name (line 102):
    
    # Call to _maybe_view_as_subclass(...): (line 102)
    # Processing the call arguments (line 102)
    # Getting the type of 'array' (line 102)
    array_273632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 37), 'array', False)
    # Getting the type of 'broadcast' (line 102)
    broadcast_273633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 44), 'broadcast', False)
    # Processing the call keyword arguments (line 102)
    kwargs_273634 = {}
    # Getting the type of '_maybe_view_as_subclass' (line 102)
    _maybe_view_as_subclass_273631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 13), '_maybe_view_as_subclass', False)
    # Calling _maybe_view_as_subclass(args, kwargs) (line 102)
    _maybe_view_as_subclass_call_result_273635 = invoke(stypy.reporting.localization.Localization(__file__, 102, 13), _maybe_view_as_subclass_273631, *[array_273632, broadcast_273633], **kwargs_273634)
    
    # Assigning a type to the variable 'result' (line 102)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'result', _maybe_view_as_subclass_call_result_273635)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'needs_writeable' (line 103)
    needs_writeable_273636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 7), 'needs_writeable')
    
    # Getting the type of 'result' (line 103)
    result_273637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 31), 'result')
    # Obtaining the member 'flags' of a type (line 103)
    flags_273638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 31), result_273637, 'flags')
    # Obtaining the member 'writeable' of a type (line 103)
    writeable_273639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 31), flags_273638, 'writeable')
    # Applying the 'not' unary operator (line 103)
    result_not__273640 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 27), 'not', writeable_273639)
    
    # Applying the binary operator 'and' (line 103)
    result_and_keyword_273641 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), 'and', needs_writeable_273636, result_not__273640)
    
    # Testing the type of an if condition (line 103)
    if_condition_273642 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), result_and_keyword_273641)
    # Assigning a type to the variable 'if_condition_273642' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_273642', if_condition_273642)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Attribute (line 104):
    # Getting the type of 'True' (line 104)
    True_273643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 33), 'True')
    # Getting the type of 'result' (line 104)
    result_273644 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'result')
    # Obtaining the member 'flags' of a type (line 104)
    flags_273645 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), result_273644, 'flags')
    # Setting the type of the member 'writeable' of a type (line 104)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 8), flags_273645, 'writeable', True_273643)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'result' (line 105)
    result_273646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 11), 'result')
    # Assigning a type to the variable 'stypy_return_type' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'stypy_return_type', result_273646)
    
    # ################# End of '_broadcast_to(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_broadcast_to' in the type store
    # Getting the type of 'stypy_return_type' (line 88)
    stypy_return_type_273647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273647)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_broadcast_to'
    return stypy_return_type_273647

# Assigning a type to the variable '_broadcast_to' (line 88)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), '_broadcast_to', _broadcast_to)

@norecursion
def broadcast_to(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'False' (line 109)
    False_273648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 37), 'False')
    defaults = [False_273648]
    # Create a new context for function 'broadcast_to'
    module_type_store = module_type_store.open_function_context('broadcast_to', 109, 0, False)
    
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

    str_273649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, (-1)), 'str', "Broadcast an array to a new shape.\n\n    Parameters\n    ----------\n    array : array_like\n        The array to broadcast.\n    shape : tuple\n        The shape of the desired array.\n    subok : bool, optional\n        If True, then sub-classes will be passed-through, otherwise\n        the returned array will be forced to be a base-class array (default).\n\n    Returns\n    -------\n    broadcast : array\n        A readonly view on the original array with the given shape. It is\n        typically not contiguous. Furthermore, more than one element of a\n        broadcasted array may refer to a single memory location.\n\n    Raises\n    ------\n    ValueError\n        If the array is not compatible with the new shape according to NumPy's\n        broadcasting rules.\n\n    Notes\n    -----\n    .. versionadded:: 1.10.0\n\n    Examples\n    --------\n    >>> x = np.array([1, 2, 3])\n    >>> np.broadcast_to(x, (3, 3))\n    array([[1, 2, 3],\n           [1, 2, 3],\n           [1, 2, 3]])\n    ")
    
    # Call to _broadcast_to(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'array' (line 147)
    array_273651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'array', False)
    # Getting the type of 'shape' (line 147)
    shape_273652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 32), 'shape', False)
    # Processing the call keyword arguments (line 147)
    # Getting the type of 'subok' (line 147)
    subok_273653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 45), 'subok', False)
    keyword_273654 = subok_273653
    # Getting the type of 'True' (line 147)
    True_273655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 61), 'True', False)
    keyword_273656 = True_273655
    kwargs_273657 = {'subok': keyword_273654, 'readonly': keyword_273656}
    # Getting the type of '_broadcast_to' (line 147)
    _broadcast_to_273650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), '_broadcast_to', False)
    # Calling _broadcast_to(args, kwargs) (line 147)
    _broadcast_to_call_result_273658 = invoke(stypy.reporting.localization.Localization(__file__, 147, 11), _broadcast_to_273650, *[array_273651, shape_273652], **kwargs_273657)
    
    # Assigning a type to the variable 'stypy_return_type' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 4), 'stypy_return_type', _broadcast_to_call_result_273658)
    
    # ################# End of 'broadcast_to(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'broadcast_to' in the type store
    # Getting the type of 'stypy_return_type' (line 109)
    stypy_return_type_273659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_273659)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'broadcast_to'
    return stypy_return_type_273659

# Assigning a type to the variable 'broadcast_to' (line 109)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 0), 'broadcast_to', broadcast_to)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
