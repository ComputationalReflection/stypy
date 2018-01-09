
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from distutils.core import Extension as _Extension
2: from distutils.core import Distribution as _Distribution
3: 
4: def _get_unpatched(cls):
5:     '''Protect against re-patching the distutils if reloaded
6: 
7:     Also ensures that no other distutils extension monkeypatched the distutils
8:     first.
9:     '''
10:     while cls.__module__.startswith('setuptools'):
11:         cls, = cls.__bases__
12:     if not cls.__module__.startswith('distutils'):
13:         raise AssertionError(
14:             "distutils has already been patched by %r" % cls
15:         )
16:     return cls
17: 
18: _Distribution = _get_unpatched(_Distribution)
19: _Extension = _get_unpatched(_Extension)
20: 
21: try:
22:     from Pyrex.Distutils.build_ext import build_ext
23: except ImportError:
24:     have_pyrex = False
25: else:
26:     have_pyrex = True
27: 
28: 
29: class Extension(_Extension):
30:     '''Extension that uses '.c' files in place of '.pyx' files'''
31: 
32:     if not have_pyrex:
33:         # convert .pyx extensions to .c
34:         def __init__(self,*args,**kw):
35:             _Extension.__init__(self,*args,**kw)
36:             sources = []
37:             for s in self.sources:
38:                 if s.endswith('.pyx'):
39:                     sources.append(s[:-3]+'c')
40:                 else:
41:                     sources.append(s)
42:             self.sources = sources
43: 
44: class Library(Extension):
45:     '''Just like a regular Extension, but built as a library instead'''
46: 
47: import sys, distutils.core, distutils.extension
48: distutils.core.Extension = Extension
49: distutils.extension.Extension = Extension
50: if 'distutils.command.build_ext' in sys.modules:
51:     sys.modules['distutils.command.build_ext'].Extension = Extension
52: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from distutils.core import _Extension' statement (line 1)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28300 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'distutils.core')

if (type(import_28300) is not StypyTypeError):

    if (import_28300 != 'pyd_module'):
        __import__(import_28300)
        sys_modules_28301 = sys.modules[import_28300]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'distutils.core', sys_modules_28301.module_type_store, module_type_store, ['Extension'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_28301, sys_modules_28301.module_type_store, module_type_store)
    else:
        from distutils.core import Extension as _Extension

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'distutils.core', None, module_type_store, ['Extension'], [_Extension])

else:
    # Assigning a type to the variable 'distutils.core' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'distutils.core', import_28300)

# Adding an alias
module_type_store.add_alias('_Extension', 'Extension')
remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'from distutils.core import _Distribution' statement (line 2)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28302 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'distutils.core')

if (type(import_28302) is not StypyTypeError):

    if (import_28302 != 'pyd_module'):
        __import__(import_28302)
        sys_modules_28303 = sys.modules[import_28302]
        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'distutils.core', sys_modules_28303.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 2, 0), __file__, sys_modules_28303, sys_modules_28303.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution as _Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'distutils.core', None, module_type_store, ['Distribution'], [_Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 2)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 2, 0), 'distutils.core', import_28302)

# Adding an alias
module_type_store.add_alias('_Distribution', 'Distribution')
remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


@norecursion
def _get_unpatched(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_unpatched'
    module_type_store = module_type_store.open_function_context('_get_unpatched', 4, 0, False)
    
    # Passed parameters checking function
    _get_unpatched.stypy_localization = localization
    _get_unpatched.stypy_type_of_self = None
    _get_unpatched.stypy_type_store = module_type_store
    _get_unpatched.stypy_function_name = '_get_unpatched'
    _get_unpatched.stypy_param_names_list = ['cls']
    _get_unpatched.stypy_varargs_param_name = None
    _get_unpatched.stypy_kwargs_param_name = None
    _get_unpatched.stypy_call_defaults = defaults
    _get_unpatched.stypy_call_varargs = varargs
    _get_unpatched.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_unpatched', ['cls'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_unpatched', localization, ['cls'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_unpatched(...)' code ##################

    str_28304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', 'Protect against re-patching the distutils if reloaded\n\n    Also ensures that no other distutils extension monkeypatched the distutils\n    first.\n    ')
    
    
    # Call to startswith(...): (line 10)
    # Processing the call arguments (line 10)
    str_28308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 36), 'str', 'setuptools')
    # Processing the call keyword arguments (line 10)
    kwargs_28309 = {}
    # Getting the type of 'cls' (line 10)
    cls_28305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 10), 'cls', False)
    # Obtaining the member '__module__' of a type (line 10)
    module___28306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), cls_28305, '__module__')
    # Obtaining the member 'startswith' of a type (line 10)
    startswith_28307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 10), module___28306, 'startswith')
    # Calling startswith(args, kwargs) (line 10)
    startswith_call_result_28310 = invoke(stypy.reporting.localization.Localization(__file__, 10, 10), startswith_28307, *[str_28308], **kwargs_28309)
    
    # Testing the type of an if condition (line 10)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 4), startswith_call_result_28310)
    # SSA begins for while statement (line 10)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Attribute to a Tuple (line 11):
    
    # Assigning a Subscript to a Name (line 11):
    
    # Obtaining the type of the subscript
    int_28311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 8), 'int')
    # Getting the type of 'cls' (line 11)
    cls_28312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 15), 'cls')
    # Obtaining the member '__bases__' of a type (line 11)
    bases___28313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 15), cls_28312, '__bases__')
    # Obtaining the member '__getitem__' of a type (line 11)
    getitem___28314 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 8), bases___28313, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 11)
    subscript_call_result_28315 = invoke(stypy.reporting.localization.Localization(__file__, 11, 8), getitem___28314, int_28311)
    
    # Assigning a type to the variable 'tuple_var_assignment_28299' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'tuple_var_assignment_28299', subscript_call_result_28315)
    
    # Assigning a Name to a Name (line 11):
    # Getting the type of 'tuple_var_assignment_28299' (line 11)
    tuple_var_assignment_28299_28316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'tuple_var_assignment_28299')
    # Assigning a type to the variable 'cls' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 8), 'cls', tuple_var_assignment_28299_28316)
    # SSA join for while statement (line 10)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to startswith(...): (line 12)
    # Processing the call arguments (line 12)
    str_28320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 37), 'str', 'distutils')
    # Processing the call keyword arguments (line 12)
    kwargs_28321 = {}
    # Getting the type of 'cls' (line 12)
    cls_28317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 11), 'cls', False)
    # Obtaining the member '__module__' of a type (line 12)
    module___28318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 11), cls_28317, '__module__')
    # Obtaining the member 'startswith' of a type (line 12)
    startswith_28319 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 11), module___28318, 'startswith')
    # Calling startswith(args, kwargs) (line 12)
    startswith_call_result_28322 = invoke(stypy.reporting.localization.Localization(__file__, 12, 11), startswith_28319, *[str_28320], **kwargs_28321)
    
    # Applying the 'not' unary operator (line 12)
    result_not__28323 = python_operator(stypy.reporting.localization.Localization(__file__, 12, 7), 'not', startswith_call_result_28322)
    
    # Testing the type of an if condition (line 12)
    if_condition_28324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 12, 4), result_not__28323)
    # Assigning a type to the variable 'if_condition_28324' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'if_condition_28324', if_condition_28324)
    # SSA begins for if statement (line 12)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to AssertionError(...): (line 13)
    # Processing the call arguments (line 13)
    str_28326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 12), 'str', 'distutils has already been patched by %r')
    # Getting the type of 'cls' (line 14)
    cls_28327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 57), 'cls', False)
    # Applying the binary operator '%' (line 14)
    result_mod_28328 = python_operator(stypy.reporting.localization.Localization(__file__, 14, 12), '%', str_28326, cls_28327)
    
    # Processing the call keyword arguments (line 13)
    kwargs_28329 = {}
    # Getting the type of 'AssertionError' (line 13)
    AssertionError_28325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 14), 'AssertionError', False)
    # Calling AssertionError(args, kwargs) (line 13)
    AssertionError_call_result_28330 = invoke(stypy.reporting.localization.Localization(__file__, 13, 14), AssertionError_28325, *[result_mod_28328], **kwargs_28329)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 13, 8), AssertionError_call_result_28330, 'raise parameter', BaseException)
    # SSA join for if statement (line 12)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'cls' (line 16)
    cls_28331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 11), 'cls')
    # Assigning a type to the variable 'stypy_return_type' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'stypy_return_type', cls_28331)
    
    # ################# End of '_get_unpatched(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_unpatched' in the type store
    # Getting the type of 'stypy_return_type' (line 4)
    stypy_return_type_28332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_unpatched'
    return stypy_return_type_28332

# Assigning a type to the variable '_get_unpatched' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), '_get_unpatched', _get_unpatched)

# Assigning a Call to a Name (line 18):

# Assigning a Call to a Name (line 18):

# Call to _get_unpatched(...): (line 18)
# Processing the call arguments (line 18)
# Getting the type of '_Distribution' (line 18)
_Distribution_28334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 31), '_Distribution', False)
# Processing the call keyword arguments (line 18)
kwargs_28335 = {}
# Getting the type of '_get_unpatched' (line 18)
_get_unpatched_28333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 16), '_get_unpatched', False)
# Calling _get_unpatched(args, kwargs) (line 18)
_get_unpatched_call_result_28336 = invoke(stypy.reporting.localization.Localization(__file__, 18, 16), _get_unpatched_28333, *[_Distribution_28334], **kwargs_28335)

# Assigning a type to the variable '_Distribution' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '_Distribution', _get_unpatched_call_result_28336)

# Assigning a Call to a Name (line 19):

# Assigning a Call to a Name (line 19):

# Call to _get_unpatched(...): (line 19)
# Processing the call arguments (line 19)
# Getting the type of '_Extension' (line 19)
_Extension_28338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 28), '_Extension', False)
# Processing the call keyword arguments (line 19)
kwargs_28339 = {}
# Getting the type of '_get_unpatched' (line 19)
_get_unpatched_28337 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 13), '_get_unpatched', False)
# Calling _get_unpatched(args, kwargs) (line 19)
_get_unpatched_call_result_28340 = invoke(stypy.reporting.localization.Localization(__file__, 19, 13), _get_unpatched_28337, *[_Extension_28338], **kwargs_28339)

# Assigning a type to the variable '_Extension' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), '_Extension', _get_unpatched_call_result_28340)


# SSA begins for try-except statement (line 21)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 4))

# 'from Pyrex.Distutils.build_ext import build_ext' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28341 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'Pyrex.Distutils.build_ext')

if (type(import_28341) is not StypyTypeError):

    if (import_28341 != 'pyd_module'):
        __import__(import_28341)
        sys_modules_28342 = sys.modules[import_28341]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'Pyrex.Distutils.build_ext', sys_modules_28342.module_type_store, module_type_store, ['build_ext'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 4), __file__, sys_modules_28342, sys_modules_28342.module_type_store, module_type_store)
    else:
        from Pyrex.Distutils.build_ext import build_ext

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 4), 'Pyrex.Distutils.build_ext', None, module_type_store, ['build_ext'], [build_ext])

else:
    # Assigning a type to the variable 'Pyrex.Distutils.build_ext' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'Pyrex.Distutils.build_ext', import_28341)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

# SSA branch for the except part of a try statement (line 21)
# SSA branch for the except 'ImportError' branch of a try statement (line 21)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 24):

# Assigning a Name to a Name (line 24):
# Getting the type of 'False' (line 24)
False_28343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 17), 'False')
# Assigning a type to the variable 'have_pyrex' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'have_pyrex', False_28343)
# SSA branch for the else branch of a try statement (line 21)
module_type_store.open_ssa_branch('except else')

# Assigning a Name to a Name (line 26):

# Assigning a Name to a Name (line 26):
# Getting the type of 'True' (line 26)
True_28344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 17), 'True')
# Assigning a type to the variable 'have_pyrex' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'have_pyrex', True_28344)
# SSA join for try-except statement (line 21)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'Extension' class
# Getting the type of '_Extension' (line 29)
_Extension_28345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 16), '_Extension')

class Extension(_Extension_28345, ):
    str_28346 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 4), 'str', "Extension that uses '.c' files in place of '.pyx' files")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Extension.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Extension' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'Extension', Extension)


# Getting the type of 'have_pyrex' (line 32)
have_pyrex_28347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 11), 'have_pyrex')
# Applying the 'not' unary operator (line 32)
result_not__28348 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 7), 'not', have_pyrex_28347)

# Testing the type of an if condition (line 32)
if_condition_28349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 4), result_not__28348)
# Assigning a type to the variable 'if_condition_28349' (line 32)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'if_condition_28349', if_condition_28349)
# SSA begins for if statement (line 32)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

@norecursion
def __init__(type_of_self, localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '__init__'
    module_type_store = module_type_store.open_function_context('__init__', 34, 8, False)
    
    # Passed parameters checking function
    arguments = process_argument_values(localization, None, module_type_store, '__init__', ['self'], 'args', 'kw', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return

    # Initialize method data
    init_call_information(module_type_store, '__init__', localization, ['self'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '__init__(...)' code ##################

    
    # Call to __init__(...): (line 35)
    # Processing the call arguments (line 35)
    # Getting the type of 'self' (line 35)
    self_28352 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 32), 'self', False)
    # Getting the type of 'args' (line 35)
    args_28353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 38), 'args', False)
    # Processing the call keyword arguments (line 35)
    # Getting the type of 'kw' (line 35)
    kw_28354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 45), 'kw', False)
    kwargs_28355 = {'kw_28354': kw_28354}
    # Getting the type of '_Extension' (line 35)
    _Extension_28350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), '_Extension', False)
    # Obtaining the member '__init__' of a type (line 35)
    init___28351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), _Extension_28350, '__init__')
    # Calling __init__(args, kwargs) (line 35)
    init___call_result_28356 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), init___28351, *[self_28352, args_28353], **kwargs_28355)
    
    
    # Assigning a List to a Name (line 36):
    
    # Assigning a List to a Name (line 36):
    
    # Obtaining an instance of the builtin type 'list' (line 36)
    list_28357 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 36)
    
    # Assigning a type to the variable 'sources' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'sources', list_28357)
    
    # Getting the type of 'self' (line 37)
    self_28358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 21), 'self')
    # Obtaining the member 'sources' of a type (line 37)
    sources_28359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 21), self_28358, 'sources')
    # Testing the type of a for loop iterable (line 37)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 37, 12), sources_28359)
    # Getting the type of the for loop variable (line 37)
    for_loop_var_28360 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 37, 12), sources_28359)
    # Assigning a type to the variable 's' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 's', for_loop_var_28360)
    # SSA begins for a for statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to endswith(...): (line 38)
    # Processing the call arguments (line 38)
    str_28363 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 30), 'str', '.pyx')
    # Processing the call keyword arguments (line 38)
    kwargs_28364 = {}
    # Getting the type of 's' (line 38)
    s_28361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 19), 's', False)
    # Obtaining the member 'endswith' of a type (line 38)
    endswith_28362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 19), s_28361, 'endswith')
    # Calling endswith(args, kwargs) (line 38)
    endswith_call_result_28365 = invoke(stypy.reporting.localization.Localization(__file__, 38, 19), endswith_28362, *[str_28363], **kwargs_28364)
    
    # Testing the type of an if condition (line 38)
    if_condition_28366 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 16), endswith_call_result_28365)
    # Assigning a type to the variable 'if_condition_28366' (line 38)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'if_condition_28366', if_condition_28366)
    # SSA begins for if statement (line 38)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 39)
    # Processing the call arguments (line 39)
    
    # Obtaining the type of the subscript
    int_28369 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 38), 'int')
    slice_28370 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 39, 35), None, int_28369, None)
    # Getting the type of 's' (line 39)
    s_28371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 35), 's', False)
    # Obtaining the member '__getitem__' of a type (line 39)
    getitem___28372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 35), s_28371, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 39)
    subscript_call_result_28373 = invoke(stypy.reporting.localization.Localization(__file__, 39, 35), getitem___28372, slice_28370)
    
    str_28374 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 42), 'str', 'c')
    # Applying the binary operator '+' (line 39)
    result_add_28375 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 35), '+', subscript_call_result_28373, str_28374)
    
    # Processing the call keyword arguments (line 39)
    kwargs_28376 = {}
    # Getting the type of 'sources' (line 39)
    sources_28367 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 20), 'sources', False)
    # Obtaining the member 'append' of a type (line 39)
    append_28368 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 20), sources_28367, 'append')
    # Calling append(args, kwargs) (line 39)
    append_call_result_28377 = invoke(stypy.reporting.localization.Localization(__file__, 39, 20), append_28368, *[result_add_28375], **kwargs_28376)
    
    # SSA branch for the else part of an if statement (line 38)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 41)
    # Processing the call arguments (line 41)
    # Getting the type of 's' (line 41)
    s_28380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 's', False)
    # Processing the call keyword arguments (line 41)
    kwargs_28381 = {}
    # Getting the type of 'sources' (line 41)
    sources_28378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'sources', False)
    # Obtaining the member 'append' of a type (line 41)
    append_28379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 20), sources_28378, 'append')
    # Calling append(args, kwargs) (line 41)
    append_call_result_28382 = invoke(stypy.reporting.localization.Localization(__file__, 41, 20), append_28379, *[s_28380], **kwargs_28381)
    
    # SSA join for if statement (line 38)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Name to a Attribute (line 42):
    
    # Assigning a Name to a Attribute (line 42):
    # Getting the type of 'sources' (line 42)
    sources_28383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 27), 'sources')
    # Getting the type of 'self' (line 42)
    self_28384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'self')
    # Setting the type of the member 'sources' of a type (line 42)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), self_28384, 'sources', sources_28383)
    
    # ################# End of '__init__(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()

# Assigning a type to the variable '__init__' (line 34)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), '__init__', __init__)
# SSA join for if statement (line 32)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'Library' class
# Getting the type of 'Extension' (line 44)
Extension_28385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'Extension')

class Library(Extension_28385, ):
    str_28386 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 4), 'str', 'Just like a regular Extension, but built as a library instead')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 44, 0, False)
        # Assigning a type to the variable 'self' (line 45)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Library.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'Library' (line 44)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 0), 'Library', Library)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 0))

# Multiple import statement. import sys (1/3) (line 47)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'sys', sys, module_type_store)
# Multiple import statement. import distutils.core (2/3) (line 47)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28387 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'distutils.core')

if (type(import_28387) is not StypyTypeError):

    if (import_28387 != 'pyd_module'):
        __import__(import_28387)
        sys_modules_28388 = sys.modules[import_28387]
        import_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'distutils.core', sys_modules_28388.module_type_store, module_type_store)
    else:
        import distutils.core

        import_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'distutils.core', distutils.core, module_type_store)

else:
    # Assigning a type to the variable 'distutils.core' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'distutils.core', import_28387)

# Multiple import statement. import distutils.extension (3/3) (line 47)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28389 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'distutils.extension')

if (type(import_28389) is not StypyTypeError):

    if (import_28389 != 'pyd_module'):
        __import__(import_28389)
        sys_modules_28390 = sys.modules[import_28389]
        import_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'distutils.extension', sys_modules_28390.module_type_store, module_type_store)
    else:
        import distutils.extension

        import_module(stypy.reporting.localization.Localization(__file__, 47, 0), 'distutils.extension', distutils.extension, module_type_store)

else:
    # Assigning a type to the variable 'distutils.extension' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'distutils.extension', import_28389)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


# Assigning a Name to a Attribute (line 48):

# Assigning a Name to a Attribute (line 48):
# Getting the type of 'Extension' (line 48)
Extension_28391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 27), 'Extension')
# Getting the type of 'distutils' (line 48)
distutils_28392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'distutils')
# Obtaining the member 'core' of a type (line 48)
core_28393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 0), distutils_28392, 'core')
# Setting the type of the member 'Extension' of a type (line 48)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 0), core_28393, 'Extension', Extension_28391)

# Assigning a Name to a Attribute (line 49):

# Assigning a Name to a Attribute (line 49):
# Getting the type of 'Extension' (line 49)
Extension_28394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 32), 'Extension')
# Getting the type of 'distutils' (line 49)
distutils_28395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 0), 'distutils')
# Obtaining the member 'extension' of a type (line 49)
extension_28396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 0), distutils_28395, 'extension')
# Setting the type of the member 'Extension' of a type (line 49)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 0), extension_28396, 'Extension', Extension_28394)


str_28397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 3), 'str', 'distutils.command.build_ext')
# Getting the type of 'sys' (line 50)
sys_28398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 36), 'sys')
# Obtaining the member 'modules' of a type (line 50)
modules_28399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 36), sys_28398, 'modules')
# Applying the binary operator 'in' (line 50)
result_contains_28400 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 3), 'in', str_28397, modules_28399)

# Testing the type of an if condition (line 50)
if_condition_28401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 0), result_contains_28400)
# Assigning a type to the variable 'if_condition_28401' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'if_condition_28401', if_condition_28401)
# SSA begins for if statement (line 50)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')

# Assigning a Name to a Attribute (line 51):

# Assigning a Name to a Attribute (line 51):
# Getting the type of 'Extension' (line 51)
Extension_28402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 59), 'Extension')

# Obtaining the type of the subscript
str_28403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 16), 'str', 'distutils.command.build_ext')
# Getting the type of 'sys' (line 51)
sys_28404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'sys')
# Obtaining the member 'modules' of a type (line 51)
modules_28405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), sys_28404, 'modules')
# Obtaining the member '__getitem__' of a type (line 51)
getitem___28406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), modules_28405, '__getitem__')
# Calling the subscript (__getitem__) to obtain the elements type (line 51)
subscript_call_result_28407 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), getitem___28406, str_28403)

# Setting the type of the member 'Extension' of a type (line 51)
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), subscript_call_result_28407, 'Extension', Extension_28402)
# SSA join for if statement (line 50)
module_type_store = module_type_store.join_ssa_context()


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
