
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''This module implements additional tests ala autoconf which can be useful.
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: 
7: # We put them here since they could be easily reused outside numpy.distutils
8: 
9: def check_inline(cmd):
10:     '''Return the inline identifier (may be empty).'''
11:     cmd._check_compiler()
12:     body = '''
13: #ifndef __cplusplus
14: static %(inline)s int static_func (void)
15: {
16:     return 0;
17: }
18: %(inline)s int nostatic_func (void)
19: {
20:     return 0;
21: }
22: #endif'''
23: 
24:     for kw in ['inline', '__inline__', '__inline']:
25:         st = cmd.try_compile(body % {'inline': kw}, None, None)
26:         if st:
27:             return kw
28: 
29:     return ''
30: 
31: def check_restrict(cmd):
32:     '''Return the restrict identifier (may be empty).'''
33:     cmd._check_compiler()
34:     body = '''
35: static int static_func (char * %(restrict)s a)
36: {
37:     return 0;
38: }
39: '''
40: 
41:     for kw in ['restrict', '__restrict__', '__restrict']:
42:         st = cmd.try_compile(body % {'restrict': kw}, None, None)
43:         if st:
44:             return kw
45: 
46:     return ''
47: 
48: def check_compiler_gcc4(cmd):
49:     '''Return True if the C compiler is GCC 4.x.'''
50:     cmd._check_compiler()
51:     body = '''
52: int
53: main()
54: {
55: #if (! defined __GNUC__) || (__GNUC__ < 4)
56: #error gcc >= 4 required
57: #endif
58:     return 0;
59: }
60: '''
61:     return cmd.try_compile(body, None, None)
62: 
63: 
64: def check_gcc_function_attribute(cmd, attribute, name):
65:     '''Return True if the given function attribute is supported.'''
66:     cmd._check_compiler()
67:     body = '''
68: #pragma GCC diagnostic error "-Wattributes"
69: #pragma clang diagnostic error "-Wattributes"
70: 
71: int %s %s(void*);
72: 
73: int
74: main()
75: {
76:     return 0;
77: }
78: ''' % (attribute, name)
79:     return cmd.try_compile(body, None, None) != 0
80: 
81: def check_gcc_variable_attribute(cmd, attribute):
82:     '''Return True if the given variable attribute is supported.'''
83:     cmd._check_compiler()
84:     body = '''
85: #pragma GCC diagnostic error "-Wattributes"
86: #pragma clang diagnostic error "-Wattributes"
87: 
88: int %s foo;
89: 
90: int
91: main()
92: {
93:     return 0;
94: }
95: ''' % (attribute, )
96:     return cmd.try_compile(body, None, None) != 0
97: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_52278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'This module implements additional tests ala autoconf which can be useful.\n\n')

@norecursion
def check_inline(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_inline'
    module_type_store = module_type_store.open_function_context('check_inline', 9, 0, False)
    
    # Passed parameters checking function
    check_inline.stypy_localization = localization
    check_inline.stypy_type_of_self = None
    check_inline.stypy_type_store = module_type_store
    check_inline.stypy_function_name = 'check_inline'
    check_inline.stypy_param_names_list = ['cmd']
    check_inline.stypy_varargs_param_name = None
    check_inline.stypy_kwargs_param_name = None
    check_inline.stypy_call_defaults = defaults
    check_inline.stypy_call_varargs = varargs
    check_inline.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_inline', ['cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_inline', localization, ['cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_inline(...)' code ##################

    str_52279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 4), 'str', 'Return the inline identifier (may be empty).')
    
    # Call to _check_compiler(...): (line 11)
    # Processing the call keyword arguments (line 11)
    kwargs_52282 = {}
    # Getting the type of 'cmd' (line 11)
    cmd_52280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'cmd', False)
    # Obtaining the member '_check_compiler' of a type (line 11)
    _check_compiler_52281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), cmd_52280, '_check_compiler')
    # Calling _check_compiler(args, kwargs) (line 11)
    _check_compiler_call_result_52283 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), _check_compiler_52281, *[], **kwargs_52282)
    
    
    # Assigning a Str to a Name (line 12):
    str_52284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n#ifndef __cplusplus\nstatic %(inline)s int static_func (void)\n{\n    return 0;\n}\n%(inline)s int nostatic_func (void)\n{\n    return 0;\n}\n#endif')
    # Assigning a type to the variable 'body' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'body', str_52284)
    
    
    # Obtaining an instance of the builtin type 'list' (line 24)
    list_52285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 24)
    # Adding element type (line 24)
    str_52286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 15), 'str', 'inline')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_52285, str_52286)
    # Adding element type (line 24)
    str_52287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'str', '__inline__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_52285, str_52287)
    # Adding element type (line 24)
    str_52288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 39), 'str', '__inline')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 14), list_52285, str_52288)
    
    # Testing the type of a for loop iterable (line 24)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 24, 4), list_52285)
    # Getting the type of the for loop variable (line 24)
    for_loop_var_52289 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 24, 4), list_52285)
    # Assigning a type to the variable 'kw' (line 24)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'kw', for_loop_var_52289)
    # SSA begins for a for statement (line 24)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 25):
    
    # Call to try_compile(...): (line 25)
    # Processing the call arguments (line 25)
    # Getting the type of 'body' (line 25)
    body_52292 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 29), 'body', False)
    
    # Obtaining an instance of the builtin type 'dict' (line 25)
    dict_52293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 25)
    # Adding element type (key, value) (line 25)
    str_52294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'str', 'inline')
    # Getting the type of 'kw' (line 25)
    kw_52295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 47), 'kw', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), dict_52293, (str_52294, kw_52295))
    
    # Applying the binary operator '%' (line 25)
    result_mod_52296 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 29), '%', body_52292, dict_52293)
    
    # Getting the type of 'None' (line 25)
    None_52297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 52), 'None', False)
    # Getting the type of 'None' (line 25)
    None_52298 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 58), 'None', False)
    # Processing the call keyword arguments (line 25)
    kwargs_52299 = {}
    # Getting the type of 'cmd' (line 25)
    cmd_52290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 13), 'cmd', False)
    # Obtaining the member 'try_compile' of a type (line 25)
    try_compile_52291 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 13), cmd_52290, 'try_compile')
    # Calling try_compile(args, kwargs) (line 25)
    try_compile_call_result_52300 = invoke(stypy.reporting.localization.Localization(__file__, 25, 13), try_compile_52291, *[result_mod_52296, None_52297, None_52298], **kwargs_52299)
    
    # Assigning a type to the variable 'st' (line 25)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'st', try_compile_call_result_52300)
    
    # Getting the type of 'st' (line 26)
    st_52301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 11), 'st')
    # Testing the type of an if condition (line 26)
    if_condition_52302 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 26, 8), st_52301)
    # Assigning a type to the variable 'if_condition_52302' (line 26)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'if_condition_52302', if_condition_52302)
    # SSA begins for if statement (line 26)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'kw' (line 27)
    kw_52303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'kw')
    # Assigning a type to the variable 'stypy_return_type' (line 27)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'stypy_return_type', kw_52303)
    # SSA join for if statement (line 26)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    str_52304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 11), 'str', '')
    # Assigning a type to the variable 'stypy_return_type' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type', str_52304)
    
    # ################# End of 'check_inline(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_inline' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_52305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52305)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_inline'
    return stypy_return_type_52305

# Assigning a type to the variable 'check_inline' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'check_inline', check_inline)

@norecursion
def check_restrict(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_restrict'
    module_type_store = module_type_store.open_function_context('check_restrict', 31, 0, False)
    
    # Passed parameters checking function
    check_restrict.stypy_localization = localization
    check_restrict.stypy_type_of_self = None
    check_restrict.stypy_type_store = module_type_store
    check_restrict.stypy_function_name = 'check_restrict'
    check_restrict.stypy_param_names_list = ['cmd']
    check_restrict.stypy_varargs_param_name = None
    check_restrict.stypy_kwargs_param_name = None
    check_restrict.stypy_call_defaults = defaults
    check_restrict.stypy_call_varargs = varargs
    check_restrict.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_restrict', ['cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_restrict', localization, ['cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_restrict(...)' code ##################

    str_52306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 4), 'str', 'Return the restrict identifier (may be empty).')
    
    # Call to _check_compiler(...): (line 33)
    # Processing the call keyword arguments (line 33)
    kwargs_52309 = {}
    # Getting the type of 'cmd' (line 33)
    cmd_52307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'cmd', False)
    # Obtaining the member '_check_compiler' of a type (line 33)
    _check_compiler_52308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), cmd_52307, '_check_compiler')
    # Calling _check_compiler(args, kwargs) (line 33)
    _check_compiler_call_result_52310 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), _check_compiler_52308, *[], **kwargs_52309)
    
    
    # Assigning a Str to a Name (line 34):
    str_52311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, (-1)), 'str', '\nstatic int static_func (char * %(restrict)s a)\n{\n    return 0;\n}\n')
    # Assigning a type to the variable 'body' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'body', str_52311)
    
    
    # Obtaining an instance of the builtin type 'list' (line 41)
    list_52312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 41)
    # Adding element type (line 41)
    str_52313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'str', 'restrict')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_52312, str_52313)
    # Adding element type (line 41)
    str_52314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 27), 'str', '__restrict__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_52312, str_52314)
    # Adding element type (line 41)
    str_52315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 43), 'str', '__restrict')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 14), list_52312, str_52315)
    
    # Testing the type of a for loop iterable (line 41)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 4), list_52312)
    # Getting the type of the for loop variable (line 41)
    for_loop_var_52316 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 4), list_52312)
    # Assigning a type to the variable 'kw' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'kw', for_loop_var_52316)
    # SSA begins for a for statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 42):
    
    # Call to try_compile(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'body' (line 42)
    body_52319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 29), 'body', False)
    
    # Obtaining an instance of the builtin type 'dict' (line 42)
    dict_52320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 36), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 42)
    # Adding element type (key, value) (line 42)
    str_52321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 37), 'str', 'restrict')
    # Getting the type of 'kw' (line 42)
    kw_52322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 49), 'kw', False)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 36), dict_52320, (str_52321, kw_52322))
    
    # Applying the binary operator '%' (line 42)
    result_mod_52323 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 29), '%', body_52319, dict_52320)
    
    # Getting the type of 'None' (line 42)
    None_52324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 54), 'None', False)
    # Getting the type of 'None' (line 42)
    None_52325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 60), 'None', False)
    # Processing the call keyword arguments (line 42)
    kwargs_52326 = {}
    # Getting the type of 'cmd' (line 42)
    cmd_52317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'cmd', False)
    # Obtaining the member 'try_compile' of a type (line 42)
    try_compile_52318 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 13), cmd_52317, 'try_compile')
    # Calling try_compile(args, kwargs) (line 42)
    try_compile_call_result_52327 = invoke(stypy.reporting.localization.Localization(__file__, 42, 13), try_compile_52318, *[result_mod_52323, None_52324, None_52325], **kwargs_52326)
    
    # Assigning a type to the variable 'st' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'st', try_compile_call_result_52327)
    
    # Getting the type of 'st' (line 43)
    st_52328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 11), 'st')
    # Testing the type of an if condition (line 43)
    if_condition_52329 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 43, 8), st_52328)
    # Assigning a type to the variable 'if_condition_52329' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'if_condition_52329', if_condition_52329)
    # SSA begins for if statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'kw' (line 44)
    kw_52330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'kw')
    # Assigning a type to the variable 'stypy_return_type' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'stypy_return_type', kw_52330)
    # SSA join for if statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    str_52331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 11), 'str', '')
    # Assigning a type to the variable 'stypy_return_type' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type', str_52331)
    
    # ################# End of 'check_restrict(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_restrict' in the type store
    # Getting the type of 'stypy_return_type' (line 31)
    stypy_return_type_52332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52332)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_restrict'
    return stypy_return_type_52332

# Assigning a type to the variable 'check_restrict' (line 31)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'check_restrict', check_restrict)

@norecursion
def check_compiler_gcc4(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_compiler_gcc4'
    module_type_store = module_type_store.open_function_context('check_compiler_gcc4', 48, 0, False)
    
    # Passed parameters checking function
    check_compiler_gcc4.stypy_localization = localization
    check_compiler_gcc4.stypy_type_of_self = None
    check_compiler_gcc4.stypy_type_store = module_type_store
    check_compiler_gcc4.stypy_function_name = 'check_compiler_gcc4'
    check_compiler_gcc4.stypy_param_names_list = ['cmd']
    check_compiler_gcc4.stypy_varargs_param_name = None
    check_compiler_gcc4.stypy_kwargs_param_name = None
    check_compiler_gcc4.stypy_call_defaults = defaults
    check_compiler_gcc4.stypy_call_varargs = varargs
    check_compiler_gcc4.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_compiler_gcc4', ['cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_compiler_gcc4', localization, ['cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_compiler_gcc4(...)' code ##################

    str_52333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'str', 'Return True if the C compiler is GCC 4.x.')
    
    # Call to _check_compiler(...): (line 50)
    # Processing the call keyword arguments (line 50)
    kwargs_52336 = {}
    # Getting the type of 'cmd' (line 50)
    cmd_52334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'cmd', False)
    # Obtaining the member '_check_compiler' of a type (line 50)
    _check_compiler_52335 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 4), cmd_52334, '_check_compiler')
    # Calling _check_compiler(args, kwargs) (line 50)
    _check_compiler_call_result_52337 = invoke(stypy.reporting.localization.Localization(__file__, 50, 4), _check_compiler_52335, *[], **kwargs_52336)
    
    
    # Assigning a Str to a Name (line 51):
    str_52338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, (-1)), 'str', '\nint\nmain()\n{\n#if (! defined __GNUC__) || (__GNUC__ < 4)\n#error gcc >= 4 required\n#endif\n    return 0;\n}\n')
    # Assigning a type to the variable 'body' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'body', str_52338)
    
    # Call to try_compile(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'body' (line 61)
    body_52341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 27), 'body', False)
    # Getting the type of 'None' (line 61)
    None_52342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 33), 'None', False)
    # Getting the type of 'None' (line 61)
    None_52343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 39), 'None', False)
    # Processing the call keyword arguments (line 61)
    kwargs_52344 = {}
    # Getting the type of 'cmd' (line 61)
    cmd_52339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'cmd', False)
    # Obtaining the member 'try_compile' of a type (line 61)
    try_compile_52340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 11), cmd_52339, 'try_compile')
    # Calling try_compile(args, kwargs) (line 61)
    try_compile_call_result_52345 = invoke(stypy.reporting.localization.Localization(__file__, 61, 11), try_compile_52340, *[body_52341, None_52342, None_52343], **kwargs_52344)
    
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', try_compile_call_result_52345)
    
    # ################# End of 'check_compiler_gcc4(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_compiler_gcc4' in the type store
    # Getting the type of 'stypy_return_type' (line 48)
    stypy_return_type_52346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52346)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_compiler_gcc4'
    return stypy_return_type_52346

# Assigning a type to the variable 'check_compiler_gcc4' (line 48)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 0), 'check_compiler_gcc4', check_compiler_gcc4)

@norecursion
def check_gcc_function_attribute(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_gcc_function_attribute'
    module_type_store = module_type_store.open_function_context('check_gcc_function_attribute', 64, 0, False)
    
    # Passed parameters checking function
    check_gcc_function_attribute.stypy_localization = localization
    check_gcc_function_attribute.stypy_type_of_self = None
    check_gcc_function_attribute.stypy_type_store = module_type_store
    check_gcc_function_attribute.stypy_function_name = 'check_gcc_function_attribute'
    check_gcc_function_attribute.stypy_param_names_list = ['cmd', 'attribute', 'name']
    check_gcc_function_attribute.stypy_varargs_param_name = None
    check_gcc_function_attribute.stypy_kwargs_param_name = None
    check_gcc_function_attribute.stypy_call_defaults = defaults
    check_gcc_function_attribute.stypy_call_varargs = varargs
    check_gcc_function_attribute.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_gcc_function_attribute', ['cmd', 'attribute', 'name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_gcc_function_attribute', localization, ['cmd', 'attribute', 'name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_gcc_function_attribute(...)' code ##################

    str_52347 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 4), 'str', 'Return True if the given function attribute is supported.')
    
    # Call to _check_compiler(...): (line 66)
    # Processing the call keyword arguments (line 66)
    kwargs_52350 = {}
    # Getting the type of 'cmd' (line 66)
    cmd_52348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'cmd', False)
    # Obtaining the member '_check_compiler' of a type (line 66)
    _check_compiler_52349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 4), cmd_52348, '_check_compiler')
    # Calling _check_compiler(args, kwargs) (line 66)
    _check_compiler_call_result_52351 = invoke(stypy.reporting.localization.Localization(__file__, 66, 4), _check_compiler_52349, *[], **kwargs_52350)
    
    
    # Assigning a BinOp to a Name (line 67):
    str_52352 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, (-1)), 'str', '\n#pragma GCC diagnostic error "-Wattributes"\n#pragma clang diagnostic error "-Wattributes"\n\nint %s %s(void*);\n\nint\nmain()\n{\n    return 0;\n}\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 78)
    tuple_52353 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 7), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 78)
    # Adding element type (line 78)
    # Getting the type of 'attribute' (line 78)
    attribute_52354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), 'attribute')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 7), tuple_52353, attribute_52354)
    # Adding element type (line 78)
    # Getting the type of 'name' (line 78)
    name_52355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), 'name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 7), tuple_52353, name_52355)
    
    # Applying the binary operator '%' (line 78)
    result_mod_52356 = python_operator(stypy.reporting.localization.Localization(__file__, 78, (-1)), '%', str_52352, tuple_52353)
    
    # Assigning a type to the variable 'body' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'body', result_mod_52356)
    
    
    # Call to try_compile(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'body' (line 79)
    body_52359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'body', False)
    # Getting the type of 'None' (line 79)
    None_52360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 33), 'None', False)
    # Getting the type of 'None' (line 79)
    None_52361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 39), 'None', False)
    # Processing the call keyword arguments (line 79)
    kwargs_52362 = {}
    # Getting the type of 'cmd' (line 79)
    cmd_52357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'cmd', False)
    # Obtaining the member 'try_compile' of a type (line 79)
    try_compile_52358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), cmd_52357, 'try_compile')
    # Calling try_compile(args, kwargs) (line 79)
    try_compile_call_result_52363 = invoke(stypy.reporting.localization.Localization(__file__, 79, 11), try_compile_52358, *[body_52359, None_52360, None_52361], **kwargs_52362)
    
    int_52364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 48), 'int')
    # Applying the binary operator '!=' (line 79)
    result_ne_52365 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), '!=', try_compile_call_result_52363, int_52364)
    
    # Assigning a type to the variable 'stypy_return_type' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type', result_ne_52365)
    
    # ################# End of 'check_gcc_function_attribute(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_gcc_function_attribute' in the type store
    # Getting the type of 'stypy_return_type' (line 64)
    stypy_return_type_52366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52366)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_gcc_function_attribute'
    return stypy_return_type_52366

# Assigning a type to the variable 'check_gcc_function_attribute' (line 64)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'check_gcc_function_attribute', check_gcc_function_attribute)

@norecursion
def check_gcc_variable_attribute(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_gcc_variable_attribute'
    module_type_store = module_type_store.open_function_context('check_gcc_variable_attribute', 81, 0, False)
    
    # Passed parameters checking function
    check_gcc_variable_attribute.stypy_localization = localization
    check_gcc_variable_attribute.stypy_type_of_self = None
    check_gcc_variable_attribute.stypy_type_store = module_type_store
    check_gcc_variable_attribute.stypy_function_name = 'check_gcc_variable_attribute'
    check_gcc_variable_attribute.stypy_param_names_list = ['cmd', 'attribute']
    check_gcc_variable_attribute.stypy_varargs_param_name = None
    check_gcc_variable_attribute.stypy_kwargs_param_name = None
    check_gcc_variable_attribute.stypy_call_defaults = defaults
    check_gcc_variable_attribute.stypy_call_varargs = varargs
    check_gcc_variable_attribute.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_gcc_variable_attribute', ['cmd', 'attribute'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_gcc_variable_attribute', localization, ['cmd', 'attribute'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_gcc_variable_attribute(...)' code ##################

    str_52367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 4), 'str', 'Return True if the given variable attribute is supported.')
    
    # Call to _check_compiler(...): (line 83)
    # Processing the call keyword arguments (line 83)
    kwargs_52370 = {}
    # Getting the type of 'cmd' (line 83)
    cmd_52368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'cmd', False)
    # Obtaining the member '_check_compiler' of a type (line 83)
    _check_compiler_52369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 4), cmd_52368, '_check_compiler')
    # Calling _check_compiler(args, kwargs) (line 83)
    _check_compiler_call_result_52371 = invoke(stypy.reporting.localization.Localization(__file__, 83, 4), _check_compiler_52369, *[], **kwargs_52370)
    
    
    # Assigning a BinOp to a Name (line 84):
    str_52372 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, (-1)), 'str', '\n#pragma GCC diagnostic error "-Wattributes"\n#pragma clang diagnostic error "-Wattributes"\n\nint %s foo;\n\nint\nmain()\n{\n    return 0;\n}\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 95)
    tuple_52373 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 7), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 95)
    # Adding element type (line 95)
    # Getting the type of 'attribute' (line 95)
    attribute_52374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 7), 'attribute')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 7), tuple_52373, attribute_52374)
    
    # Applying the binary operator '%' (line 95)
    result_mod_52375 = python_operator(stypy.reporting.localization.Localization(__file__, 95, (-1)), '%', str_52372, tuple_52373)
    
    # Assigning a type to the variable 'body' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'body', result_mod_52375)
    
    
    # Call to try_compile(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'body' (line 96)
    body_52378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'body', False)
    # Getting the type of 'None' (line 96)
    None_52379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 33), 'None', False)
    # Getting the type of 'None' (line 96)
    None_52380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 39), 'None', False)
    # Processing the call keyword arguments (line 96)
    kwargs_52381 = {}
    # Getting the type of 'cmd' (line 96)
    cmd_52376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 11), 'cmd', False)
    # Obtaining the member 'try_compile' of a type (line 96)
    try_compile_52377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 11), cmd_52376, 'try_compile')
    # Calling try_compile(args, kwargs) (line 96)
    try_compile_call_result_52382 = invoke(stypy.reporting.localization.Localization(__file__, 96, 11), try_compile_52377, *[body_52378, None_52379, None_52380], **kwargs_52381)
    
    int_52383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 48), 'int')
    # Applying the binary operator '!=' (line 96)
    result_ne_52384 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 11), '!=', try_compile_call_result_52382, int_52383)
    
    # Assigning a type to the variable 'stypy_return_type' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type', result_ne_52384)
    
    # ################# End of 'check_gcc_variable_attribute(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_gcc_variable_attribute' in the type store
    # Getting the type of 'stypy_return_type' (line 81)
    stypy_return_type_52385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52385)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_gcc_variable_attribute'
    return stypy_return_type_52385

# Assigning a type to the variable 'check_gcc_variable_attribute' (line 81)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 0), 'check_gcc_variable_attribute', check_gcc_variable_attribute)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
