
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: #!/usr/bin/env python
2: from __future__ import division, absolute_import, print_function
3: 
4: import os
5: import sys
6: import tempfile
7: 
8: 
9: def run_command(cmd):
10:     print('Running %r:' % (cmd))
11:     os.system(cmd)
12:     print('------')
13: 
14: 
15: def run():
16:     _path = os.getcwd()
17:     os.chdir(tempfile.gettempdir())
18:     print('------')
19:     print('os.name=%r' % (os.name))
20:     print('------')
21:     print('sys.platform=%r' % (sys.platform))
22:     print('------')
23:     print('sys.version:')
24:     print(sys.version)
25:     print('------')
26:     print('sys.prefix:')
27:     print(sys.prefix)
28:     print('------')
29:     print('sys.path=%r' % (':'.join(sys.path)))
30:     print('------')
31: 
32:     try:
33:         import numpy
34:         has_newnumpy = 1
35:     except ImportError:
36:         print('Failed to import new numpy:', sys.exc_info()[1])
37:         has_newnumpy = 0
38: 
39:     try:
40:         from numpy.f2py import f2py2e
41:         has_f2py2e = 1
42:     except ImportError:
43:         print('Failed to import f2py2e:', sys.exc_info()[1])
44:         has_f2py2e = 0
45: 
46:     try:
47:         import numpy.distutils
48:         has_numpy_distutils = 2
49:     except ImportError:
50:         try:
51:             import numpy_distutils
52:             has_numpy_distutils = 1
53:         except ImportError:
54:             print('Failed to import numpy_distutils:', sys.exc_info()[1])
55:             has_numpy_distutils = 0
56: 
57:     if has_newnumpy:
58:         try:
59:             print('Found new numpy version %r in %s' %
60:                   (numpy.__version__, numpy.__file__))
61:         except Exception as msg:
62:             print('error:', msg)
63:             print('------')
64: 
65:     if has_f2py2e:
66:         try:
67:             print('Found f2py2e version %r in %s' %
68:                   (f2py2e.__version__.version, f2py2e.__file__))
69:         except Exception as msg:
70:             print('error:', msg)
71:             print('------')
72: 
73:     if has_numpy_distutils:
74:         try:
75:             if has_numpy_distutils == 2:
76:                 print('Found numpy.distutils version %r in %r' % (
77:                     numpy.distutils.__version__,
78:                     numpy.distutils.__file__))
79:             else:
80:                 print('Found numpy_distutils version %r in %r' % (
81:                     numpy_distutils.numpy_distutils_version.numpy_distutils_version,
82:                     numpy_distutils.__file__))
83:             print('------')
84:         except Exception as msg:
85:             print('error:', msg)
86:             print('------')
87:         try:
88:             if has_numpy_distutils == 1:
89:                 print(
90:                     'Importing numpy_distutils.command.build_flib ...', end=' ')
91:                 import numpy_distutils.command.build_flib as build_flib
92:                 print('ok')
93:                 print('------')
94:                 try:
95:                     print(
96:                         'Checking availability of supported Fortran compilers:')
97:                     for compiler_class in build_flib.all_compilers:
98:                         compiler_class(verbose=1).is_available()
99:                         print('------')
100:                 except Exception as msg:
101:                     print('error:', msg)
102:                     print('------')
103:         except Exception as msg:
104:             print(
105:                 'error:', msg, '(ignore it, build_flib is obsolute for numpy.distutils 0.2.2 and up)')
106:             print('------')
107:         try:
108:             if has_numpy_distutils == 2:
109:                 print('Importing numpy.distutils.fcompiler ...', end=' ')
110:                 import numpy.distutils.fcompiler as fcompiler
111:             else:
112:                 print('Importing numpy_distutils.fcompiler ...', end=' ')
113:                 import numpy_distutils.fcompiler as fcompiler
114:             print('ok')
115:             print('------')
116:             try:
117:                 print('Checking availability of supported Fortran compilers:')
118:                 fcompiler.show_fcompilers()
119:                 print('------')
120:             except Exception as msg:
121:                 print('error:', msg)
122:                 print('------')
123:         except Exception as msg:
124:             print('error:', msg)
125:             print('------')
126:         try:
127:             if has_numpy_distutils == 2:
128:                 print('Importing numpy.distutils.cpuinfo ...', end=' ')
129:                 from numpy.distutils.cpuinfo import cpuinfo
130:                 print('ok')
131:                 print('------')
132:             else:
133:                 try:
134:                     print(
135:                         'Importing numpy_distutils.command.cpuinfo ...', end=' ')
136:                     from numpy_distutils.command.cpuinfo import cpuinfo
137:                     print('ok')
138:                     print('------')
139:                 except Exception as msg:
140:                     print('error:', msg, '(ignore it)')
141:                     print('Importing numpy_distutils.cpuinfo ...', end=' ')
142:                     from numpy_distutils.cpuinfo import cpuinfo
143:                     print('ok')
144:                     print('------')
145:             cpu = cpuinfo()
146:             print('CPU information:', end=' ')
147:             for name in dir(cpuinfo):
148:                 if name[0] == '_' and name[1] != '_' and getattr(cpu, name[1:])():
149:                     print(name[1:], end=' ')
150:             print('------')
151:         except Exception as msg:
152:             print('error:', msg)
153:             print('------')
154:     os.chdir(_path)
155: if __name__ == "__main__":
156:     run()
157: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import os' statement (line 4)
import os

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import sys' statement (line 5)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import tempfile' statement (line 6)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'tempfile', tempfile, module_type_store)


@norecursion
def run_command(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run_command'
    module_type_store = module_type_store.open_function_context('run_command', 9, 0, False)
    
    # Passed parameters checking function
    run_command.stypy_localization = localization
    run_command.stypy_type_of_self = None
    run_command.stypy_type_store = module_type_store
    run_command.stypy_function_name = 'run_command'
    run_command.stypy_param_names_list = ['cmd']
    run_command.stypy_varargs_param_name = None
    run_command.stypy_kwargs_param_name = None
    run_command.stypy_call_defaults = defaults
    run_command.stypy_call_varargs = varargs
    run_command.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run_command', ['cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run_command', localization, ['cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run_command(...)' code ##################

    
    # Call to print(...): (line 10)
    # Processing the call arguments (line 10)
    str_90356 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 10), 'str', 'Running %r:')
    # Getting the type of 'cmd' (line 10)
    cmd_90357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 27), 'cmd', False)
    # Applying the binary operator '%' (line 10)
    result_mod_90358 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 10), '%', str_90356, cmd_90357)
    
    # Processing the call keyword arguments (line 10)
    kwargs_90359 = {}
    # Getting the type of 'print' (line 10)
    print_90355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'print', False)
    # Calling print(args, kwargs) (line 10)
    print_call_result_90360 = invoke(stypy.reporting.localization.Localization(__file__, 10, 4), print_90355, *[result_mod_90358], **kwargs_90359)
    
    
    # Call to system(...): (line 11)
    # Processing the call arguments (line 11)
    # Getting the type of 'cmd' (line 11)
    cmd_90363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 14), 'cmd', False)
    # Processing the call keyword arguments (line 11)
    kwargs_90364 = {}
    # Getting the type of 'os' (line 11)
    os_90361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'os', False)
    # Obtaining the member 'system' of a type (line 11)
    system_90362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), os_90361, 'system')
    # Calling system(args, kwargs) (line 11)
    system_call_result_90365 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), system_90362, *[cmd_90363], **kwargs_90364)
    
    
    # Call to print(...): (line 12)
    # Processing the call arguments (line 12)
    str_90367 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 10), 'str', '------')
    # Processing the call keyword arguments (line 12)
    kwargs_90368 = {}
    # Getting the type of 'print' (line 12)
    print_90366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'print', False)
    # Calling print(args, kwargs) (line 12)
    print_call_result_90369 = invoke(stypy.reporting.localization.Localization(__file__, 12, 4), print_90366, *[str_90367], **kwargs_90368)
    
    
    # ################# End of 'run_command(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run_command' in the type store
    # Getting the type of 'stypy_return_type' (line 9)
    stypy_return_type_90370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_90370)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run_command'
    return stypy_return_type_90370

# Assigning a type to the variable 'run_command' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'run_command', run_command)

@norecursion
def run(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'run'
    module_type_store = module_type_store.open_function_context('run', 15, 0, False)
    
    # Passed parameters checking function
    run.stypy_localization = localization
    run.stypy_type_of_self = None
    run.stypy_type_store = module_type_store
    run.stypy_function_name = 'run'
    run.stypy_param_names_list = []
    run.stypy_varargs_param_name = None
    run.stypy_kwargs_param_name = None
    run.stypy_call_defaults = defaults
    run.stypy_call_varargs = varargs
    run.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run(...)' code ##################

    
    # Assigning a Call to a Name (line 16):
    
    # Call to getcwd(...): (line 16)
    # Processing the call keyword arguments (line 16)
    kwargs_90373 = {}
    # Getting the type of 'os' (line 16)
    os_90371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'os', False)
    # Obtaining the member 'getcwd' of a type (line 16)
    getcwd_90372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 12), os_90371, 'getcwd')
    # Calling getcwd(args, kwargs) (line 16)
    getcwd_call_result_90374 = invoke(stypy.reporting.localization.Localization(__file__, 16, 12), getcwd_90372, *[], **kwargs_90373)
    
    # Assigning a type to the variable '_path' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), '_path', getcwd_call_result_90374)
    
    # Call to chdir(...): (line 17)
    # Processing the call arguments (line 17)
    
    # Call to gettempdir(...): (line 17)
    # Processing the call keyword arguments (line 17)
    kwargs_90379 = {}
    # Getting the type of 'tempfile' (line 17)
    tempfile_90377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 13), 'tempfile', False)
    # Obtaining the member 'gettempdir' of a type (line 17)
    gettempdir_90378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 13), tempfile_90377, 'gettempdir')
    # Calling gettempdir(args, kwargs) (line 17)
    gettempdir_call_result_90380 = invoke(stypy.reporting.localization.Localization(__file__, 17, 13), gettempdir_90378, *[], **kwargs_90379)
    
    # Processing the call keyword arguments (line 17)
    kwargs_90381 = {}
    # Getting the type of 'os' (line 17)
    os_90375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'os', False)
    # Obtaining the member 'chdir' of a type (line 17)
    chdir_90376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 4), os_90375, 'chdir')
    # Calling chdir(args, kwargs) (line 17)
    chdir_call_result_90382 = invoke(stypy.reporting.localization.Localization(__file__, 17, 4), chdir_90376, *[gettempdir_call_result_90380], **kwargs_90381)
    
    
    # Call to print(...): (line 18)
    # Processing the call arguments (line 18)
    str_90384 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 10), 'str', '------')
    # Processing the call keyword arguments (line 18)
    kwargs_90385 = {}
    # Getting the type of 'print' (line 18)
    print_90383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'print', False)
    # Calling print(args, kwargs) (line 18)
    print_call_result_90386 = invoke(stypy.reporting.localization.Localization(__file__, 18, 4), print_90383, *[str_90384], **kwargs_90385)
    
    
    # Call to print(...): (line 19)
    # Processing the call arguments (line 19)
    str_90388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 10), 'str', 'os.name=%r')
    # Getting the type of 'os' (line 19)
    os_90389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 26), 'os', False)
    # Obtaining the member 'name' of a type (line 19)
    name_90390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 26), os_90389, 'name')
    # Applying the binary operator '%' (line 19)
    result_mod_90391 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 10), '%', str_90388, name_90390)
    
    # Processing the call keyword arguments (line 19)
    kwargs_90392 = {}
    # Getting the type of 'print' (line 19)
    print_90387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'print', False)
    # Calling print(args, kwargs) (line 19)
    print_call_result_90393 = invoke(stypy.reporting.localization.Localization(__file__, 19, 4), print_90387, *[result_mod_90391], **kwargs_90392)
    
    
    # Call to print(...): (line 20)
    # Processing the call arguments (line 20)
    str_90395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 10), 'str', '------')
    # Processing the call keyword arguments (line 20)
    kwargs_90396 = {}
    # Getting the type of 'print' (line 20)
    print_90394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'print', False)
    # Calling print(args, kwargs) (line 20)
    print_call_result_90397 = invoke(stypy.reporting.localization.Localization(__file__, 20, 4), print_90394, *[str_90395], **kwargs_90396)
    
    
    # Call to print(...): (line 21)
    # Processing the call arguments (line 21)
    str_90399 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 10), 'str', 'sys.platform=%r')
    # Getting the type of 'sys' (line 21)
    sys_90400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 31), 'sys', False)
    # Obtaining the member 'platform' of a type (line 21)
    platform_90401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 31), sys_90400, 'platform')
    # Applying the binary operator '%' (line 21)
    result_mod_90402 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 10), '%', str_90399, platform_90401)
    
    # Processing the call keyword arguments (line 21)
    kwargs_90403 = {}
    # Getting the type of 'print' (line 21)
    print_90398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'print', False)
    # Calling print(args, kwargs) (line 21)
    print_call_result_90404 = invoke(stypy.reporting.localization.Localization(__file__, 21, 4), print_90398, *[result_mod_90402], **kwargs_90403)
    
    
    # Call to print(...): (line 22)
    # Processing the call arguments (line 22)
    str_90406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 10), 'str', '------')
    # Processing the call keyword arguments (line 22)
    kwargs_90407 = {}
    # Getting the type of 'print' (line 22)
    print_90405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'print', False)
    # Calling print(args, kwargs) (line 22)
    print_call_result_90408 = invoke(stypy.reporting.localization.Localization(__file__, 22, 4), print_90405, *[str_90406], **kwargs_90407)
    
    
    # Call to print(...): (line 23)
    # Processing the call arguments (line 23)
    str_90410 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 10), 'str', 'sys.version:')
    # Processing the call keyword arguments (line 23)
    kwargs_90411 = {}
    # Getting the type of 'print' (line 23)
    print_90409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'print', False)
    # Calling print(args, kwargs) (line 23)
    print_call_result_90412 = invoke(stypy.reporting.localization.Localization(__file__, 23, 4), print_90409, *[str_90410], **kwargs_90411)
    
    
    # Call to print(...): (line 24)
    # Processing the call arguments (line 24)
    # Getting the type of 'sys' (line 24)
    sys_90414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 10), 'sys', False)
    # Obtaining the member 'version' of a type (line 24)
    version_90415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 10), sys_90414, 'version')
    # Processing the call keyword arguments (line 24)
    kwargs_90416 = {}
    # Getting the type of 'print' (line 24)
    print_90413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'print', False)
    # Calling print(args, kwargs) (line 24)
    print_call_result_90417 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), print_90413, *[version_90415], **kwargs_90416)
    
    
    # Call to print(...): (line 25)
    # Processing the call arguments (line 25)
    str_90419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 10), 'str', '------')
    # Processing the call keyword arguments (line 25)
    kwargs_90420 = {}
    # Getting the type of 'print' (line 25)
    print_90418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'print', False)
    # Calling print(args, kwargs) (line 25)
    print_call_result_90421 = invoke(stypy.reporting.localization.Localization(__file__, 25, 4), print_90418, *[str_90419], **kwargs_90420)
    
    
    # Call to print(...): (line 26)
    # Processing the call arguments (line 26)
    str_90423 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 10), 'str', 'sys.prefix:')
    # Processing the call keyword arguments (line 26)
    kwargs_90424 = {}
    # Getting the type of 'print' (line 26)
    print_90422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'print', False)
    # Calling print(args, kwargs) (line 26)
    print_call_result_90425 = invoke(stypy.reporting.localization.Localization(__file__, 26, 4), print_90422, *[str_90423], **kwargs_90424)
    
    
    # Call to print(...): (line 27)
    # Processing the call arguments (line 27)
    # Getting the type of 'sys' (line 27)
    sys_90427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 10), 'sys', False)
    # Obtaining the member 'prefix' of a type (line 27)
    prefix_90428 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 10), sys_90427, 'prefix')
    # Processing the call keyword arguments (line 27)
    kwargs_90429 = {}
    # Getting the type of 'print' (line 27)
    print_90426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'print', False)
    # Calling print(args, kwargs) (line 27)
    print_call_result_90430 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), print_90426, *[prefix_90428], **kwargs_90429)
    
    
    # Call to print(...): (line 28)
    # Processing the call arguments (line 28)
    str_90432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 10), 'str', '------')
    # Processing the call keyword arguments (line 28)
    kwargs_90433 = {}
    # Getting the type of 'print' (line 28)
    print_90431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'print', False)
    # Calling print(args, kwargs) (line 28)
    print_call_result_90434 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), print_90431, *[str_90432], **kwargs_90433)
    
    
    # Call to print(...): (line 29)
    # Processing the call arguments (line 29)
    str_90436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 10), 'str', 'sys.path=%r')
    
    # Call to join(...): (line 29)
    # Processing the call arguments (line 29)
    # Getting the type of 'sys' (line 29)
    sys_90439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 36), 'sys', False)
    # Obtaining the member 'path' of a type (line 29)
    path_90440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 36), sys_90439, 'path')
    # Processing the call keyword arguments (line 29)
    kwargs_90441 = {}
    str_90437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'str', ':')
    # Obtaining the member 'join' of a type (line 29)
    join_90438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 27), str_90437, 'join')
    # Calling join(args, kwargs) (line 29)
    join_call_result_90442 = invoke(stypy.reporting.localization.Localization(__file__, 29, 27), join_90438, *[path_90440], **kwargs_90441)
    
    # Applying the binary operator '%' (line 29)
    result_mod_90443 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 10), '%', str_90436, join_call_result_90442)
    
    # Processing the call keyword arguments (line 29)
    kwargs_90444 = {}
    # Getting the type of 'print' (line 29)
    print_90435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'print', False)
    # Calling print(args, kwargs) (line 29)
    print_call_result_90445 = invoke(stypy.reporting.localization.Localization(__file__, 29, 4), print_90435, *[result_mod_90443], **kwargs_90444)
    
    
    # Call to print(...): (line 30)
    # Processing the call arguments (line 30)
    str_90447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 10), 'str', '------')
    # Processing the call keyword arguments (line 30)
    kwargs_90448 = {}
    # Getting the type of 'print' (line 30)
    print_90446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'print', False)
    # Calling print(args, kwargs) (line 30)
    print_call_result_90449 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), print_90446, *[str_90447], **kwargs_90448)
    
    
    
    # SSA begins for try-except statement (line 32)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 33, 8))
    
    # 'import numpy' statement (line 33)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90450 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 33, 8), 'numpy')

    if (type(import_90450) is not StypyTypeError):

        if (import_90450 != 'pyd_module'):
            __import__(import_90450)
            sys_modules_90451 = sys.modules[import_90450]
            import_module(stypy.reporting.localization.Localization(__file__, 33, 8), 'numpy', sys_modules_90451.module_type_store, module_type_store)
        else:
            import numpy

            import_module(stypy.reporting.localization.Localization(__file__, 33, 8), 'numpy', numpy, module_type_store)

    else:
        # Assigning a type to the variable 'numpy' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'numpy', import_90450)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Num to a Name (line 34):
    int_90452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'int')
    # Assigning a type to the variable 'has_newnumpy' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'has_newnumpy', int_90452)
    # SSA branch for the except part of a try statement (line 32)
    # SSA branch for the except 'ImportError' branch of a try statement (line 32)
    module_type_store.open_ssa_branch('except')
    
    # Call to print(...): (line 36)
    # Processing the call arguments (line 36)
    str_90454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 14), 'str', 'Failed to import new numpy:')
    
    # Obtaining the type of the subscript
    int_90455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 60), 'int')
    
    # Call to exc_info(...): (line 36)
    # Processing the call keyword arguments (line 36)
    kwargs_90458 = {}
    # Getting the type of 'sys' (line 36)
    sys_90456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 45), 'sys', False)
    # Obtaining the member 'exc_info' of a type (line 36)
    exc_info_90457 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 45), sys_90456, 'exc_info')
    # Calling exc_info(args, kwargs) (line 36)
    exc_info_call_result_90459 = invoke(stypy.reporting.localization.Localization(__file__, 36, 45), exc_info_90457, *[], **kwargs_90458)
    
    # Obtaining the member '__getitem__' of a type (line 36)
    getitem___90460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 45), exc_info_call_result_90459, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 36)
    subscript_call_result_90461 = invoke(stypy.reporting.localization.Localization(__file__, 36, 45), getitem___90460, int_90455)
    
    # Processing the call keyword arguments (line 36)
    kwargs_90462 = {}
    # Getting the type of 'print' (line 36)
    print_90453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'print', False)
    # Calling print(args, kwargs) (line 36)
    print_call_result_90463 = invoke(stypy.reporting.localization.Localization(__file__, 36, 8), print_90453, *[str_90454, subscript_call_result_90461], **kwargs_90462)
    
    
    # Assigning a Num to a Name (line 37):
    int_90464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 23), 'int')
    # Assigning a type to the variable 'has_newnumpy' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'has_newnumpy', int_90464)
    # SSA join for try-except statement (line 32)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 8))
    
    # 'from numpy.f2py import f2py2e' statement (line 40)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90465 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy.f2py')

    if (type(import_90465) is not StypyTypeError):

        if (import_90465 != 'pyd_module'):
            __import__(import_90465)
            sys_modules_90466 = sys.modules[import_90465]
            import_from_module(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy.f2py', sys_modules_90466.module_type_store, module_type_store, ['f2py2e'])
            nest_module(stypy.reporting.localization.Localization(__file__, 40, 8), __file__, sys_modules_90466, sys_modules_90466.module_type_store, module_type_store)
        else:
            from numpy.f2py import f2py2e

            import_from_module(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy.f2py', None, module_type_store, ['f2py2e'], [f2py2e])

    else:
        # Assigning a type to the variable 'numpy.f2py' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'numpy.f2py', import_90465)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Num to a Name (line 41):
    int_90467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'int')
    # Assigning a type to the variable 'has_f2py2e' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'has_f2py2e', int_90467)
    # SSA branch for the except part of a try statement (line 39)
    # SSA branch for the except 'ImportError' branch of a try statement (line 39)
    module_type_store.open_ssa_branch('except')
    
    # Call to print(...): (line 43)
    # Processing the call arguments (line 43)
    str_90469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 14), 'str', 'Failed to import f2py2e:')
    
    # Obtaining the type of the subscript
    int_90470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 57), 'int')
    
    # Call to exc_info(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_90473 = {}
    # Getting the type of 'sys' (line 43)
    sys_90471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 42), 'sys', False)
    # Obtaining the member 'exc_info' of a type (line 43)
    exc_info_90472 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 42), sys_90471, 'exc_info')
    # Calling exc_info(args, kwargs) (line 43)
    exc_info_call_result_90474 = invoke(stypy.reporting.localization.Localization(__file__, 43, 42), exc_info_90472, *[], **kwargs_90473)
    
    # Obtaining the member '__getitem__' of a type (line 43)
    getitem___90475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 42), exc_info_call_result_90474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 43)
    subscript_call_result_90476 = invoke(stypy.reporting.localization.Localization(__file__, 43, 42), getitem___90475, int_90470)
    
    # Processing the call keyword arguments (line 43)
    kwargs_90477 = {}
    # Getting the type of 'print' (line 43)
    print_90468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'print', False)
    # Calling print(args, kwargs) (line 43)
    print_call_result_90478 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), print_90468, *[str_90469, subscript_call_result_90476], **kwargs_90477)
    
    
    # Assigning a Num to a Name (line 44):
    int_90479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'int')
    # Assigning a type to the variable 'has_f2py2e' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'has_f2py2e', int_90479)
    # SSA join for try-except statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 47, 8))
    
    # 'import numpy.distutils' statement (line 47)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90480 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 47, 8), 'numpy.distutils')

    if (type(import_90480) is not StypyTypeError):

        if (import_90480 != 'pyd_module'):
            __import__(import_90480)
            sys_modules_90481 = sys.modules[import_90480]
            import_module(stypy.reporting.localization.Localization(__file__, 47, 8), 'numpy.distutils', sys_modules_90481.module_type_store, module_type_store)
        else:
            import numpy.distutils

            import_module(stypy.reporting.localization.Localization(__file__, 47, 8), 'numpy.distutils', numpy.distutils, module_type_store)

    else:
        # Assigning a type to the variable 'numpy.distutils' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'numpy.distutils', import_90480)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Num to a Name (line 48):
    int_90482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 30), 'int')
    # Assigning a type to the variable 'has_numpy_distutils' (line 48)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'has_numpy_distutils', int_90482)
    # SSA branch for the except part of a try statement (line 46)
    # SSA branch for the except 'ImportError' branch of a try statement (line 46)
    module_type_store.open_ssa_branch('except')
    
    
    # SSA begins for try-except statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 12))
    
    # 'import numpy_distutils' statement (line 51)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90483 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 51, 12), 'numpy_distutils')

    if (type(import_90483) is not StypyTypeError):

        if (import_90483 != 'pyd_module'):
            __import__(import_90483)
            sys_modules_90484 = sys.modules[import_90483]
            import_module(stypy.reporting.localization.Localization(__file__, 51, 12), 'numpy_distutils', sys_modules_90484.module_type_store, module_type_store)
        else:
            import numpy_distutils

            import_module(stypy.reporting.localization.Localization(__file__, 51, 12), 'numpy_distutils', numpy_distutils, module_type_store)

    else:
        # Assigning a type to the variable 'numpy_distutils' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 12), 'numpy_distutils', import_90483)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Assigning a Num to a Name (line 52):
    int_90485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 34), 'int')
    # Assigning a type to the variable 'has_numpy_distutils' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'has_numpy_distutils', int_90485)
    # SSA branch for the except part of a try statement (line 50)
    # SSA branch for the except 'ImportError' branch of a try statement (line 50)
    module_type_store.open_ssa_branch('except')
    
    # Call to print(...): (line 54)
    # Processing the call arguments (line 54)
    str_90487 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 18), 'str', 'Failed to import numpy_distutils:')
    
    # Obtaining the type of the subscript
    int_90488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 70), 'int')
    
    # Call to exc_info(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_90491 = {}
    # Getting the type of 'sys' (line 54)
    sys_90489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 55), 'sys', False)
    # Obtaining the member 'exc_info' of a type (line 54)
    exc_info_90490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 55), sys_90489, 'exc_info')
    # Calling exc_info(args, kwargs) (line 54)
    exc_info_call_result_90492 = invoke(stypy.reporting.localization.Localization(__file__, 54, 55), exc_info_90490, *[], **kwargs_90491)
    
    # Obtaining the member '__getitem__' of a type (line 54)
    getitem___90493 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 55), exc_info_call_result_90492, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 54)
    subscript_call_result_90494 = invoke(stypy.reporting.localization.Localization(__file__, 54, 55), getitem___90493, int_90488)
    
    # Processing the call keyword arguments (line 54)
    kwargs_90495 = {}
    # Getting the type of 'print' (line 54)
    print_90486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'print', False)
    # Calling print(args, kwargs) (line 54)
    print_call_result_90496 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), print_90486, *[str_90487, subscript_call_result_90494], **kwargs_90495)
    
    
    # Assigning a Num to a Name (line 55):
    int_90497 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 34), 'int')
    # Assigning a type to the variable 'has_numpy_distutils' (line 55)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'has_numpy_distutils', int_90497)
    # SSA join for try-except statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'has_newnumpy' (line 57)
    has_newnumpy_90498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 7), 'has_newnumpy')
    # Testing the type of an if condition (line 57)
    if_condition_90499 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 57, 4), has_newnumpy_90498)
    # Assigning a type to the variable 'if_condition_90499' (line 57)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 4), 'if_condition_90499', if_condition_90499)
    # SSA begins for if statement (line 57)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to print(...): (line 59)
    # Processing the call arguments (line 59)
    str_90501 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 18), 'str', 'Found new numpy version %r in %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 60)
    tuple_90502 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 60)
    # Adding element type (line 60)
    # Getting the type of 'numpy' (line 60)
    numpy_90503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 19), 'numpy', False)
    # Obtaining the member '__version__' of a type (line 60)
    version___90504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 19), numpy_90503, '__version__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), tuple_90502, version___90504)
    # Adding element type (line 60)
    # Getting the type of 'numpy' (line 60)
    numpy_90505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 38), 'numpy', False)
    # Obtaining the member '__file__' of a type (line 60)
    file___90506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 38), numpy_90505, '__file__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 19), tuple_90502, file___90506)
    
    # Applying the binary operator '%' (line 59)
    result_mod_90507 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 18), '%', str_90501, tuple_90502)
    
    # Processing the call keyword arguments (line 59)
    kwargs_90508 = {}
    # Getting the type of 'print' (line 59)
    print_90500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'print', False)
    # Calling print(args, kwargs) (line 59)
    print_call_result_90509 = invoke(stypy.reporting.localization.Localization(__file__, 59, 12), print_90500, *[result_mod_90507], **kwargs_90508)
    
    # SSA branch for the except part of a try statement (line 58)
    # SSA branch for the except 'Exception' branch of a try statement (line 58)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 61)
    Exception_90510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'Exception')
    # Assigning a type to the variable 'msg' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'msg', Exception_90510)
    
    # Call to print(...): (line 62)
    # Processing the call arguments (line 62)
    str_90512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 18), 'str', 'error:')
    # Getting the type of 'msg' (line 62)
    msg_90513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'msg', False)
    # Processing the call keyword arguments (line 62)
    kwargs_90514 = {}
    # Getting the type of 'print' (line 62)
    print_90511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'print', False)
    # Calling print(args, kwargs) (line 62)
    print_call_result_90515 = invoke(stypy.reporting.localization.Localization(__file__, 62, 12), print_90511, *[str_90512, msg_90513], **kwargs_90514)
    
    
    # Call to print(...): (line 63)
    # Processing the call arguments (line 63)
    str_90517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 18), 'str', '------')
    # Processing the call keyword arguments (line 63)
    kwargs_90518 = {}
    # Getting the type of 'print' (line 63)
    print_90516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 12), 'print', False)
    # Calling print(args, kwargs) (line 63)
    print_call_result_90519 = invoke(stypy.reporting.localization.Localization(__file__, 63, 12), print_90516, *[str_90517], **kwargs_90518)
    
    # SSA join for try-except statement (line 58)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 57)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'has_f2py2e' (line 65)
    has_f2py2e_90520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 7), 'has_f2py2e')
    # Testing the type of an if condition (line 65)
    if_condition_90521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 4), has_f2py2e_90520)
    # Assigning a type to the variable 'if_condition_90521' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'if_condition_90521', if_condition_90521)
    # SSA begins for if statement (line 65)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to print(...): (line 67)
    # Processing the call arguments (line 67)
    str_90523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 18), 'str', 'Found f2py2e version %r in %s')
    
    # Obtaining an instance of the builtin type 'tuple' (line 68)
    tuple_90524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 68)
    # Adding element type (line 68)
    # Getting the type of 'f2py2e' (line 68)
    f2py2e_90525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 19), 'f2py2e', False)
    # Obtaining the member '__version__' of a type (line 68)
    version___90526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), f2py2e_90525, '__version__')
    # Obtaining the member 'version' of a type (line 68)
    version_90527 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 19), version___90526, 'version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 19), tuple_90524, version_90527)
    # Adding element type (line 68)
    # Getting the type of 'f2py2e' (line 68)
    f2py2e_90528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 47), 'f2py2e', False)
    # Obtaining the member '__file__' of a type (line 68)
    file___90529 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 47), f2py2e_90528, '__file__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 19), tuple_90524, file___90529)
    
    # Applying the binary operator '%' (line 67)
    result_mod_90530 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 18), '%', str_90523, tuple_90524)
    
    # Processing the call keyword arguments (line 67)
    kwargs_90531 = {}
    # Getting the type of 'print' (line 67)
    print_90522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'print', False)
    # Calling print(args, kwargs) (line 67)
    print_call_result_90532 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), print_90522, *[result_mod_90530], **kwargs_90531)
    
    # SSA branch for the except part of a try statement (line 66)
    # SSA branch for the except 'Exception' branch of a try statement (line 66)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 69)
    Exception_90533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'Exception')
    # Assigning a type to the variable 'msg' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'msg', Exception_90533)
    
    # Call to print(...): (line 70)
    # Processing the call arguments (line 70)
    str_90535 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'str', 'error:')
    # Getting the type of 'msg' (line 70)
    msg_90536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 28), 'msg', False)
    # Processing the call keyword arguments (line 70)
    kwargs_90537 = {}
    # Getting the type of 'print' (line 70)
    print_90534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'print', False)
    # Calling print(args, kwargs) (line 70)
    print_call_result_90538 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), print_90534, *[str_90535, msg_90536], **kwargs_90537)
    
    
    # Call to print(...): (line 71)
    # Processing the call arguments (line 71)
    str_90540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 18), 'str', '------')
    # Processing the call keyword arguments (line 71)
    kwargs_90541 = {}
    # Getting the type of 'print' (line 71)
    print_90539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 12), 'print', False)
    # Calling print(args, kwargs) (line 71)
    print_call_result_90542 = invoke(stypy.reporting.localization.Localization(__file__, 71, 12), print_90539, *[str_90540], **kwargs_90541)
    
    # SSA join for try-except statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 65)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'has_numpy_distutils' (line 73)
    has_numpy_distutils_90543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 7), 'has_numpy_distutils')
    # Testing the type of an if condition (line 73)
    if_condition_90544 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 4), has_numpy_distutils_90543)
    # Assigning a type to the variable 'if_condition_90544' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 4), 'if_condition_90544', if_condition_90544)
    # SSA begins for if statement (line 73)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Getting the type of 'has_numpy_distutils' (line 75)
    has_numpy_distutils_90545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 15), 'has_numpy_distutils')
    int_90546 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 38), 'int')
    # Applying the binary operator '==' (line 75)
    result_eq_90547 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 15), '==', has_numpy_distutils_90545, int_90546)
    
    # Testing the type of an if condition (line 75)
    if_condition_90548 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 75, 12), result_eq_90547)
    # Assigning a type to the variable 'if_condition_90548' (line 75)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'if_condition_90548', if_condition_90548)
    # SSA begins for if statement (line 75)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 76)
    # Processing the call arguments (line 76)
    str_90550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'str', 'Found numpy.distutils version %r in %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 77)
    tuple_90551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 77)
    # Adding element type (line 77)
    # Getting the type of 'numpy' (line 77)
    numpy_90552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 20), 'numpy', False)
    # Obtaining the member 'distutils' of a type (line 77)
    distutils_90553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), numpy_90552, 'distutils')
    # Obtaining the member '__version__' of a type (line 77)
    version___90554 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 20), distutils_90553, '__version__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 20), tuple_90551, version___90554)
    # Adding element type (line 77)
    # Getting the type of 'numpy' (line 78)
    numpy_90555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'numpy', False)
    # Obtaining the member 'distutils' of a type (line 78)
    distutils_90556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), numpy_90555, 'distutils')
    # Obtaining the member '__file__' of a type (line 78)
    file___90557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), distutils_90556, '__file__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 20), tuple_90551, file___90557)
    
    # Applying the binary operator '%' (line 76)
    result_mod_90558 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 22), '%', str_90550, tuple_90551)
    
    # Processing the call keyword arguments (line 76)
    kwargs_90559 = {}
    # Getting the type of 'print' (line 76)
    print_90549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'print', False)
    # Calling print(args, kwargs) (line 76)
    print_call_result_90560 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), print_90549, *[result_mod_90558], **kwargs_90559)
    
    # SSA branch for the else part of an if statement (line 75)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 80)
    # Processing the call arguments (line 80)
    str_90562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'str', 'Found numpy_distutils version %r in %r')
    
    # Obtaining an instance of the builtin type 'tuple' (line 81)
    tuple_90563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 81)
    # Adding element type (line 81)
    # Getting the type of 'numpy_distutils' (line 81)
    numpy_distutils_90564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'numpy_distutils', False)
    # Obtaining the member 'numpy_distutils_version' of a type (line 81)
    numpy_distutils_version_90565 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 20), numpy_distutils_90564, 'numpy_distutils_version')
    # Obtaining the member 'numpy_distutils_version' of a type (line 81)
    numpy_distutils_version_90566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 20), numpy_distutils_version_90565, 'numpy_distutils_version')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), tuple_90563, numpy_distutils_version_90566)
    # Adding element type (line 81)
    # Getting the type of 'numpy_distutils' (line 82)
    numpy_distutils_90567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'numpy_distutils', False)
    # Obtaining the member '__file__' of a type (line 82)
    file___90568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 20), numpy_distutils_90567, '__file__')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 20), tuple_90563, file___90568)
    
    # Applying the binary operator '%' (line 80)
    result_mod_90569 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 22), '%', str_90562, tuple_90563)
    
    # Processing the call keyword arguments (line 80)
    kwargs_90570 = {}
    # Getting the type of 'print' (line 80)
    print_90561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 16), 'print', False)
    # Calling print(args, kwargs) (line 80)
    print_call_result_90571 = invoke(stypy.reporting.localization.Localization(__file__, 80, 16), print_90561, *[result_mod_90569], **kwargs_90570)
    
    # SSA join for if statement (line 75)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 83)
    # Processing the call arguments (line 83)
    str_90573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 83, 18), 'str', '------')
    # Processing the call keyword arguments (line 83)
    kwargs_90574 = {}
    # Getting the type of 'print' (line 83)
    print_90572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'print', False)
    # Calling print(args, kwargs) (line 83)
    print_call_result_90575 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), print_90572, *[str_90573], **kwargs_90574)
    
    # SSA branch for the except part of a try statement (line 74)
    # SSA branch for the except 'Exception' branch of a try statement (line 74)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 84)
    Exception_90576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'Exception')
    # Assigning a type to the variable 'msg' (line 84)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'msg', Exception_90576)
    
    # Call to print(...): (line 85)
    # Processing the call arguments (line 85)
    str_90578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, 18), 'str', 'error:')
    # Getting the type of 'msg' (line 85)
    msg_90579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 28), 'msg', False)
    # Processing the call keyword arguments (line 85)
    kwargs_90580 = {}
    # Getting the type of 'print' (line 85)
    print_90577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'print', False)
    # Calling print(args, kwargs) (line 85)
    print_call_result_90581 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), print_90577, *[str_90578, msg_90579], **kwargs_90580)
    
    
    # Call to print(...): (line 86)
    # Processing the call arguments (line 86)
    str_90583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 18), 'str', '------')
    # Processing the call keyword arguments (line 86)
    kwargs_90584 = {}
    # Getting the type of 'print' (line 86)
    print_90582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 12), 'print', False)
    # Calling print(args, kwargs) (line 86)
    print_call_result_90585 = invoke(stypy.reporting.localization.Localization(__file__, 86, 12), print_90582, *[str_90583], **kwargs_90584)
    
    # SSA join for try-except statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 87)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Getting the type of 'has_numpy_distutils' (line 88)
    has_numpy_distutils_90586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 15), 'has_numpy_distutils')
    int_90587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 38), 'int')
    # Applying the binary operator '==' (line 88)
    result_eq_90588 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 15), '==', has_numpy_distutils_90586, int_90587)
    
    # Testing the type of an if condition (line 88)
    if_condition_90589 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 88, 12), result_eq_90588)
    # Assigning a type to the variable 'if_condition_90589' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'if_condition_90589', if_condition_90589)
    # SSA begins for if statement (line 88)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 89)
    # Processing the call arguments (line 89)
    str_90591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 20), 'str', 'Importing numpy_distutils.command.build_flib ...')
    # Processing the call keyword arguments (line 89)
    str_90592 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 76), 'str', ' ')
    keyword_90593 = str_90592
    kwargs_90594 = {'end': keyword_90593}
    # Getting the type of 'print' (line 89)
    print_90590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 16), 'print', False)
    # Calling print(args, kwargs) (line 89)
    print_call_result_90595 = invoke(stypy.reporting.localization.Localization(__file__, 89, 16), print_90590, *[str_90591], **kwargs_90594)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 91, 16))
    
    # 'import numpy_distutils.command.build_flib' statement (line 91)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90596 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 91, 16), 'numpy_distutils.command.build_flib')

    if (type(import_90596) is not StypyTypeError):

        if (import_90596 != 'pyd_module'):
            __import__(import_90596)
            sys_modules_90597 = sys.modules[import_90596]
            import_module(stypy.reporting.localization.Localization(__file__, 91, 16), 'build_flib', sys_modules_90597.module_type_store, module_type_store)
        else:
            import numpy_distutils.command.build_flib as build_flib

            import_module(stypy.reporting.localization.Localization(__file__, 91, 16), 'build_flib', numpy_distutils.command.build_flib, module_type_store)

    else:
        # Assigning a type to the variable 'numpy_distutils.command.build_flib' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 16), 'numpy_distutils.command.build_flib', import_90596)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Call to print(...): (line 92)
    # Processing the call arguments (line 92)
    str_90599 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'str', 'ok')
    # Processing the call keyword arguments (line 92)
    kwargs_90600 = {}
    # Getting the type of 'print' (line 92)
    print_90598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 16), 'print', False)
    # Calling print(args, kwargs) (line 92)
    print_call_result_90601 = invoke(stypy.reporting.localization.Localization(__file__, 92, 16), print_90598, *[str_90599], **kwargs_90600)
    
    
    # Call to print(...): (line 93)
    # Processing the call arguments (line 93)
    str_90603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 22), 'str', '------')
    # Processing the call keyword arguments (line 93)
    kwargs_90604 = {}
    # Getting the type of 'print' (line 93)
    print_90602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 16), 'print', False)
    # Calling print(args, kwargs) (line 93)
    print_call_result_90605 = invoke(stypy.reporting.localization.Localization(__file__, 93, 16), print_90602, *[str_90603], **kwargs_90604)
    
    
    
    # SSA begins for try-except statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to print(...): (line 95)
    # Processing the call arguments (line 95)
    str_90607 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 96, 24), 'str', 'Checking availability of supported Fortran compilers:')
    # Processing the call keyword arguments (line 95)
    kwargs_90608 = {}
    # Getting the type of 'print' (line 95)
    print_90606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 20), 'print', False)
    # Calling print(args, kwargs) (line 95)
    print_call_result_90609 = invoke(stypy.reporting.localization.Localization(__file__, 95, 20), print_90606, *[str_90607], **kwargs_90608)
    
    
    # Getting the type of 'build_flib' (line 97)
    build_flib_90610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 42), 'build_flib')
    # Obtaining the member 'all_compilers' of a type (line 97)
    all_compilers_90611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 42), build_flib_90610, 'all_compilers')
    # Testing the type of a for loop iterable (line 97)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 97, 20), all_compilers_90611)
    # Getting the type of the for loop variable (line 97)
    for_loop_var_90612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 97, 20), all_compilers_90611)
    # Assigning a type to the variable 'compiler_class' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'compiler_class', for_loop_var_90612)
    # SSA begins for a for statement (line 97)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to is_available(...): (line 98)
    # Processing the call keyword arguments (line 98)
    kwargs_90619 = {}
    
    # Call to compiler_class(...): (line 98)
    # Processing the call keyword arguments (line 98)
    int_90614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 47), 'int')
    keyword_90615 = int_90614
    kwargs_90616 = {'verbose': keyword_90615}
    # Getting the type of 'compiler_class' (line 98)
    compiler_class_90613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 24), 'compiler_class', False)
    # Calling compiler_class(args, kwargs) (line 98)
    compiler_class_call_result_90617 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), compiler_class_90613, *[], **kwargs_90616)
    
    # Obtaining the member 'is_available' of a type (line 98)
    is_available_90618 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 24), compiler_class_call_result_90617, 'is_available')
    # Calling is_available(args, kwargs) (line 98)
    is_available_call_result_90620 = invoke(stypy.reporting.localization.Localization(__file__, 98, 24), is_available_90618, *[], **kwargs_90619)
    
    
    # Call to print(...): (line 99)
    # Processing the call arguments (line 99)
    str_90622 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 30), 'str', '------')
    # Processing the call keyword arguments (line 99)
    kwargs_90623 = {}
    # Getting the type of 'print' (line 99)
    print_90621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 24), 'print', False)
    # Calling print(args, kwargs) (line 99)
    print_call_result_90624 = invoke(stypy.reporting.localization.Localization(__file__, 99, 24), print_90621, *[str_90622], **kwargs_90623)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 94)
    # SSA branch for the except 'Exception' branch of a try statement (line 94)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 100)
    Exception_90625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 23), 'Exception')
    # Assigning a type to the variable 'msg' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'msg', Exception_90625)
    
    # Call to print(...): (line 101)
    # Processing the call arguments (line 101)
    str_90627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'str', 'error:')
    # Getting the type of 'msg' (line 101)
    msg_90628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 36), 'msg', False)
    # Processing the call keyword arguments (line 101)
    kwargs_90629 = {}
    # Getting the type of 'print' (line 101)
    print_90626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'print', False)
    # Calling print(args, kwargs) (line 101)
    print_call_result_90630 = invoke(stypy.reporting.localization.Localization(__file__, 101, 20), print_90626, *[str_90627, msg_90628], **kwargs_90629)
    
    
    # Call to print(...): (line 102)
    # Processing the call arguments (line 102)
    str_90632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 26), 'str', '------')
    # Processing the call keyword arguments (line 102)
    kwargs_90633 = {}
    # Getting the type of 'print' (line 102)
    print_90631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'print', False)
    # Calling print(args, kwargs) (line 102)
    print_call_result_90634 = invoke(stypy.reporting.localization.Localization(__file__, 102, 20), print_90631, *[str_90632], **kwargs_90633)
    
    # SSA join for try-except statement (line 94)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 88)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 87)
    # SSA branch for the except 'Exception' branch of a try statement (line 87)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 103)
    Exception_90635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 15), 'Exception')
    # Assigning a type to the variable 'msg' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'msg', Exception_90635)
    
    # Call to print(...): (line 104)
    # Processing the call arguments (line 104)
    str_90637 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 16), 'str', 'error:')
    # Getting the type of 'msg' (line 105)
    msg_90638 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'msg', False)
    str_90639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 31), 'str', '(ignore it, build_flib is obsolute for numpy.distutils 0.2.2 and up)')
    # Processing the call keyword arguments (line 104)
    kwargs_90640 = {}
    # Getting the type of 'print' (line 104)
    print_90636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'print', False)
    # Calling print(args, kwargs) (line 104)
    print_call_result_90641 = invoke(stypy.reporting.localization.Localization(__file__, 104, 12), print_90636, *[str_90637, msg_90638, str_90639], **kwargs_90640)
    
    
    # Call to print(...): (line 106)
    # Processing the call arguments (line 106)
    str_90643 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 18), 'str', '------')
    # Processing the call keyword arguments (line 106)
    kwargs_90644 = {}
    # Getting the type of 'print' (line 106)
    print_90642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'print', False)
    # Calling print(args, kwargs) (line 106)
    print_call_result_90645 = invoke(stypy.reporting.localization.Localization(__file__, 106, 12), print_90642, *[str_90643], **kwargs_90644)
    
    # SSA join for try-except statement (line 87)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 107)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Getting the type of 'has_numpy_distutils' (line 108)
    has_numpy_distutils_90646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 15), 'has_numpy_distutils')
    int_90647 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 38), 'int')
    # Applying the binary operator '==' (line 108)
    result_eq_90648 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 15), '==', has_numpy_distutils_90646, int_90647)
    
    # Testing the type of an if condition (line 108)
    if_condition_90649 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 12), result_eq_90648)
    # Assigning a type to the variable 'if_condition_90649' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'if_condition_90649', if_condition_90649)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 109)
    # Processing the call arguments (line 109)
    str_90651 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'str', 'Importing numpy.distutils.fcompiler ...')
    # Processing the call keyword arguments (line 109)
    str_90652 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 69), 'str', ' ')
    keyword_90653 = str_90652
    kwargs_90654 = {'end': keyword_90653}
    # Getting the type of 'print' (line 109)
    print_90650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'print', False)
    # Calling print(args, kwargs) (line 109)
    print_call_result_90655 = invoke(stypy.reporting.localization.Localization(__file__, 109, 16), print_90650, *[str_90651], **kwargs_90654)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 110, 16))
    
    # 'import numpy.distutils.fcompiler' statement (line 110)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90656 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 110, 16), 'numpy.distutils.fcompiler')

    if (type(import_90656) is not StypyTypeError):

        if (import_90656 != 'pyd_module'):
            __import__(import_90656)
            sys_modules_90657 = sys.modules[import_90656]
            import_module(stypy.reporting.localization.Localization(__file__, 110, 16), 'fcompiler', sys_modules_90657.module_type_store, module_type_store)
        else:
            import numpy.distutils.fcompiler as fcompiler

            import_module(stypy.reporting.localization.Localization(__file__, 110, 16), 'fcompiler', numpy.distutils.fcompiler, module_type_store)

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'numpy.distutils.fcompiler', import_90656)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    # SSA branch for the else part of an if statement (line 108)
    module_type_store.open_ssa_branch('else')
    
    # Call to print(...): (line 112)
    # Processing the call arguments (line 112)
    str_90659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 22), 'str', 'Importing numpy_distutils.fcompiler ...')
    # Processing the call keyword arguments (line 112)
    str_90660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 69), 'str', ' ')
    keyword_90661 = str_90660
    kwargs_90662 = {'end': keyword_90661}
    # Getting the type of 'print' (line 112)
    print_90658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'print', False)
    # Calling print(args, kwargs) (line 112)
    print_call_result_90663 = invoke(stypy.reporting.localization.Localization(__file__, 112, 16), print_90658, *[str_90659], **kwargs_90662)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 113, 16))
    
    # 'import numpy_distutils.fcompiler' statement (line 113)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90664 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 113, 16), 'numpy_distutils.fcompiler')

    if (type(import_90664) is not StypyTypeError):

        if (import_90664 != 'pyd_module'):
            __import__(import_90664)
            sys_modules_90665 = sys.modules[import_90664]
            import_module(stypy.reporting.localization.Localization(__file__, 113, 16), 'fcompiler', sys_modules_90665.module_type_store, module_type_store)
        else:
            import numpy_distutils.fcompiler as fcompiler

            import_module(stypy.reporting.localization.Localization(__file__, 113, 16), 'fcompiler', numpy_distutils.fcompiler, module_type_store)

    else:
        # Assigning a type to the variable 'numpy_distutils.fcompiler' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 16), 'numpy_distutils.fcompiler', import_90664)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 114)
    # Processing the call arguments (line 114)
    str_90667 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 18), 'str', 'ok')
    # Processing the call keyword arguments (line 114)
    kwargs_90668 = {}
    # Getting the type of 'print' (line 114)
    print_90666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'print', False)
    # Calling print(args, kwargs) (line 114)
    print_call_result_90669 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), print_90666, *[str_90667], **kwargs_90668)
    
    
    # Call to print(...): (line 115)
    # Processing the call arguments (line 115)
    str_90671 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 18), 'str', '------')
    # Processing the call keyword arguments (line 115)
    kwargs_90672 = {}
    # Getting the type of 'print' (line 115)
    print_90670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'print', False)
    # Calling print(args, kwargs) (line 115)
    print_call_result_90673 = invoke(stypy.reporting.localization.Localization(__file__, 115, 12), print_90670, *[str_90671], **kwargs_90672)
    
    
    
    # SSA begins for try-except statement (line 116)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to print(...): (line 117)
    # Processing the call arguments (line 117)
    str_90675 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 22), 'str', 'Checking availability of supported Fortran compilers:')
    # Processing the call keyword arguments (line 117)
    kwargs_90676 = {}
    # Getting the type of 'print' (line 117)
    print_90674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 16), 'print', False)
    # Calling print(args, kwargs) (line 117)
    print_call_result_90677 = invoke(stypy.reporting.localization.Localization(__file__, 117, 16), print_90674, *[str_90675], **kwargs_90676)
    
    
    # Call to show_fcompilers(...): (line 118)
    # Processing the call keyword arguments (line 118)
    kwargs_90680 = {}
    # Getting the type of 'fcompiler' (line 118)
    fcompiler_90678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 16), 'fcompiler', False)
    # Obtaining the member 'show_fcompilers' of a type (line 118)
    show_fcompilers_90679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 16), fcompiler_90678, 'show_fcompilers')
    # Calling show_fcompilers(args, kwargs) (line 118)
    show_fcompilers_call_result_90681 = invoke(stypy.reporting.localization.Localization(__file__, 118, 16), show_fcompilers_90679, *[], **kwargs_90680)
    
    
    # Call to print(...): (line 119)
    # Processing the call arguments (line 119)
    str_90683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 22), 'str', '------')
    # Processing the call keyword arguments (line 119)
    kwargs_90684 = {}
    # Getting the type of 'print' (line 119)
    print_90682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'print', False)
    # Calling print(args, kwargs) (line 119)
    print_call_result_90685 = invoke(stypy.reporting.localization.Localization(__file__, 119, 16), print_90682, *[str_90683], **kwargs_90684)
    
    # SSA branch for the except part of a try statement (line 116)
    # SSA branch for the except 'Exception' branch of a try statement (line 116)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 120)
    Exception_90686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 19), 'Exception')
    # Assigning a type to the variable 'msg' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'msg', Exception_90686)
    
    # Call to print(...): (line 121)
    # Processing the call arguments (line 121)
    str_90688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 22), 'str', 'error:')
    # Getting the type of 'msg' (line 121)
    msg_90689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 32), 'msg', False)
    # Processing the call keyword arguments (line 121)
    kwargs_90690 = {}
    # Getting the type of 'print' (line 121)
    print_90687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 16), 'print', False)
    # Calling print(args, kwargs) (line 121)
    print_call_result_90691 = invoke(stypy.reporting.localization.Localization(__file__, 121, 16), print_90687, *[str_90688, msg_90689], **kwargs_90690)
    
    
    # Call to print(...): (line 122)
    # Processing the call arguments (line 122)
    str_90693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 22), 'str', '------')
    # Processing the call keyword arguments (line 122)
    kwargs_90694 = {}
    # Getting the type of 'print' (line 122)
    print_90692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'print', False)
    # Calling print(args, kwargs) (line 122)
    print_call_result_90695 = invoke(stypy.reporting.localization.Localization(__file__, 122, 16), print_90692, *[str_90693], **kwargs_90694)
    
    # SSA join for try-except statement (line 116)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 107)
    # SSA branch for the except 'Exception' branch of a try statement (line 107)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 123)
    Exception_90696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 15), 'Exception')
    # Assigning a type to the variable 'msg' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'msg', Exception_90696)
    
    # Call to print(...): (line 124)
    # Processing the call arguments (line 124)
    str_90698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 18), 'str', 'error:')
    # Getting the type of 'msg' (line 124)
    msg_90699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 28), 'msg', False)
    # Processing the call keyword arguments (line 124)
    kwargs_90700 = {}
    # Getting the type of 'print' (line 124)
    print_90697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'print', False)
    # Calling print(args, kwargs) (line 124)
    print_call_result_90701 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), print_90697, *[str_90698, msg_90699], **kwargs_90700)
    
    
    # Call to print(...): (line 125)
    # Processing the call arguments (line 125)
    str_90703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 125, 18), 'str', '------')
    # Processing the call keyword arguments (line 125)
    kwargs_90704 = {}
    # Getting the type of 'print' (line 125)
    print_90702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 12), 'print', False)
    # Calling print(args, kwargs) (line 125)
    print_call_result_90705 = invoke(stypy.reporting.localization.Localization(__file__, 125, 12), print_90702, *[str_90703], **kwargs_90704)
    
    # SSA join for try-except statement (line 107)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    
    # Getting the type of 'has_numpy_distutils' (line 127)
    has_numpy_distutils_90706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'has_numpy_distutils')
    int_90707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 38), 'int')
    # Applying the binary operator '==' (line 127)
    result_eq_90708 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 15), '==', has_numpy_distutils_90706, int_90707)
    
    # Testing the type of an if condition (line 127)
    if_condition_90709 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), result_eq_90708)
    # Assigning a type to the variable 'if_condition_90709' (line 127)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_90709', if_condition_90709)
    # SSA begins for if statement (line 127)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 128)
    # Processing the call arguments (line 128)
    str_90711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 22), 'str', 'Importing numpy.distutils.cpuinfo ...')
    # Processing the call keyword arguments (line 128)
    str_90712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 67), 'str', ' ')
    keyword_90713 = str_90712
    kwargs_90714 = {'end': keyword_90713}
    # Getting the type of 'print' (line 128)
    print_90710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 16), 'print', False)
    # Calling print(args, kwargs) (line 128)
    print_call_result_90715 = invoke(stypy.reporting.localization.Localization(__file__, 128, 16), print_90710, *[str_90711], **kwargs_90714)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 129, 16))
    
    # 'from numpy.distutils.cpuinfo import cpuinfo' statement (line 129)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90716 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 129, 16), 'numpy.distutils.cpuinfo')

    if (type(import_90716) is not StypyTypeError):

        if (import_90716 != 'pyd_module'):
            __import__(import_90716)
            sys_modules_90717 = sys.modules[import_90716]
            import_from_module(stypy.reporting.localization.Localization(__file__, 129, 16), 'numpy.distutils.cpuinfo', sys_modules_90717.module_type_store, module_type_store, ['cpuinfo'])
            nest_module(stypy.reporting.localization.Localization(__file__, 129, 16), __file__, sys_modules_90717, sys_modules_90717.module_type_store, module_type_store)
        else:
            from numpy.distutils.cpuinfo import cpuinfo

            import_from_module(stypy.reporting.localization.Localization(__file__, 129, 16), 'numpy.distutils.cpuinfo', None, module_type_store, ['cpuinfo'], [cpuinfo])

    else:
        # Assigning a type to the variable 'numpy.distutils.cpuinfo' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'numpy.distutils.cpuinfo', import_90716)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Call to print(...): (line 130)
    # Processing the call arguments (line 130)
    str_90719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 22), 'str', 'ok')
    # Processing the call keyword arguments (line 130)
    kwargs_90720 = {}
    # Getting the type of 'print' (line 130)
    print_90718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'print', False)
    # Calling print(args, kwargs) (line 130)
    print_call_result_90721 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), print_90718, *[str_90719], **kwargs_90720)
    
    
    # Call to print(...): (line 131)
    # Processing the call arguments (line 131)
    str_90723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 22), 'str', '------')
    # Processing the call keyword arguments (line 131)
    kwargs_90724 = {}
    # Getting the type of 'print' (line 131)
    print_90722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 16), 'print', False)
    # Calling print(args, kwargs) (line 131)
    print_call_result_90725 = invoke(stypy.reporting.localization.Localization(__file__, 131, 16), print_90722, *[str_90723], **kwargs_90724)
    
    # SSA branch for the else part of an if statement (line 127)
    module_type_store.open_ssa_branch('else')
    
    
    # SSA begins for try-except statement (line 133)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to print(...): (line 134)
    # Processing the call arguments (line 134)
    str_90727 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 24), 'str', 'Importing numpy_distutils.command.cpuinfo ...')
    # Processing the call keyword arguments (line 134)
    str_90728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 77), 'str', ' ')
    keyword_90729 = str_90728
    kwargs_90730 = {'end': keyword_90729}
    # Getting the type of 'print' (line 134)
    print_90726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 20), 'print', False)
    # Calling print(args, kwargs) (line 134)
    print_call_result_90731 = invoke(stypy.reporting.localization.Localization(__file__, 134, 20), print_90726, *[str_90727], **kwargs_90730)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 136, 20))
    
    # 'from numpy_distutils.command.cpuinfo import cpuinfo' statement (line 136)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90732 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 136, 20), 'numpy_distutils.command.cpuinfo')

    if (type(import_90732) is not StypyTypeError):

        if (import_90732 != 'pyd_module'):
            __import__(import_90732)
            sys_modules_90733 = sys.modules[import_90732]
            import_from_module(stypy.reporting.localization.Localization(__file__, 136, 20), 'numpy_distutils.command.cpuinfo', sys_modules_90733.module_type_store, module_type_store, ['cpuinfo'])
            nest_module(stypy.reporting.localization.Localization(__file__, 136, 20), __file__, sys_modules_90733, sys_modules_90733.module_type_store, module_type_store)
        else:
            from numpy_distutils.command.cpuinfo import cpuinfo

            import_from_module(stypy.reporting.localization.Localization(__file__, 136, 20), 'numpy_distutils.command.cpuinfo', None, module_type_store, ['cpuinfo'], [cpuinfo])

    else:
        # Assigning a type to the variable 'numpy_distutils.command.cpuinfo' (line 136)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 20), 'numpy_distutils.command.cpuinfo', import_90732)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Call to print(...): (line 137)
    # Processing the call arguments (line 137)
    str_90735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 137, 26), 'str', 'ok')
    # Processing the call keyword arguments (line 137)
    kwargs_90736 = {}
    # Getting the type of 'print' (line 137)
    print_90734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 20), 'print', False)
    # Calling print(args, kwargs) (line 137)
    print_call_result_90737 = invoke(stypy.reporting.localization.Localization(__file__, 137, 20), print_90734, *[str_90735], **kwargs_90736)
    
    
    # Call to print(...): (line 138)
    # Processing the call arguments (line 138)
    str_90739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 26), 'str', '------')
    # Processing the call keyword arguments (line 138)
    kwargs_90740 = {}
    # Getting the type of 'print' (line 138)
    print_90738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 20), 'print', False)
    # Calling print(args, kwargs) (line 138)
    print_call_result_90741 = invoke(stypy.reporting.localization.Localization(__file__, 138, 20), print_90738, *[str_90739], **kwargs_90740)
    
    # SSA branch for the except part of a try statement (line 133)
    # SSA branch for the except 'Exception' branch of a try statement (line 133)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 139)
    Exception_90742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 23), 'Exception')
    # Assigning a type to the variable 'msg' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'msg', Exception_90742)
    
    # Call to print(...): (line 140)
    # Processing the call arguments (line 140)
    str_90744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 26), 'str', 'error:')
    # Getting the type of 'msg' (line 140)
    msg_90745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 36), 'msg', False)
    str_90746 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 140, 41), 'str', '(ignore it)')
    # Processing the call keyword arguments (line 140)
    kwargs_90747 = {}
    # Getting the type of 'print' (line 140)
    print_90743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 20), 'print', False)
    # Calling print(args, kwargs) (line 140)
    print_call_result_90748 = invoke(stypy.reporting.localization.Localization(__file__, 140, 20), print_90743, *[str_90744, msg_90745, str_90746], **kwargs_90747)
    
    
    # Call to print(...): (line 141)
    # Processing the call arguments (line 141)
    str_90750 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 26), 'str', 'Importing numpy_distutils.cpuinfo ...')
    # Processing the call keyword arguments (line 141)
    str_90751 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 71), 'str', ' ')
    keyword_90752 = str_90751
    kwargs_90753 = {'end': keyword_90752}
    # Getting the type of 'print' (line 141)
    print_90749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 20), 'print', False)
    # Calling print(args, kwargs) (line 141)
    print_call_result_90754 = invoke(stypy.reporting.localization.Localization(__file__, 141, 20), print_90749, *[str_90750], **kwargs_90753)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 142, 20))
    
    # 'from numpy_distutils.cpuinfo import cpuinfo' statement (line 142)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/f2py/')
    import_90755 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 142, 20), 'numpy_distutils.cpuinfo')

    if (type(import_90755) is not StypyTypeError):

        if (import_90755 != 'pyd_module'):
            __import__(import_90755)
            sys_modules_90756 = sys.modules[import_90755]
            import_from_module(stypy.reporting.localization.Localization(__file__, 142, 20), 'numpy_distutils.cpuinfo', sys_modules_90756.module_type_store, module_type_store, ['cpuinfo'])
            nest_module(stypy.reporting.localization.Localization(__file__, 142, 20), __file__, sys_modules_90756, sys_modules_90756.module_type_store, module_type_store)
        else:
            from numpy_distutils.cpuinfo import cpuinfo

            import_from_module(stypy.reporting.localization.Localization(__file__, 142, 20), 'numpy_distutils.cpuinfo', None, module_type_store, ['cpuinfo'], [cpuinfo])

    else:
        # Assigning a type to the variable 'numpy_distutils.cpuinfo' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 20), 'numpy_distutils.cpuinfo', import_90755)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/f2py/')
    
    
    # Call to print(...): (line 143)
    # Processing the call arguments (line 143)
    str_90758 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 26), 'str', 'ok')
    # Processing the call keyword arguments (line 143)
    kwargs_90759 = {}
    # Getting the type of 'print' (line 143)
    print_90757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'print', False)
    # Calling print(args, kwargs) (line 143)
    print_call_result_90760 = invoke(stypy.reporting.localization.Localization(__file__, 143, 20), print_90757, *[str_90758], **kwargs_90759)
    
    
    # Call to print(...): (line 144)
    # Processing the call arguments (line 144)
    str_90762 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 26), 'str', '------')
    # Processing the call keyword arguments (line 144)
    kwargs_90763 = {}
    # Getting the type of 'print' (line 144)
    print_90761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 20), 'print', False)
    # Calling print(args, kwargs) (line 144)
    print_call_result_90764 = invoke(stypy.reporting.localization.Localization(__file__, 144, 20), print_90761, *[str_90762], **kwargs_90763)
    
    # SSA join for try-except statement (line 133)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 127)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 145):
    
    # Call to cpuinfo(...): (line 145)
    # Processing the call keyword arguments (line 145)
    kwargs_90766 = {}
    # Getting the type of 'cpuinfo' (line 145)
    cpuinfo_90765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 18), 'cpuinfo', False)
    # Calling cpuinfo(args, kwargs) (line 145)
    cpuinfo_call_result_90767 = invoke(stypy.reporting.localization.Localization(__file__, 145, 18), cpuinfo_90765, *[], **kwargs_90766)
    
    # Assigning a type to the variable 'cpu' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'cpu', cpuinfo_call_result_90767)
    
    # Call to print(...): (line 146)
    # Processing the call arguments (line 146)
    str_90769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 18), 'str', 'CPU information:')
    # Processing the call keyword arguments (line 146)
    str_90770 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 146, 42), 'str', ' ')
    keyword_90771 = str_90770
    kwargs_90772 = {'end': keyword_90771}
    # Getting the type of 'print' (line 146)
    print_90768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'print', False)
    # Calling print(args, kwargs) (line 146)
    print_call_result_90773 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), print_90768, *[str_90769], **kwargs_90772)
    
    
    
    # Call to dir(...): (line 147)
    # Processing the call arguments (line 147)
    # Getting the type of 'cpuinfo' (line 147)
    cpuinfo_90775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 28), 'cpuinfo', False)
    # Processing the call keyword arguments (line 147)
    kwargs_90776 = {}
    # Getting the type of 'dir' (line 147)
    dir_90774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 24), 'dir', False)
    # Calling dir(args, kwargs) (line 147)
    dir_call_result_90777 = invoke(stypy.reporting.localization.Localization(__file__, 147, 24), dir_90774, *[cpuinfo_90775], **kwargs_90776)
    
    # Testing the type of a for loop iterable (line 147)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 147, 12), dir_call_result_90777)
    # Getting the type of the for loop variable (line 147)
    for_loop_var_90778 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 147, 12), dir_call_result_90777)
    # Assigning a type to the variable 'name' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'name', for_loop_var_90778)
    # SSA begins for a for statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Evaluating a boolean operation
    
    
    # Obtaining the type of the subscript
    int_90779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 24), 'int')
    # Getting the type of 'name' (line 148)
    name_90780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 19), 'name')
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___90781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 19), name_90780, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_90782 = invoke(stypy.reporting.localization.Localization(__file__, 148, 19), getitem___90781, int_90779)
    
    str_90783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 30), 'str', '_')
    # Applying the binary operator '==' (line 148)
    result_eq_90784 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 19), '==', subscript_call_result_90782, str_90783)
    
    
    
    # Obtaining the type of the subscript
    int_90785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 43), 'int')
    # Getting the type of 'name' (line 148)
    name_90786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 38), 'name')
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___90787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 38), name_90786, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_90788 = invoke(stypy.reporting.localization.Localization(__file__, 148, 38), getitem___90787, int_90785)
    
    str_90789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 49), 'str', '_')
    # Applying the binary operator '!=' (line 148)
    result_ne_90790 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 38), '!=', subscript_call_result_90788, str_90789)
    
    # Applying the binary operator 'and' (line 148)
    result_and_keyword_90791 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 19), 'and', result_eq_90784, result_ne_90790)
    
    # Call to (...): (line 148)
    # Processing the call keyword arguments (line 148)
    kwargs_90801 = {}
    
    # Call to getattr(...): (line 148)
    # Processing the call arguments (line 148)
    # Getting the type of 'cpu' (line 148)
    cpu_90793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 65), 'cpu', False)
    
    # Obtaining the type of the subscript
    int_90794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 148, 75), 'int')
    slice_90795 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 148, 70), int_90794, None, None)
    # Getting the type of 'name' (line 148)
    name_90796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 70), 'name', False)
    # Obtaining the member '__getitem__' of a type (line 148)
    getitem___90797 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 148, 70), name_90796, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 148)
    subscript_call_result_90798 = invoke(stypy.reporting.localization.Localization(__file__, 148, 70), getitem___90797, slice_90795)
    
    # Processing the call keyword arguments (line 148)
    kwargs_90799 = {}
    # Getting the type of 'getattr' (line 148)
    getattr_90792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 57), 'getattr', False)
    # Calling getattr(args, kwargs) (line 148)
    getattr_call_result_90800 = invoke(stypy.reporting.localization.Localization(__file__, 148, 57), getattr_90792, *[cpu_90793, subscript_call_result_90798], **kwargs_90799)
    
    # Calling (args, kwargs) (line 148)
    _call_result_90802 = invoke(stypy.reporting.localization.Localization(__file__, 148, 57), getattr_call_result_90800, *[], **kwargs_90801)
    
    # Applying the binary operator 'and' (line 148)
    result_and_keyword_90803 = python_operator(stypy.reporting.localization.Localization(__file__, 148, 19), 'and', result_and_keyword_90791, _call_result_90802)
    
    # Testing the type of an if condition (line 148)
    if_condition_90804 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 148, 16), result_and_keyword_90803)
    # Assigning a type to the variable 'if_condition_90804' (line 148)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'if_condition_90804', if_condition_90804)
    # SSA begins for if statement (line 148)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to print(...): (line 149)
    # Processing the call arguments (line 149)
    
    # Obtaining the type of the subscript
    int_90806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 31), 'int')
    slice_90807 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 149, 26), int_90806, None, None)
    # Getting the type of 'name' (line 149)
    name_90808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 26), 'name', False)
    # Obtaining the member '__getitem__' of a type (line 149)
    getitem___90809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 26), name_90808, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 149)
    subscript_call_result_90810 = invoke(stypy.reporting.localization.Localization(__file__, 149, 26), getitem___90809, slice_90807)
    
    # Processing the call keyword arguments (line 149)
    str_90811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 40), 'str', ' ')
    keyword_90812 = str_90811
    kwargs_90813 = {'end': keyword_90812}
    # Getting the type of 'print' (line 149)
    print_90805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 20), 'print', False)
    # Calling print(args, kwargs) (line 149)
    print_call_result_90814 = invoke(stypy.reporting.localization.Localization(__file__, 149, 20), print_90805, *[subscript_call_result_90810], **kwargs_90813)
    
    # SSA join for if statement (line 148)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to print(...): (line 150)
    # Processing the call arguments (line 150)
    str_90816 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 150, 18), 'str', '------')
    # Processing the call keyword arguments (line 150)
    kwargs_90817 = {}
    # Getting the type of 'print' (line 150)
    print_90815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'print', False)
    # Calling print(args, kwargs) (line 150)
    print_call_result_90818 = invoke(stypy.reporting.localization.Localization(__file__, 150, 12), print_90815, *[str_90816], **kwargs_90817)
    
    # SSA branch for the except part of a try statement (line 126)
    # SSA branch for the except 'Exception' branch of a try statement (line 126)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'Exception' (line 151)
    Exception_90819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'Exception')
    # Assigning a type to the variable 'msg' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'msg', Exception_90819)
    
    # Call to print(...): (line 152)
    # Processing the call arguments (line 152)
    str_90821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 18), 'str', 'error:')
    # Getting the type of 'msg' (line 152)
    msg_90822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 28), 'msg', False)
    # Processing the call keyword arguments (line 152)
    kwargs_90823 = {}
    # Getting the type of 'print' (line 152)
    print_90820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'print', False)
    # Calling print(args, kwargs) (line 152)
    print_call_result_90824 = invoke(stypy.reporting.localization.Localization(__file__, 152, 12), print_90820, *[str_90821, msg_90822], **kwargs_90823)
    
    
    # Call to print(...): (line 153)
    # Processing the call arguments (line 153)
    str_90826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 18), 'str', '------')
    # Processing the call keyword arguments (line 153)
    kwargs_90827 = {}
    # Getting the type of 'print' (line 153)
    print_90825 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'print', False)
    # Calling print(args, kwargs) (line 153)
    print_call_result_90828 = invoke(stypy.reporting.localization.Localization(__file__, 153, 12), print_90825, *[str_90826], **kwargs_90827)
    
    # SSA join for try-except statement (line 126)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 73)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to chdir(...): (line 154)
    # Processing the call arguments (line 154)
    # Getting the type of '_path' (line 154)
    _path_90831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 13), '_path', False)
    # Processing the call keyword arguments (line 154)
    kwargs_90832 = {}
    # Getting the type of 'os' (line 154)
    os_90829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 4), 'os', False)
    # Obtaining the member 'chdir' of a type (line 154)
    chdir_90830 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 4), os_90829, 'chdir')
    # Calling chdir(args, kwargs) (line 154)
    chdir_call_result_90833 = invoke(stypy.reporting.localization.Localization(__file__, 154, 4), chdir_90830, *[_path_90831], **kwargs_90832)
    
    
    # ################# End of 'run(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_90834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_90834)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run'
    return stypy_return_type_90834

# Assigning a type to the variable 'run' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'run', run)

if (__name__ == '__main__'):
    
    # Call to run(...): (line 156)
    # Processing the call keyword arguments (line 156)
    kwargs_90836 = {}
    # Getting the type of 'run' (line 156)
    run_90835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'run', False)
    # Calling run(args, kwargs) (line 156)
    run_call_result_90837 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), run_90835, *[], **kwargs_90836)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
