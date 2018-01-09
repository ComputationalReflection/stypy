
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''Support code for distutils test cases.'''
2: import os
3: import sys
4: import shutil
5: import tempfile
6: import unittest
7: import sysconfig
8: from copy import deepcopy
9: import warnings
10: 
11: from distutils import log
12: from distutils.log import DEBUG, INFO, WARN, ERROR, FATAL
13: from distutils.core import Distribution
14: 
15: 
16: def capture_warnings(func):
17:     def _capture_warnings(*args, **kw):
18:         with warnings.catch_warnings():
19:             warnings.simplefilter("ignore")
20:             return func(*args, **kw)
21:     return _capture_warnings
22: 
23: 
24: class LoggingSilencer(object):
25: 
26:     def setUp(self):
27:         super(LoggingSilencer, self).setUp()
28:         self.threshold = log.set_threshold(log.FATAL)
29:         # catching warnings
30:         # when log will be replaced by logging
31:         # we won't need such monkey-patch anymore
32:         self._old_log = log.Log._log
33:         log.Log._log = self._log
34:         self.logs = []
35: 
36:     def tearDown(self):
37:         log.set_threshold(self.threshold)
38:         log.Log._log = self._old_log
39:         super(LoggingSilencer, self).tearDown()
40: 
41:     def _log(self, level, msg, args):
42:         if level not in (DEBUG, INFO, WARN, ERROR, FATAL):
43:             raise ValueError('%s wrong log level' % str(level))
44:         self.logs.append((level, msg, args))
45: 
46:     def get_logs(self, *levels):
47:         def _format(msg, args):
48:             if len(args) == 0:
49:                 return msg
50:             return msg % args
51:         return [_format(msg, args) for level, msg, args
52:                 in self.logs if level in levels]
53: 
54:     def clear_logs(self):
55:         self.logs = []
56: 
57: 
58: class TempdirManager(object):
59:     '''Mix-in class that handles temporary directories for test cases.
60: 
61:     This is intended to be used with unittest.TestCase.
62:     '''
63: 
64:     def setUp(self):
65:         super(TempdirManager, self).setUp()
66:         self.old_cwd = os.getcwd()
67:         self.tempdirs = []
68: 
69:     def tearDown(self):
70:         # Restore working dir, for Solaris and derivatives, where rmdir()
71:         # on the current directory fails.
72:         os.chdir(self.old_cwd)
73:         super(TempdirManager, self).tearDown()
74:         while self.tempdirs:
75:             d = self.tempdirs.pop()
76:             shutil.rmtree(d, os.name in ('nt', 'cygwin'))
77: 
78:     def mkdtemp(self):
79:         '''Create a temporary directory that will be cleaned up.
80: 
81:         Returns the path of the directory.
82:         '''
83:         d = tempfile.mkdtemp()
84:         self.tempdirs.append(d)
85:         return d
86: 
87:     def write_file(self, path, content='xxx'):
88:         '''Writes a file in the given path.
89: 
90: 
91:         path can be a string or a sequence.
92:         '''
93:         if isinstance(path, (list, tuple)):
94:             path = os.path.join(*path)
95:         f = open(path, 'w')
96:         try:
97:             f.write(content)
98:         finally:
99:             f.close()
100: 
101:     def create_dist(self, pkg_name='foo', **kw):
102:         '''Will generate a test environment.
103: 
104:         This function creates:
105:          - a Distribution instance using keywords
106:          - a temporary directory with a package structure
107: 
108:         It returns the package directory and the distribution
109:         instance.
110:         '''
111:         tmp_dir = self.mkdtemp()
112:         pkg_dir = os.path.join(tmp_dir, pkg_name)
113:         os.mkdir(pkg_dir)
114:         dist = Distribution(attrs=kw)
115: 
116:         return pkg_dir, dist
117: 
118: 
119: class DummyCommand:
120:     '''Class to store options for retrieval via set_undefined_options().'''
121: 
122:     def __init__(self, **kwargs):
123:         for kw, val in kwargs.items():
124:             setattr(self, kw, val)
125: 
126:     def ensure_finalized(self):
127:         pass
128: 
129: 
130: class EnvironGuard(object):
131: 
132:     def setUp(self):
133:         super(EnvironGuard, self).setUp()
134:         self.old_environ = deepcopy(os.environ)
135: 
136:     def tearDown(self):
137:         for key, value in self.old_environ.items():
138:             if os.environ.get(key) != value:
139:                 os.environ[key] = value
140: 
141:         for key in os.environ.keys():
142:             if key not in self.old_environ:
143:                 del os.environ[key]
144: 
145:         super(EnvironGuard, self).tearDown()
146: 
147: 
148: def copy_xxmodule_c(directory):
149:     '''Helper for tests that need the xxmodule.c source file.
150: 
151:     Example use:
152: 
153:         def test_compile(self):
154:             copy_xxmodule_c(self.tmpdir)
155:             self.assertIn('xxmodule.c', os.listdir(self.tmpdir))
156: 
157:     If the source file can be found, it will be copied to *directory*.  If not,
158:     the test will be skipped.  Errors during copy are not caught.
159:     '''
160:     filename = _get_xxmodule_path()
161:     if filename is None:
162:         raise unittest.SkipTest('cannot find xxmodule.c (test must run in '
163:                                 'the python build dir)')
164:     shutil.copy(filename, directory)
165: 
166: 
167: def _get_xxmodule_path():
168:     # FIXME when run from regrtest, srcdir seems to be '.', which does not help
169:     # us find the xxmodule.c file
170:     srcdir = sysconfig.get_config_var('srcdir')
171:     candidates = [
172:         # use installed copy if available
173:         os.path.join(os.path.dirname(__file__), 'xxmodule.c'),
174:         # otherwise try using copy from build directory
175:         os.path.join(srcdir, 'Modules', 'xxmodule.c'),
176:         # srcdir mysteriously can be $srcdir/Lib/distutils/tests when
177:         # this file is run from its parent directory, so walk up the
178:         # tree to find the real srcdir
179:         os.path.join(srcdir, '..', '..', '..', 'Modules', 'xxmodule.c'),
180:     ]
181:     for path in candidates:
182:         if os.path.exists(path):
183:             return path
184: 
185: 
186: def fixup_build_ext(cmd):
187:     '''Function needed to make build_ext tests pass.
188: 
189:     When Python was build with --enable-shared on Unix, -L. is not good
190:     enough to find the libpython<blah>.so.  This is because regrtest runs
191:     it under a tempdir, not in the top level where the .so lives.  By the
192:     time we've gotten here, Python's already been chdir'd to the tempdir.
193: 
194:     When Python was built with in debug mode on Windows, build_ext commands
195:     need their debug attribute set, and it is not done automatically for
196:     some reason.
197: 
198:     This function handles both of these things.  Example use:
199: 
200:         cmd = build_ext(dist)
201:         support.fixup_build_ext(cmd)
202:         cmd.ensure_finalized()
203: 
204:     Unlike most other Unix platforms, Mac OS X embeds absolute paths
205:     to shared libraries into executables, so the fixup is not needed there.
206:     '''
207:     if os.name == 'nt':
208:         cmd.debug = sys.executable.endswith('_d.exe')
209:     elif sysconfig.get_config_var('Py_ENABLE_SHARED'):
210:         # To further add to the shared builds fun on Unix, we can't just add
211:         # library_dirs to the Extension() instance because that doesn't get
212:         # plumbed through to the final compiler command.
213:         runshared = sysconfig.get_config_var('RUNSHARED')
214:         if runshared is None:
215:             cmd.library_dirs = ['.']
216:         else:
217:             if sys.platform == 'darwin':
218:                 cmd.library_dirs = []
219:             else:
220:                 name, equals, value = runshared.partition('=')
221:                 cmd.library_dirs = [d for d in value.split(os.pathsep) if d]
222: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_28411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 1, 0), 'str', 'Support code for distutils test cases.')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import os' statement (line 2)
import os

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import shutil' statement (line 4)
import shutil

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'shutil', shutil, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'import tempfile' statement (line 5)
import tempfile

import_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'tempfile', tempfile, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'import unittest' statement (line 6)
import unittest

import_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'unittest', unittest, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import sysconfig' statement (line 7)
import sysconfig

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'sysconfig', sysconfig, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from copy import deepcopy' statement (line 8)
try:
    from copy import deepcopy

except:
    deepcopy = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'copy', None, module_type_store, ['deepcopy'], [deepcopy])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import warnings' statement (line 9)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', warnings, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils import log' statement (line 11)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils', None, module_type_store, ['log'], [log])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.log import DEBUG, INFO, WARN, ERROR, FATAL' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28412 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.log')

if (type(import_28412) is not StypyTypeError):

    if (import_28412 != 'pyd_module'):
        __import__(import_28412)
        sys_modules_28413 = sys.modules[import_28412]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.log', sys_modules_28413.module_type_store, module_type_store, ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_28413, sys_modules_28413.module_type_store, module_type_store)
    else:
        from distutils.log import DEBUG, INFO, WARN, ERROR, FATAL

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.log', None, module_type_store, ['DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'], [DEBUG, INFO, WARN, ERROR, FATAL])

else:
    # Assigning a type to the variable 'distutils.log' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.log', import_28412)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.core import Distribution' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/tests/')
import_28414 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core')

if (type(import_28414) is not StypyTypeError):

    if (import_28414 != 'pyd_module'):
        __import__(import_28414)
        sys_modules_28415 = sys.modules[import_28414]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', sys_modules_28415.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_28415, sys_modules_28415.module_type_store, module_type_store)
    else:
        from distutils.core import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.core' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.core', import_28414)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/tests/')


@norecursion
def capture_warnings(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'capture_warnings'
    module_type_store = module_type_store.open_function_context('capture_warnings', 16, 0, False)
    
    # Passed parameters checking function
    capture_warnings.stypy_localization = localization
    capture_warnings.stypy_type_of_self = None
    capture_warnings.stypy_type_store = module_type_store
    capture_warnings.stypy_function_name = 'capture_warnings'
    capture_warnings.stypy_param_names_list = ['func']
    capture_warnings.stypy_varargs_param_name = None
    capture_warnings.stypy_kwargs_param_name = None
    capture_warnings.stypy_call_defaults = defaults
    capture_warnings.stypy_call_varargs = varargs
    capture_warnings.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'capture_warnings', ['func'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'capture_warnings', localization, ['func'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'capture_warnings(...)' code ##################


    @norecursion
    def _capture_warnings(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_capture_warnings'
        module_type_store = module_type_store.open_function_context('_capture_warnings', 17, 4, False)
        
        # Passed parameters checking function
        _capture_warnings.stypy_localization = localization
        _capture_warnings.stypy_type_of_self = None
        _capture_warnings.stypy_type_store = module_type_store
        _capture_warnings.stypy_function_name = '_capture_warnings'
        _capture_warnings.stypy_param_names_list = []
        _capture_warnings.stypy_varargs_param_name = 'args'
        _capture_warnings.stypy_kwargs_param_name = 'kw'
        _capture_warnings.stypy_call_defaults = defaults
        _capture_warnings.stypy_call_varargs = varargs
        _capture_warnings.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_capture_warnings', [], 'args', 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_capture_warnings', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_capture_warnings(...)' code ##################

        
        # Call to catch_warnings(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_28418 = {}
        # Getting the type of 'warnings' (line 18)
        warnings_28416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 13), 'warnings', False)
        # Obtaining the member 'catch_warnings' of a type (line 18)
        catch_warnings_28417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 13), warnings_28416, 'catch_warnings')
        # Calling catch_warnings(args, kwargs) (line 18)
        catch_warnings_call_result_28419 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), catch_warnings_28417, *[], **kwargs_28418)
        
        with_28420 = ensure_var_has_members(stypy.reporting.localization.Localization(__file__, 18, 13), catch_warnings_call_result_28419, 'with parameter', '__enter__', '__exit__')

        if with_28420:
            # Calling the __enter__ method to initiate a with section
            # Obtaining the member '__enter__' of a type (line 18)
            enter___28421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 13), catch_warnings_call_result_28419, '__enter__')
            with_enter_28422 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), enter___28421)
            
            # Call to simplefilter(...): (line 19)
            # Processing the call arguments (line 19)
            str_28425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 34), 'str', 'ignore')
            # Processing the call keyword arguments (line 19)
            kwargs_28426 = {}
            # Getting the type of 'warnings' (line 19)
            warnings_28423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'warnings', False)
            # Obtaining the member 'simplefilter' of a type (line 19)
            simplefilter_28424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), warnings_28423, 'simplefilter')
            # Calling simplefilter(args, kwargs) (line 19)
            simplefilter_call_result_28427 = invoke(stypy.reporting.localization.Localization(__file__, 19, 12), simplefilter_28424, *[str_28425], **kwargs_28426)
            
            
            # Call to func(...): (line 20)
            # Getting the type of 'args' (line 20)
            args_28429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 25), 'args', False)
            # Processing the call keyword arguments (line 20)
            # Getting the type of 'kw' (line 20)
            kw_28430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 33), 'kw', False)
            kwargs_28431 = {'kw_28430': kw_28430}
            # Getting the type of 'func' (line 20)
            func_28428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 19), 'func', False)
            # Calling func(args, kwargs) (line 20)
            func_call_result_28432 = invoke(stypy.reporting.localization.Localization(__file__, 20, 19), func_28428, *[args_28429], **kwargs_28431)
            
            # Assigning a type to the variable 'stypy_return_type' (line 20)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 12), 'stypy_return_type', func_call_result_28432)
            # Calling the __exit__ method to finish a with section
            # Obtaining the member '__exit__' of a type (line 18)
            exit___28433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 13), catch_warnings_call_result_28419, '__exit__')
            with_exit_28434 = invoke(stypy.reporting.localization.Localization(__file__, 18, 13), exit___28433, None, None, None)

        
        # ################# End of '_capture_warnings(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_capture_warnings' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_28435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28435)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_capture_warnings'
        return stypy_return_type_28435

    # Assigning a type to the variable '_capture_warnings' (line 17)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), '_capture_warnings', _capture_warnings)
    # Getting the type of '_capture_warnings' (line 21)
    _capture_warnings_28436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 11), '_capture_warnings')
    # Assigning a type to the variable 'stypy_return_type' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type', _capture_warnings_28436)
    
    # ################# End of 'capture_warnings(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'capture_warnings' in the type store
    # Getting the type of 'stypy_return_type' (line 16)
    stypy_return_type_28437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28437)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'capture_warnings'
    return stypy_return_type_28437

# Assigning a type to the variable 'capture_warnings' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'capture_warnings', capture_warnings)
# Declaration of the 'LoggingSilencer' class

class LoggingSilencer(object, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_localization', localization)
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_function_name', 'LoggingSilencer.setUp')
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingSilencer.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingSilencer.setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUp(...)' code ##################

        
        # Call to setUp(...): (line 27)
        # Processing the call keyword arguments (line 27)
        kwargs_28444 = {}
        
        # Call to super(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'LoggingSilencer' (line 27)
        LoggingSilencer_28439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 14), 'LoggingSilencer', False)
        # Getting the type of 'self' (line 27)
        self_28440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 31), 'self', False)
        # Processing the call keyword arguments (line 27)
        kwargs_28441 = {}
        # Getting the type of 'super' (line 27)
        super_28438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'super', False)
        # Calling super(args, kwargs) (line 27)
        super_call_result_28442 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), super_28438, *[LoggingSilencer_28439, self_28440], **kwargs_28441)
        
        # Obtaining the member 'setUp' of a type (line 27)
        setUp_28443 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), super_call_result_28442, 'setUp')
        # Calling setUp(args, kwargs) (line 27)
        setUp_call_result_28445 = invoke(stypy.reporting.localization.Localization(__file__, 27, 8), setUp_28443, *[], **kwargs_28444)
        
        
        # Assigning a Call to a Attribute (line 28):
        
        # Assigning a Call to a Attribute (line 28):
        
        # Call to set_threshold(...): (line 28)
        # Processing the call arguments (line 28)
        # Getting the type of 'log' (line 28)
        log_28448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 43), 'log', False)
        # Obtaining the member 'FATAL' of a type (line 28)
        FATAL_28449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 43), log_28448, 'FATAL')
        # Processing the call keyword arguments (line 28)
        kwargs_28450 = {}
        # Getting the type of 'log' (line 28)
        log_28446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 25), 'log', False)
        # Obtaining the member 'set_threshold' of a type (line 28)
        set_threshold_28447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 25), log_28446, 'set_threshold')
        # Calling set_threshold(args, kwargs) (line 28)
        set_threshold_call_result_28451 = invoke(stypy.reporting.localization.Localization(__file__, 28, 25), set_threshold_28447, *[FATAL_28449], **kwargs_28450)
        
        # Getting the type of 'self' (line 28)
        self_28452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Setting the type of the member 'threshold' of a type (line 28)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_28452, 'threshold', set_threshold_call_result_28451)
        
        # Assigning a Attribute to a Attribute (line 32):
        
        # Assigning a Attribute to a Attribute (line 32):
        # Getting the type of 'log' (line 32)
        log_28453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'log')
        # Obtaining the member 'Log' of a type (line 32)
        Log_28454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 24), log_28453, 'Log')
        # Obtaining the member '_log' of a type (line 32)
        _log_28455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 24), Log_28454, '_log')
        # Getting the type of 'self' (line 32)
        self_28456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'self')
        # Setting the type of the member '_old_log' of a type (line 32)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 8), self_28456, '_old_log', _log_28455)
        
        # Assigning a Attribute to a Attribute (line 33):
        
        # Assigning a Attribute to a Attribute (line 33):
        # Getting the type of 'self' (line 33)
        self_28457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 23), 'self')
        # Obtaining the member '_log' of a type (line 33)
        _log_28458 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 23), self_28457, '_log')
        # Getting the type of 'log' (line 33)
        log_28459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'log')
        # Obtaining the member 'Log' of a type (line 33)
        Log_28460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), log_28459, 'Log')
        # Setting the type of the member '_log' of a type (line 33)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 8), Log_28460, '_log', _log_28458)
        
        # Assigning a List to a Attribute (line 34):
        
        # Assigning a List to a Attribute (line 34):
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_28461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        
        # Getting the type of 'self' (line 34)
        self_28462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'self')
        # Setting the type of the member 'logs' of a type (line 34)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 8), self_28462, 'logs', list_28461)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_28463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28463)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_28463


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_localization', localization)
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_function_name', 'LoggingSilencer.tearDown')
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingSilencer.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingSilencer.tearDown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tearDown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tearDown(...)' code ##################

        
        # Call to set_threshold(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'self' (line 37)
        self_28466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'self', False)
        # Obtaining the member 'threshold' of a type (line 37)
        threshold_28467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 26), self_28466, 'threshold')
        # Processing the call keyword arguments (line 37)
        kwargs_28468 = {}
        # Getting the type of 'log' (line 37)
        log_28464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'log', False)
        # Obtaining the member 'set_threshold' of a type (line 37)
        set_threshold_28465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), log_28464, 'set_threshold')
        # Calling set_threshold(args, kwargs) (line 37)
        set_threshold_call_result_28469 = invoke(stypy.reporting.localization.Localization(__file__, 37, 8), set_threshold_28465, *[threshold_28467], **kwargs_28468)
        
        
        # Assigning a Attribute to a Attribute (line 38):
        
        # Assigning a Attribute to a Attribute (line 38):
        # Getting the type of 'self' (line 38)
        self_28470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 23), 'self')
        # Obtaining the member '_old_log' of a type (line 38)
        _old_log_28471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 23), self_28470, '_old_log')
        # Getting the type of 'log' (line 38)
        log_28472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'log')
        # Obtaining the member 'Log' of a type (line 38)
        Log_28473 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), log_28472, 'Log')
        # Setting the type of the member '_log' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), Log_28473, '_log', _old_log_28471)
        
        # Call to tearDown(...): (line 39)
        # Processing the call keyword arguments (line 39)
        kwargs_28480 = {}
        
        # Call to super(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'LoggingSilencer' (line 39)
        LoggingSilencer_28475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 14), 'LoggingSilencer', False)
        # Getting the type of 'self' (line 39)
        self_28476 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 31), 'self', False)
        # Processing the call keyword arguments (line 39)
        kwargs_28477 = {}
        # Getting the type of 'super' (line 39)
        super_28474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'super', False)
        # Calling super(args, kwargs) (line 39)
        super_call_result_28478 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), super_28474, *[LoggingSilencer_28475, self_28476], **kwargs_28477)
        
        # Obtaining the member 'tearDown' of a type (line 39)
        tearDown_28479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), super_call_result_28478, 'tearDown')
        # Calling tearDown(args, kwargs) (line 39)
        tearDown_call_result_28481 = invoke(stypy.reporting.localization.Localization(__file__, 39, 8), tearDown_28479, *[], **kwargs_28480)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_28482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28482)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_28482


    @norecursion
    def _log(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_log'
        module_type_store = module_type_store.open_function_context('_log', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingSilencer._log.__dict__.__setitem__('stypy_localization', localization)
        LoggingSilencer._log.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingSilencer._log.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingSilencer._log.__dict__.__setitem__('stypy_function_name', 'LoggingSilencer._log')
        LoggingSilencer._log.__dict__.__setitem__('stypy_param_names_list', ['level', 'msg', 'args'])
        LoggingSilencer._log.__dict__.__setitem__('stypy_varargs_param_name', None)
        LoggingSilencer._log.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingSilencer._log.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingSilencer._log.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingSilencer._log.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingSilencer._log.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingSilencer._log', ['level', 'msg', 'args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_log', localization, ['level', 'msg', 'args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_log(...)' code ##################

        
        
        # Getting the type of 'level' (line 42)
        level_28483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'level')
        
        # Obtaining an instance of the builtin type 'tuple' (line 42)
        tuple_28484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 25), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 42)
        # Adding element type (line 42)
        # Getting the type of 'DEBUG' (line 42)
        DEBUG_28485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 25), 'DEBUG')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), tuple_28484, DEBUG_28485)
        # Adding element type (line 42)
        # Getting the type of 'INFO' (line 42)
        INFO_28486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 32), 'INFO')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), tuple_28484, INFO_28486)
        # Adding element type (line 42)
        # Getting the type of 'WARN' (line 42)
        WARN_28487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'WARN')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), tuple_28484, WARN_28487)
        # Adding element type (line 42)
        # Getting the type of 'ERROR' (line 42)
        ERROR_28488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 44), 'ERROR')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), tuple_28484, ERROR_28488)
        # Adding element type (line 42)
        # Getting the type of 'FATAL' (line 42)
        FATAL_28489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 51), 'FATAL')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 25), tuple_28484, FATAL_28489)
        
        # Applying the binary operator 'notin' (line 42)
        result_contains_28490 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), 'notin', level_28483, tuple_28484)
        
        # Testing the type of an if condition (line 42)
        if_condition_28491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_contains_28490)
        # Assigning a type to the variable 'if_condition_28491' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_28491', if_condition_28491)
        # SSA begins for if statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to ValueError(...): (line 43)
        # Processing the call arguments (line 43)
        str_28493 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'str', '%s wrong log level')
        
        # Call to str(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'level' (line 43)
        level_28495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 56), 'level', False)
        # Processing the call keyword arguments (line 43)
        kwargs_28496 = {}
        # Getting the type of 'str' (line 43)
        str_28494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 52), 'str', False)
        # Calling str(args, kwargs) (line 43)
        str_call_result_28497 = invoke(stypy.reporting.localization.Localization(__file__, 43, 52), str_28494, *[level_28495], **kwargs_28496)
        
        # Applying the binary operator '%' (line 43)
        result_mod_28498 = python_operator(stypy.reporting.localization.Localization(__file__, 43, 29), '%', str_28493, str_call_result_28497)
        
        # Processing the call keyword arguments (line 43)
        kwargs_28499 = {}
        # Getting the type of 'ValueError' (line 43)
        ValueError_28492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 18), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 43)
        ValueError_call_result_28500 = invoke(stypy.reporting.localization.Localization(__file__, 43, 18), ValueError_28492, *[result_mod_28498], **kwargs_28499)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 43, 12), ValueError_call_result_28500, 'raise parameter', BaseException)
        # SSA join for if statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 44)
        # Processing the call arguments (line 44)
        
        # Obtaining an instance of the builtin type 'tuple' (line 44)
        tuple_28504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 26), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 44)
        # Adding element type (line 44)
        # Getting the type of 'level' (line 44)
        level_28505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 26), 'level', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 26), tuple_28504, level_28505)
        # Adding element type (line 44)
        # Getting the type of 'msg' (line 44)
        msg_28506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 33), 'msg', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 26), tuple_28504, msg_28506)
        # Adding element type (line 44)
        # Getting the type of 'args' (line 44)
        args_28507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 38), 'args', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 26), tuple_28504, args_28507)
        
        # Processing the call keyword arguments (line 44)
        kwargs_28508 = {}
        # Getting the type of 'self' (line 44)
        self_28501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self', False)
        # Obtaining the member 'logs' of a type (line 44)
        logs_28502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_28501, 'logs')
        # Obtaining the member 'append' of a type (line 44)
        append_28503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), logs_28502, 'append')
        # Calling append(args, kwargs) (line 44)
        append_call_result_28509 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), append_28503, *[tuple_28504], **kwargs_28508)
        
        
        # ################# End of '_log(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_log' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_28510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28510)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_log'
        return stypy_return_type_28510


    @norecursion
    def get_logs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_logs'
        module_type_store = module_type_store.open_function_context('get_logs', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_localization', localization)
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_function_name', 'LoggingSilencer.get_logs')
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_varargs_param_name', 'levels')
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingSilencer.get_logs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingSilencer.get_logs', [], 'levels', None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_logs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_logs(...)' code ##################


        @norecursion
        def _format(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_format'
            module_type_store = module_type_store.open_function_context('_format', 47, 8, False)
            
            # Passed parameters checking function
            _format.stypy_localization = localization
            _format.stypy_type_of_self = None
            _format.stypy_type_store = module_type_store
            _format.stypy_function_name = '_format'
            _format.stypy_param_names_list = ['msg', 'args']
            _format.stypy_varargs_param_name = None
            _format.stypy_kwargs_param_name = None
            _format.stypy_call_defaults = defaults
            _format.stypy_call_varargs = varargs
            _format.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_format', ['msg', 'args'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Initialize method data
            init_call_information(module_type_store, '_format', localization, ['msg', 'args'], arguments)
            
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of '_format(...)' code ##################

            
            
            
            # Call to len(...): (line 48)
            # Processing the call arguments (line 48)
            # Getting the type of 'args' (line 48)
            args_28512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 19), 'args', False)
            # Processing the call keyword arguments (line 48)
            kwargs_28513 = {}
            # Getting the type of 'len' (line 48)
            len_28511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'len', False)
            # Calling len(args, kwargs) (line 48)
            len_call_result_28514 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), len_28511, *[args_28512], **kwargs_28513)
            
            int_28515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 28), 'int')
            # Applying the binary operator '==' (line 48)
            result_eq_28516 = python_operator(stypy.reporting.localization.Localization(__file__, 48, 15), '==', len_call_result_28514, int_28515)
            
            # Testing the type of an if condition (line 48)
            if_condition_28517 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 48, 12), result_eq_28516)
            # Assigning a type to the variable 'if_condition_28517' (line 48)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'if_condition_28517', if_condition_28517)
            # SSA begins for if statement (line 48)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'msg' (line 49)
            msg_28518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'msg')
            # Assigning a type to the variable 'stypy_return_type' (line 49)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 16), 'stypy_return_type', msg_28518)
            # SSA join for if statement (line 48)
            module_type_store = module_type_store.join_ssa_context()
            
            # Getting the type of 'msg' (line 50)
            msg_28519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 19), 'msg')
            # Getting the type of 'args' (line 50)
            args_28520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 25), 'args')
            # Applying the binary operator '%' (line 50)
            result_mod_28521 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 19), '%', msg_28519, args_28520)
            
            # Assigning a type to the variable 'stypy_return_type' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 12), 'stypy_return_type', result_mod_28521)
            
            # ################# End of '_format(...)' code ##################

            # Teardown call information
            teardown_call_information(localization, arguments)
            
            # Storing the return type of function '_format' in the type store
            # Getting the type of 'stypy_return_type' (line 47)
            stypy_return_type_28522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_28522)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_format'
            return stypy_return_type_28522

        # Assigning a type to the variable '_format' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), '_format', _format)
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 52)
        self_28531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'self')
        # Obtaining the member 'logs' of a type (line 52)
        logs_28532 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), self_28531, 'logs')
        comprehension_28533 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), logs_28532)
        # Assigning a type to the variable 'level' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'level', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), comprehension_28533))
        # Assigning a type to the variable 'msg' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'msg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), comprehension_28533))
        # Assigning a type to the variable 'args' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'args', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), comprehension_28533))
        
        # Getting the type of 'level' (line 52)
        level_28528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 'level')
        # Getting the type of 'levels' (line 52)
        levels_28529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 41), 'levels')
        # Applying the binary operator 'in' (line 52)
        result_contains_28530 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 32), 'in', level_28528, levels_28529)
        
        
        # Call to _format(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'msg' (line 51)
        msg_28524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'msg', False)
        # Getting the type of 'args' (line 51)
        args_28525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 29), 'args', False)
        # Processing the call keyword arguments (line 51)
        kwargs_28526 = {}
        # Getting the type of '_format' (line 51)
        _format_28523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), '_format', False)
        # Calling _format(args, kwargs) (line 51)
        _format_call_result_28527 = invoke(stypy.reporting.localization.Localization(__file__, 51, 16), _format_28523, *[msg_28524, args_28525], **kwargs_28526)
        
        list_28534 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 16), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 16), list_28534, _format_call_result_28527)
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', list_28534)
        
        # ################# End of 'get_logs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_logs' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_28535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28535)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_logs'
        return stypy_return_type_28535


    @norecursion
    def clear_logs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clear_logs'
        module_type_store = module_type_store.open_function_context('clear_logs', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_localization', localization)
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_type_store', module_type_store)
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_function_name', 'LoggingSilencer.clear_logs')
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_param_names_list', [])
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_varargs_param_name', None)
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_call_defaults', defaults)
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_call_varargs', varargs)
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        LoggingSilencer.clear_logs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingSilencer.clear_logs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clear_logs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clear_logs(...)' code ##################

        
        # Assigning a List to a Attribute (line 55):
        
        # Assigning a List to a Attribute (line 55):
        
        # Obtaining an instance of the builtin type 'list' (line 55)
        list_28536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 55)
        
        # Getting the type of 'self' (line 55)
        self_28537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'logs' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_28537, 'logs', list_28536)
        
        # ################# End of 'clear_logs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clear_logs' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_28538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28538)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clear_logs'
        return stypy_return_type_28538


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 24, 0, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LoggingSilencer.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LoggingSilencer' (line 24)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 0), 'LoggingSilencer', LoggingSilencer)
# Declaration of the 'TempdirManager' class

class TempdirManager(object, ):
    str_28539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', 'Mix-in class that handles temporary directories for test cases.\n\n    This is intended to be used with unittest.TestCase.\n    ')

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 64, 4, False)
        # Assigning a type to the variable 'self' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TempdirManager.setUp.__dict__.__setitem__('stypy_localization', localization)
        TempdirManager.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TempdirManager.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TempdirManager.setUp.__dict__.__setitem__('stypy_function_name', 'TempdirManager.setUp')
        TempdirManager.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        TempdirManager.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TempdirManager.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TempdirManager.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TempdirManager.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TempdirManager.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TempdirManager.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempdirManager.setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUp(...)' code ##################

        
        # Call to setUp(...): (line 65)
        # Processing the call keyword arguments (line 65)
        kwargs_28546 = {}
        
        # Call to super(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'TempdirManager' (line 65)
        TempdirManager_28541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 14), 'TempdirManager', False)
        # Getting the type of 'self' (line 65)
        self_28542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 30), 'self', False)
        # Processing the call keyword arguments (line 65)
        kwargs_28543 = {}
        # Getting the type of 'super' (line 65)
        super_28540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'super', False)
        # Calling super(args, kwargs) (line 65)
        super_call_result_28544 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), super_28540, *[TempdirManager_28541, self_28542], **kwargs_28543)
        
        # Obtaining the member 'setUp' of a type (line 65)
        setUp_28545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), super_call_result_28544, 'setUp')
        # Calling setUp(args, kwargs) (line 65)
        setUp_call_result_28547 = invoke(stypy.reporting.localization.Localization(__file__, 65, 8), setUp_28545, *[], **kwargs_28546)
        
        
        # Assigning a Call to a Attribute (line 66):
        
        # Assigning a Call to a Attribute (line 66):
        
        # Call to getcwd(...): (line 66)
        # Processing the call keyword arguments (line 66)
        kwargs_28550 = {}
        # Getting the type of 'os' (line 66)
        os_28548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 23), 'os', False)
        # Obtaining the member 'getcwd' of a type (line 66)
        getcwd_28549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 23), os_28548, 'getcwd')
        # Calling getcwd(args, kwargs) (line 66)
        getcwd_call_result_28551 = invoke(stypy.reporting.localization.Localization(__file__, 66, 23), getcwd_28549, *[], **kwargs_28550)
        
        # Getting the type of 'self' (line 66)
        self_28552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'old_cwd' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_28552, 'old_cwd', getcwd_call_result_28551)
        
        # Assigning a List to a Attribute (line 67):
        
        # Assigning a List to a Attribute (line 67):
        
        # Obtaining an instance of the builtin type 'list' (line 67)
        list_28553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 67)
        
        # Getting the type of 'self' (line 67)
        self_28554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'tempdirs' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_28554, 'tempdirs', list_28553)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 64)
        stypy_return_type_28555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28555)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_28555


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 69, 4, False)
        # Assigning a type to the variable 'self' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TempdirManager.tearDown.__dict__.__setitem__('stypy_localization', localization)
        TempdirManager.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TempdirManager.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        TempdirManager.tearDown.__dict__.__setitem__('stypy_function_name', 'TempdirManager.tearDown')
        TempdirManager.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        TempdirManager.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        TempdirManager.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TempdirManager.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        TempdirManager.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        TempdirManager.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TempdirManager.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempdirManager.tearDown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tearDown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tearDown(...)' code ##################

        
        # Call to chdir(...): (line 72)
        # Processing the call arguments (line 72)
        # Getting the type of 'self' (line 72)
        self_28558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 17), 'self', False)
        # Obtaining the member 'old_cwd' of a type (line 72)
        old_cwd_28559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 17), self_28558, 'old_cwd')
        # Processing the call keyword arguments (line 72)
        kwargs_28560 = {}
        # Getting the type of 'os' (line 72)
        os_28556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'os', False)
        # Obtaining the member 'chdir' of a type (line 72)
        chdir_28557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 8), os_28556, 'chdir')
        # Calling chdir(args, kwargs) (line 72)
        chdir_call_result_28561 = invoke(stypy.reporting.localization.Localization(__file__, 72, 8), chdir_28557, *[old_cwd_28559], **kwargs_28560)
        
        
        # Call to tearDown(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_28568 = {}
        
        # Call to super(...): (line 73)
        # Processing the call arguments (line 73)
        # Getting the type of 'TempdirManager' (line 73)
        TempdirManager_28563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'TempdirManager', False)
        # Getting the type of 'self' (line 73)
        self_28564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 30), 'self', False)
        # Processing the call keyword arguments (line 73)
        kwargs_28565 = {}
        # Getting the type of 'super' (line 73)
        super_28562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'super', False)
        # Calling super(args, kwargs) (line 73)
        super_call_result_28566 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), super_28562, *[TempdirManager_28563, self_28564], **kwargs_28565)
        
        # Obtaining the member 'tearDown' of a type (line 73)
        tearDown_28567 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 8), super_call_result_28566, 'tearDown')
        # Calling tearDown(args, kwargs) (line 73)
        tearDown_call_result_28569 = invoke(stypy.reporting.localization.Localization(__file__, 73, 8), tearDown_28567, *[], **kwargs_28568)
        
        
        # Getting the type of 'self' (line 74)
        self_28570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 14), 'self')
        # Obtaining the member 'tempdirs' of a type (line 74)
        tempdirs_28571 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 14), self_28570, 'tempdirs')
        # Testing the type of an if condition (line 74)
        is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 8), tempdirs_28571)
        # SSA begins for while statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
        
        # Assigning a Call to a Name (line 75):
        
        # Assigning a Call to a Name (line 75):
        
        # Call to pop(...): (line 75)
        # Processing the call keyword arguments (line 75)
        kwargs_28575 = {}
        # Getting the type of 'self' (line 75)
        self_28572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'self', False)
        # Obtaining the member 'tempdirs' of a type (line 75)
        tempdirs_28573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), self_28572, 'tempdirs')
        # Obtaining the member 'pop' of a type (line 75)
        pop_28574 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), tempdirs_28573, 'pop')
        # Calling pop(args, kwargs) (line 75)
        pop_call_result_28576 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), pop_28574, *[], **kwargs_28575)
        
        # Assigning a type to the variable 'd' (line 75)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 12), 'd', pop_call_result_28576)
        
        # Call to rmtree(...): (line 76)
        # Processing the call arguments (line 76)
        # Getting the type of 'd' (line 76)
        d_28579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 26), 'd', False)
        
        # Getting the type of 'os' (line 76)
        os_28580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 29), 'os', False)
        # Obtaining the member 'name' of a type (line 76)
        name_28581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 29), os_28580, 'name')
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_28582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 41), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        str_28583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 41), 'str', 'nt')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 41), tuple_28582, str_28583)
        # Adding element type (line 76)
        str_28584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 47), 'str', 'cygwin')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 41), tuple_28582, str_28584)
        
        # Applying the binary operator 'in' (line 76)
        result_contains_28585 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 29), 'in', name_28581, tuple_28582)
        
        # Processing the call keyword arguments (line 76)
        kwargs_28586 = {}
        # Getting the type of 'shutil' (line 76)
        shutil_28577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'shutil', False)
        # Obtaining the member 'rmtree' of a type (line 76)
        rmtree_28578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), shutil_28577, 'rmtree')
        # Calling rmtree(args, kwargs) (line 76)
        rmtree_call_result_28587 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), rmtree_28578, *[d_28579, result_contains_28585], **kwargs_28586)
        
        # SSA join for while statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 69)
        stypy_return_type_28588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28588)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_28588


    @norecursion
    def mkdtemp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'mkdtemp'
        module_type_store = module_type_store.open_function_context('mkdtemp', 78, 4, False)
        # Assigning a type to the variable 'self' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_localization', localization)
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_type_store', module_type_store)
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_function_name', 'TempdirManager.mkdtemp')
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_param_names_list', [])
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_varargs_param_name', None)
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_call_defaults', defaults)
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_call_varargs', varargs)
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TempdirManager.mkdtemp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempdirManager.mkdtemp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'mkdtemp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'mkdtemp(...)' code ##################

        str_28589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, (-1)), 'str', 'Create a temporary directory that will be cleaned up.\n\n        Returns the path of the directory.\n        ')
        
        # Assigning a Call to a Name (line 83):
        
        # Assigning a Call to a Name (line 83):
        
        # Call to mkdtemp(...): (line 83)
        # Processing the call keyword arguments (line 83)
        kwargs_28592 = {}
        # Getting the type of 'tempfile' (line 83)
        tempfile_28590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'tempfile', False)
        # Obtaining the member 'mkdtemp' of a type (line 83)
        mkdtemp_28591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 12), tempfile_28590, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 83)
        mkdtemp_call_result_28593 = invoke(stypy.reporting.localization.Localization(__file__, 83, 12), mkdtemp_28591, *[], **kwargs_28592)
        
        # Assigning a type to the variable 'd' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'd', mkdtemp_call_result_28593)
        
        # Call to append(...): (line 84)
        # Processing the call arguments (line 84)
        # Getting the type of 'd' (line 84)
        d_28597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 29), 'd', False)
        # Processing the call keyword arguments (line 84)
        kwargs_28598 = {}
        # Getting the type of 'self' (line 84)
        self_28594 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'self', False)
        # Obtaining the member 'tempdirs' of a type (line 84)
        tempdirs_28595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), self_28594, 'tempdirs')
        # Obtaining the member 'append' of a type (line 84)
        append_28596 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 8), tempdirs_28595, 'append')
        # Calling append(args, kwargs) (line 84)
        append_call_result_28599 = invoke(stypy.reporting.localization.Localization(__file__, 84, 8), append_28596, *[d_28597], **kwargs_28598)
        
        # Getting the type of 'd' (line 85)
        d_28600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 15), 'd')
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'stypy_return_type', d_28600)
        
        # ################# End of 'mkdtemp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'mkdtemp' in the type store
        # Getting the type of 'stypy_return_type' (line 78)
        stypy_return_type_28601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28601)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'mkdtemp'
        return stypy_return_type_28601


    @norecursion
    def write_file(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_28602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 39), 'str', 'xxx')
        defaults = [str_28602]
        # Create a new context for function 'write_file'
        module_type_store = module_type_store.open_function_context('write_file', 87, 4, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TempdirManager.write_file.__dict__.__setitem__('stypy_localization', localization)
        TempdirManager.write_file.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TempdirManager.write_file.__dict__.__setitem__('stypy_type_store', module_type_store)
        TempdirManager.write_file.__dict__.__setitem__('stypy_function_name', 'TempdirManager.write_file')
        TempdirManager.write_file.__dict__.__setitem__('stypy_param_names_list', ['path', 'content'])
        TempdirManager.write_file.__dict__.__setitem__('stypy_varargs_param_name', None)
        TempdirManager.write_file.__dict__.__setitem__('stypy_kwargs_param_name', None)
        TempdirManager.write_file.__dict__.__setitem__('stypy_call_defaults', defaults)
        TempdirManager.write_file.__dict__.__setitem__('stypy_call_varargs', varargs)
        TempdirManager.write_file.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TempdirManager.write_file.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempdirManager.write_file', ['path', 'content'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'write_file', localization, ['path', 'content'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'write_file(...)' code ##################

        str_28603 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', 'Writes a file in the given path.\n\n\n        path can be a string or a sequence.\n        ')
        
        
        # Call to isinstance(...): (line 93)
        # Processing the call arguments (line 93)
        # Getting the type of 'path' (line 93)
        path_28605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'path', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 93)
        tuple_28606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 93)
        # Adding element type (line 93)
        # Getting the type of 'list' (line 93)
        list_28607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 29), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 29), tuple_28606, list_28607)
        # Adding element type (line 93)
        # Getting the type of 'tuple' (line 93)
        tuple_28608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 35), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 29), tuple_28606, tuple_28608)
        
        # Processing the call keyword arguments (line 93)
        kwargs_28609 = {}
        # Getting the type of 'isinstance' (line 93)
        isinstance_28604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 93)
        isinstance_call_result_28610 = invoke(stypy.reporting.localization.Localization(__file__, 93, 11), isinstance_28604, *[path_28605, tuple_28606], **kwargs_28609)
        
        # Testing the type of an if condition (line 93)
        if_condition_28611 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 8), isinstance_call_result_28610)
        # Assigning a type to the variable 'if_condition_28611' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'if_condition_28611', if_condition_28611)
        # SSA begins for if statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 94):
        
        # Assigning a Call to a Name (line 94):
        
        # Call to join(...): (line 94)
        # Getting the type of 'path' (line 94)
        path_28615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 33), 'path', False)
        # Processing the call keyword arguments (line 94)
        kwargs_28616 = {}
        # Getting the type of 'os' (line 94)
        os_28612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 94)
        path_28613 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 19), os_28612, 'path')
        # Obtaining the member 'join' of a type (line 94)
        join_28614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 19), path_28613, 'join')
        # Calling join(args, kwargs) (line 94)
        join_call_result_28617 = invoke(stypy.reporting.localization.Localization(__file__, 94, 19), join_28614, *[path_28615], **kwargs_28616)
        
        # Assigning a type to the variable 'path' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'path', join_call_result_28617)
        # SSA join for if statement (line 93)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 95):
        
        # Assigning a Call to a Name (line 95):
        
        # Call to open(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'path' (line 95)
        path_28619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'path', False)
        str_28620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 23), 'str', 'w')
        # Processing the call keyword arguments (line 95)
        kwargs_28621 = {}
        # Getting the type of 'open' (line 95)
        open_28618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'open', False)
        # Calling open(args, kwargs) (line 95)
        open_call_result_28622 = invoke(stypy.reporting.localization.Localization(__file__, 95, 12), open_28618, *[path_28619, str_28620], **kwargs_28621)
        
        # Assigning a type to the variable 'f' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'f', open_call_result_28622)
        
        # Try-finally block (line 96)
        
        # Call to write(...): (line 97)
        # Processing the call arguments (line 97)
        # Getting the type of 'content' (line 97)
        content_28625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 20), 'content', False)
        # Processing the call keyword arguments (line 97)
        kwargs_28626 = {}
        # Getting the type of 'f' (line 97)
        f_28623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'f', False)
        # Obtaining the member 'write' of a type (line 97)
        write_28624 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), f_28623, 'write')
        # Calling write(args, kwargs) (line 97)
        write_call_result_28627 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), write_28624, *[content_28625], **kwargs_28626)
        
        
        # finally branch of the try-finally block (line 96)
        
        # Call to close(...): (line 99)
        # Processing the call keyword arguments (line 99)
        kwargs_28630 = {}
        # Getting the type of 'f' (line 99)
        f_28628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 99)
        close_28629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), f_28628, 'close')
        # Calling close(args, kwargs) (line 99)
        close_call_result_28631 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), close_28629, *[], **kwargs_28630)
        
        
        
        # ################# End of 'write_file(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'write_file' in the type store
        # Getting the type of 'stypy_return_type' (line 87)
        stypy_return_type_28632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28632)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'write_file'
        return stypy_return_type_28632


    @norecursion
    def create_dist(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        str_28633 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 35), 'str', 'foo')
        defaults = [str_28633]
        # Create a new context for function 'create_dist'
        module_type_store = module_type_store.open_function_context('create_dist', 101, 4, False)
        # Assigning a type to the variable 'self' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        TempdirManager.create_dist.__dict__.__setitem__('stypy_localization', localization)
        TempdirManager.create_dist.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        TempdirManager.create_dist.__dict__.__setitem__('stypy_type_store', module_type_store)
        TempdirManager.create_dist.__dict__.__setitem__('stypy_function_name', 'TempdirManager.create_dist')
        TempdirManager.create_dist.__dict__.__setitem__('stypy_param_names_list', ['pkg_name'])
        TempdirManager.create_dist.__dict__.__setitem__('stypy_varargs_param_name', None)
        TempdirManager.create_dist.__dict__.__setitem__('stypy_kwargs_param_name', 'kw')
        TempdirManager.create_dist.__dict__.__setitem__('stypy_call_defaults', defaults)
        TempdirManager.create_dist.__dict__.__setitem__('stypy_call_varargs', varargs)
        TempdirManager.create_dist.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        TempdirManager.create_dist.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempdirManager.create_dist', ['pkg_name'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_dist', localization, ['pkg_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_dist(...)' code ##################

        str_28634 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, (-1)), 'str', 'Will generate a test environment.\n\n        This function creates:\n         - a Distribution instance using keywords\n         - a temporary directory with a package structure\n\n        It returns the package directory and the distribution\n        instance.\n        ')
        
        # Assigning a Call to a Name (line 111):
        
        # Assigning a Call to a Name (line 111):
        
        # Call to mkdtemp(...): (line 111)
        # Processing the call keyword arguments (line 111)
        kwargs_28637 = {}
        # Getting the type of 'self' (line 111)
        self_28635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 18), 'self', False)
        # Obtaining the member 'mkdtemp' of a type (line 111)
        mkdtemp_28636 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 18), self_28635, 'mkdtemp')
        # Calling mkdtemp(args, kwargs) (line 111)
        mkdtemp_call_result_28638 = invoke(stypy.reporting.localization.Localization(__file__, 111, 18), mkdtemp_28636, *[], **kwargs_28637)
        
        # Assigning a type to the variable 'tmp_dir' (line 111)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'tmp_dir', mkdtemp_call_result_28638)
        
        # Assigning a Call to a Name (line 112):
        
        # Assigning a Call to a Name (line 112):
        
        # Call to join(...): (line 112)
        # Processing the call arguments (line 112)
        # Getting the type of 'tmp_dir' (line 112)
        tmp_dir_28642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 31), 'tmp_dir', False)
        # Getting the type of 'pkg_name' (line 112)
        pkg_name_28643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 40), 'pkg_name', False)
        # Processing the call keyword arguments (line 112)
        kwargs_28644 = {}
        # Getting the type of 'os' (line 112)
        os_28639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 18), 'os', False)
        # Obtaining the member 'path' of a type (line 112)
        path_28640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 18), os_28639, 'path')
        # Obtaining the member 'join' of a type (line 112)
        join_28641 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 18), path_28640, 'join')
        # Calling join(args, kwargs) (line 112)
        join_call_result_28645 = invoke(stypy.reporting.localization.Localization(__file__, 112, 18), join_28641, *[tmp_dir_28642, pkg_name_28643], **kwargs_28644)
        
        # Assigning a type to the variable 'pkg_dir' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'pkg_dir', join_call_result_28645)
        
        # Call to mkdir(...): (line 113)
        # Processing the call arguments (line 113)
        # Getting the type of 'pkg_dir' (line 113)
        pkg_dir_28648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 17), 'pkg_dir', False)
        # Processing the call keyword arguments (line 113)
        kwargs_28649 = {}
        # Getting the type of 'os' (line 113)
        os_28646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'os', False)
        # Obtaining the member 'mkdir' of a type (line 113)
        mkdir_28647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), os_28646, 'mkdir')
        # Calling mkdir(args, kwargs) (line 113)
        mkdir_call_result_28650 = invoke(stypy.reporting.localization.Localization(__file__, 113, 8), mkdir_28647, *[pkg_dir_28648], **kwargs_28649)
        
        
        # Assigning a Call to a Name (line 114):
        
        # Assigning a Call to a Name (line 114):
        
        # Call to Distribution(...): (line 114)
        # Processing the call keyword arguments (line 114)
        # Getting the type of 'kw' (line 114)
        kw_28652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 34), 'kw', False)
        keyword_28653 = kw_28652
        kwargs_28654 = {'attrs': keyword_28653}
        # Getting the type of 'Distribution' (line 114)
        Distribution_28651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'Distribution', False)
        # Calling Distribution(args, kwargs) (line 114)
        Distribution_call_result_28655 = invoke(stypy.reporting.localization.Localization(__file__, 114, 15), Distribution_28651, *[], **kwargs_28654)
        
        # Assigning a type to the variable 'dist' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'dist', Distribution_call_result_28655)
        
        # Obtaining an instance of the builtin type 'tuple' (line 116)
        tuple_28656 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 116)
        # Adding element type (line 116)
        # Getting the type of 'pkg_dir' (line 116)
        pkg_dir_28657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 15), 'pkg_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 15), tuple_28656, pkg_dir_28657)
        # Adding element type (line 116)
        # Getting the type of 'dist' (line 116)
        dist_28658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 24), 'dist')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 116, 15), tuple_28656, dist_28658)
        
        # Assigning a type to the variable 'stypy_return_type' (line 116)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'stypy_return_type', tuple_28656)
        
        # ################# End of 'create_dist(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_dist' in the type store
        # Getting the type of 'stypy_return_type' (line 101)
        stypy_return_type_28659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28659)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_dist'
        return stypy_return_type_28659


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 58, 0, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'TempdirManager.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'TempdirManager' (line 58)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 0), 'TempdirManager', TempdirManager)
# Declaration of the 'DummyCommand' class

class DummyCommand:
    str_28660 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 4), 'str', 'Class to store options for retrieval via set_undefined_options().')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 122, 4, False)
        # Assigning a type to the variable 'self' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyCommand.__init__', [], None, 'kwargs', defaults, varargs, kwargs)

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

        
        
        # Call to items(...): (line 123)
        # Processing the call keyword arguments (line 123)
        kwargs_28663 = {}
        # Getting the type of 'kwargs' (line 123)
        kwargs_28661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 23), 'kwargs', False)
        # Obtaining the member 'items' of a type (line 123)
        items_28662 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 23), kwargs_28661, 'items')
        # Calling items(args, kwargs) (line 123)
        items_call_result_28664 = invoke(stypy.reporting.localization.Localization(__file__, 123, 23), items_28662, *[], **kwargs_28663)
        
        # Testing the type of a for loop iterable (line 123)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 123, 8), items_call_result_28664)
        # Getting the type of the for loop variable (line 123)
        for_loop_var_28665 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 123, 8), items_call_result_28664)
        # Assigning a type to the variable 'kw' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'kw', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), for_loop_var_28665))
        # Assigning a type to the variable 'val' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 8), for_loop_var_28665))
        # SSA begins for a for statement (line 123)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to setattr(...): (line 124)
        # Processing the call arguments (line 124)
        # Getting the type of 'self' (line 124)
        self_28667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 20), 'self', False)
        # Getting the type of 'kw' (line 124)
        kw_28668 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 26), 'kw', False)
        # Getting the type of 'val' (line 124)
        val_28669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'val', False)
        # Processing the call keyword arguments (line 124)
        kwargs_28670 = {}
        # Getting the type of 'setattr' (line 124)
        setattr_28666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 12), 'setattr', False)
        # Calling setattr(args, kwargs) (line 124)
        setattr_call_result_28671 = invoke(stypy.reporting.localization.Localization(__file__, 124, 12), setattr_28666, *[self_28667, kw_28668, val_28669], **kwargs_28670)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def ensure_finalized(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'ensure_finalized'
        module_type_store = module_type_store.open_function_context('ensure_finalized', 126, 4, False)
        # Assigning a type to the variable 'self' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_localization', localization)
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_type_store', module_type_store)
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_function_name', 'DummyCommand.ensure_finalized')
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_param_names_list', [])
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_varargs_param_name', None)
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_kwargs_param_name', None)
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_call_defaults', defaults)
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_call_varargs', varargs)
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        DummyCommand.ensure_finalized.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DummyCommand.ensure_finalized', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'ensure_finalized', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'ensure_finalized(...)' code ##################

        pass
        
        # ################# End of 'ensure_finalized(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'ensure_finalized' in the type store
        # Getting the type of 'stypy_return_type' (line 126)
        stypy_return_type_28672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28672)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'ensure_finalized'
        return stypy_return_type_28672


# Assigning a type to the variable 'DummyCommand' (line 119)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 0), 'DummyCommand', DummyCommand)
# Declaration of the 'EnvironGuard' class

class EnvironGuard(object, ):

    @norecursion
    def setUp(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setUp'
        module_type_store = module_type_store.open_function_context('setUp', 132, 4, False)
        # Assigning a type to the variable 'self' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironGuard.setUp.__dict__.__setitem__('stypy_localization', localization)
        EnvironGuard.setUp.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironGuard.setUp.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironGuard.setUp.__dict__.__setitem__('stypy_function_name', 'EnvironGuard.setUp')
        EnvironGuard.setUp.__dict__.__setitem__('stypy_param_names_list', [])
        EnvironGuard.setUp.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironGuard.setUp.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironGuard.setUp.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironGuard.setUp.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironGuard.setUp.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironGuard.setUp.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironGuard.setUp', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setUp', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setUp(...)' code ##################

        
        # Call to setUp(...): (line 133)
        # Processing the call keyword arguments (line 133)
        kwargs_28679 = {}
        
        # Call to super(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'EnvironGuard' (line 133)
        EnvironGuard_28674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 14), 'EnvironGuard', False)
        # Getting the type of 'self' (line 133)
        self_28675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 28), 'self', False)
        # Processing the call keyword arguments (line 133)
        kwargs_28676 = {}
        # Getting the type of 'super' (line 133)
        super_28673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'super', False)
        # Calling super(args, kwargs) (line 133)
        super_call_result_28677 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), super_28673, *[EnvironGuard_28674, self_28675], **kwargs_28676)
        
        # Obtaining the member 'setUp' of a type (line 133)
        setUp_28678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 8), super_call_result_28677, 'setUp')
        # Calling setUp(args, kwargs) (line 133)
        setUp_call_result_28680 = invoke(stypy.reporting.localization.Localization(__file__, 133, 8), setUp_28678, *[], **kwargs_28679)
        
        
        # Assigning a Call to a Attribute (line 134):
        
        # Assigning a Call to a Attribute (line 134):
        
        # Call to deepcopy(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'os' (line 134)
        os_28682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 36), 'os', False)
        # Obtaining the member 'environ' of a type (line 134)
        environ_28683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 36), os_28682, 'environ')
        # Processing the call keyword arguments (line 134)
        kwargs_28684 = {}
        # Getting the type of 'deepcopy' (line 134)
        deepcopy_28681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 27), 'deepcopy', False)
        # Calling deepcopy(args, kwargs) (line 134)
        deepcopy_call_result_28685 = invoke(stypy.reporting.localization.Localization(__file__, 134, 27), deepcopy_28681, *[environ_28683], **kwargs_28684)
        
        # Getting the type of 'self' (line 134)
        self_28686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 8), 'self')
        # Setting the type of the member 'old_environ' of a type (line 134)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 8), self_28686, 'old_environ', deepcopy_call_result_28685)
        
        # ################# End of 'setUp(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setUp' in the type store
        # Getting the type of 'stypy_return_type' (line 132)
        stypy_return_type_28687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28687)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setUp'
        return stypy_return_type_28687


    @norecursion
    def tearDown(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'tearDown'
        module_type_store = module_type_store.open_function_context('tearDown', 136, 4, False)
        # Assigning a type to the variable 'self' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_localization', localization)
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_type_store', module_type_store)
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_function_name', 'EnvironGuard.tearDown')
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_param_names_list', [])
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_varargs_param_name', None)
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_kwargs_param_name', None)
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_call_defaults', defaults)
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_call_varargs', varargs)
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        EnvironGuard.tearDown.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironGuard.tearDown', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'tearDown', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'tearDown(...)' code ##################

        
        
        # Call to items(...): (line 137)
        # Processing the call keyword arguments (line 137)
        kwargs_28691 = {}
        # Getting the type of 'self' (line 137)
        self_28688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 26), 'self', False)
        # Obtaining the member 'old_environ' of a type (line 137)
        old_environ_28689 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 26), self_28688, 'old_environ')
        # Obtaining the member 'items' of a type (line 137)
        items_28690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 26), old_environ_28689, 'items')
        # Calling items(args, kwargs) (line 137)
        items_call_result_28692 = invoke(stypy.reporting.localization.Localization(__file__, 137, 26), items_28690, *[], **kwargs_28691)
        
        # Testing the type of a for loop iterable (line 137)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 137, 8), items_call_result_28692)
        # Getting the type of the for loop variable (line 137)
        for_loop_var_28693 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 137, 8), items_call_result_28692)
        # Assigning a type to the variable 'key' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'key', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_28693))
        # Assigning a type to the variable 'value' (line 137)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 137, 8), for_loop_var_28693))
        # SSA begins for a for statement (line 137)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Call to get(...): (line 138)
        # Processing the call arguments (line 138)
        # Getting the type of 'key' (line 138)
        key_28697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 30), 'key', False)
        # Processing the call keyword arguments (line 138)
        kwargs_28698 = {}
        # Getting the type of 'os' (line 138)
        os_28694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'os', False)
        # Obtaining the member 'environ' of a type (line 138)
        environ_28695 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), os_28694, 'environ')
        # Obtaining the member 'get' of a type (line 138)
        get_28696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 15), environ_28695, 'get')
        # Calling get(args, kwargs) (line 138)
        get_call_result_28699 = invoke(stypy.reporting.localization.Localization(__file__, 138, 15), get_28696, *[key_28697], **kwargs_28698)
        
        # Getting the type of 'value' (line 138)
        value_28700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 38), 'value')
        # Applying the binary operator '!=' (line 138)
        result_ne_28701 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 15), '!=', get_call_result_28699, value_28700)
        
        # Testing the type of an if condition (line 138)
        if_condition_28702 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 12), result_ne_28701)
        # Assigning a type to the variable 'if_condition_28702' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'if_condition_28702', if_condition_28702)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Subscript (line 139):
        
        # Assigning a Name to a Subscript (line 139):
        # Getting the type of 'value' (line 139)
        value_28703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 34), 'value')
        # Getting the type of 'os' (line 139)
        os_28704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'os')
        # Obtaining the member 'environ' of a type (line 139)
        environ_28705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), os_28704, 'environ')
        # Getting the type of 'key' (line 139)
        key_28706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 27), 'key')
        # Storing an element on a container (line 139)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 139, 16), environ_28705, (key_28706, value_28703))
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to keys(...): (line 141)
        # Processing the call keyword arguments (line 141)
        kwargs_28710 = {}
        # Getting the type of 'os' (line 141)
        os_28707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'os', False)
        # Obtaining the member 'environ' of a type (line 141)
        environ_28708 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 19), os_28707, 'environ')
        # Obtaining the member 'keys' of a type (line 141)
        keys_28709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 19), environ_28708, 'keys')
        # Calling keys(args, kwargs) (line 141)
        keys_call_result_28711 = invoke(stypy.reporting.localization.Localization(__file__, 141, 19), keys_28709, *[], **kwargs_28710)
        
        # Testing the type of a for loop iterable (line 141)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 141, 8), keys_call_result_28711)
        # Getting the type of the for loop variable (line 141)
        for_loop_var_28712 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 141, 8), keys_call_result_28711)
        # Assigning a type to the variable 'key' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'key', for_loop_var_28712)
        # SSA begins for a for statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Getting the type of 'key' (line 142)
        key_28713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 15), 'key')
        # Getting the type of 'self' (line 142)
        self_28714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 26), 'self')
        # Obtaining the member 'old_environ' of a type (line 142)
        old_environ_28715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 26), self_28714, 'old_environ')
        # Applying the binary operator 'notin' (line 142)
        result_contains_28716 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 15), 'notin', key_28713, old_environ_28715)
        
        # Testing the type of an if condition (line 142)
        if_condition_28717 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 142, 12), result_contains_28716)
        # Assigning a type to the variable 'if_condition_28717' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'if_condition_28717', if_condition_28717)
        # SSA begins for if statement (line 142)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Deleting a member
        # Getting the type of 'os' (line 143)
        os_28718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'os')
        # Obtaining the member 'environ' of a type (line 143)
        environ_28719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 20), os_28718, 'environ')
        
        # Obtaining the type of the subscript
        # Getting the type of 'key' (line 143)
        key_28720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 31), 'key')
        # Getting the type of 'os' (line 143)
        os_28721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 20), 'os')
        # Obtaining the member 'environ' of a type (line 143)
        environ_28722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 20), os_28721, 'environ')
        # Obtaining the member '__getitem__' of a type (line 143)
        getitem___28723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 20), environ_28722, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 143)
        subscript_call_result_28724 = invoke(stypy.reporting.localization.Localization(__file__, 143, 20), getitem___28723, key_28720)
        
        del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 16), environ_28719, subscript_call_result_28724)
        # SSA join for if statement (line 142)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to tearDown(...): (line 145)
        # Processing the call keyword arguments (line 145)
        kwargs_28731 = {}
        
        # Call to super(...): (line 145)
        # Processing the call arguments (line 145)
        # Getting the type of 'EnvironGuard' (line 145)
        EnvironGuard_28726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 14), 'EnvironGuard', False)
        # Getting the type of 'self' (line 145)
        self_28727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 28), 'self', False)
        # Processing the call keyword arguments (line 145)
        kwargs_28728 = {}
        # Getting the type of 'super' (line 145)
        super_28725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'super', False)
        # Calling super(args, kwargs) (line 145)
        super_call_result_28729 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), super_28725, *[EnvironGuard_28726, self_28727], **kwargs_28728)
        
        # Obtaining the member 'tearDown' of a type (line 145)
        tearDown_28730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 8), super_call_result_28729, 'tearDown')
        # Calling tearDown(args, kwargs) (line 145)
        tearDown_call_result_28732 = invoke(stypy.reporting.localization.Localization(__file__, 145, 8), tearDown_28730, *[], **kwargs_28731)
        
        
        # ################# End of 'tearDown(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'tearDown' in the type store
        # Getting the type of 'stypy_return_type' (line 136)
        stypy_return_type_28733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_28733)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'tearDown'
        return stypy_return_type_28733


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 130, 0, False)
        # Assigning a type to the variable 'self' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'EnvironGuard.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'EnvironGuard' (line 130)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 0), 'EnvironGuard', EnvironGuard)

@norecursion
def copy_xxmodule_c(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'copy_xxmodule_c'
    module_type_store = module_type_store.open_function_context('copy_xxmodule_c', 148, 0, False)
    
    # Passed parameters checking function
    copy_xxmodule_c.stypy_localization = localization
    copy_xxmodule_c.stypy_type_of_self = None
    copy_xxmodule_c.stypy_type_store = module_type_store
    copy_xxmodule_c.stypy_function_name = 'copy_xxmodule_c'
    copy_xxmodule_c.stypy_param_names_list = ['directory']
    copy_xxmodule_c.stypy_varargs_param_name = None
    copy_xxmodule_c.stypy_kwargs_param_name = None
    copy_xxmodule_c.stypy_call_defaults = defaults
    copy_xxmodule_c.stypy_call_varargs = varargs
    copy_xxmodule_c.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'copy_xxmodule_c', ['directory'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'copy_xxmodule_c', localization, ['directory'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'copy_xxmodule_c(...)' code ##################

    str_28734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 159, (-1)), 'str', "Helper for tests that need the xxmodule.c source file.\n\n    Example use:\n\n        def test_compile(self):\n            copy_xxmodule_c(self.tmpdir)\n            self.assertIn('xxmodule.c', os.listdir(self.tmpdir))\n\n    If the source file can be found, it will be copied to *directory*.  If not,\n    the test will be skipped.  Errors during copy are not caught.\n    ")
    
    # Assigning a Call to a Name (line 160):
    
    # Assigning a Call to a Name (line 160):
    
    # Call to _get_xxmodule_path(...): (line 160)
    # Processing the call keyword arguments (line 160)
    kwargs_28736 = {}
    # Getting the type of '_get_xxmodule_path' (line 160)
    _get_xxmodule_path_28735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), '_get_xxmodule_path', False)
    # Calling _get_xxmodule_path(args, kwargs) (line 160)
    _get_xxmodule_path_call_result_28737 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), _get_xxmodule_path_28735, *[], **kwargs_28736)
    
    # Assigning a type to the variable 'filename' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 4), 'filename', _get_xxmodule_path_call_result_28737)
    
    # Type idiom detected: calculating its left and rigth part (line 161)
    # Getting the type of 'filename' (line 161)
    filename_28738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 7), 'filename')
    # Getting the type of 'None' (line 161)
    None_28739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 19), 'None')
    
    (may_be_28740, more_types_in_union_28741) = may_be_none(filename_28738, None_28739)

    if may_be_28740:

        if more_types_in_union_28741:
            # Runtime conditional SSA (line 161)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to SkipTest(...): (line 162)
        # Processing the call arguments (line 162)
        str_28744 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 32), 'str', 'cannot find xxmodule.c (test must run in the python build dir)')
        # Processing the call keyword arguments (line 162)
        kwargs_28745 = {}
        # Getting the type of 'unittest' (line 162)
        unittest_28742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 14), 'unittest', False)
        # Obtaining the member 'SkipTest' of a type (line 162)
        SkipTest_28743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 14), unittest_28742, 'SkipTest')
        # Calling SkipTest(args, kwargs) (line 162)
        SkipTest_call_result_28746 = invoke(stypy.reporting.localization.Localization(__file__, 162, 14), SkipTest_28743, *[str_28744], **kwargs_28745)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 162, 8), SkipTest_call_result_28746, 'raise parameter', BaseException)

        if more_types_in_union_28741:
            # SSA join for if statement (line 161)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Call to copy(...): (line 164)
    # Processing the call arguments (line 164)
    # Getting the type of 'filename' (line 164)
    filename_28749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 16), 'filename', False)
    # Getting the type of 'directory' (line 164)
    directory_28750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 26), 'directory', False)
    # Processing the call keyword arguments (line 164)
    kwargs_28751 = {}
    # Getting the type of 'shutil' (line 164)
    shutil_28747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 4), 'shutil', False)
    # Obtaining the member 'copy' of a type (line 164)
    copy_28748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 4), shutil_28747, 'copy')
    # Calling copy(args, kwargs) (line 164)
    copy_call_result_28752 = invoke(stypy.reporting.localization.Localization(__file__, 164, 4), copy_28748, *[filename_28749, directory_28750], **kwargs_28751)
    
    
    # ################# End of 'copy_xxmodule_c(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'copy_xxmodule_c' in the type store
    # Getting the type of 'stypy_return_type' (line 148)
    stypy_return_type_28753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28753)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'copy_xxmodule_c'
    return stypy_return_type_28753

# Assigning a type to the variable 'copy_xxmodule_c' (line 148)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 148, 0), 'copy_xxmodule_c', copy_xxmodule_c)

@norecursion
def _get_xxmodule_path(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_xxmodule_path'
    module_type_store = module_type_store.open_function_context('_get_xxmodule_path', 167, 0, False)
    
    # Passed parameters checking function
    _get_xxmodule_path.stypy_localization = localization
    _get_xxmodule_path.stypy_type_of_self = None
    _get_xxmodule_path.stypy_type_store = module_type_store
    _get_xxmodule_path.stypy_function_name = '_get_xxmodule_path'
    _get_xxmodule_path.stypy_param_names_list = []
    _get_xxmodule_path.stypy_varargs_param_name = None
    _get_xxmodule_path.stypy_kwargs_param_name = None
    _get_xxmodule_path.stypy_call_defaults = defaults
    _get_xxmodule_path.stypy_call_varargs = varargs
    _get_xxmodule_path.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_xxmodule_path', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_xxmodule_path', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_xxmodule_path(...)' code ##################

    
    # Assigning a Call to a Name (line 170):
    
    # Assigning a Call to a Name (line 170):
    
    # Call to get_config_var(...): (line 170)
    # Processing the call arguments (line 170)
    str_28756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 38), 'str', 'srcdir')
    # Processing the call keyword arguments (line 170)
    kwargs_28757 = {}
    # Getting the type of 'sysconfig' (line 170)
    sysconfig_28754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 13), 'sysconfig', False)
    # Obtaining the member 'get_config_var' of a type (line 170)
    get_config_var_28755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 13), sysconfig_28754, 'get_config_var')
    # Calling get_config_var(args, kwargs) (line 170)
    get_config_var_call_result_28758 = invoke(stypy.reporting.localization.Localization(__file__, 170, 13), get_config_var_28755, *[str_28756], **kwargs_28757)
    
    # Assigning a type to the variable 'srcdir' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'srcdir', get_config_var_call_result_28758)
    
    # Assigning a List to a Name (line 171):
    
    # Assigning a List to a Name (line 171):
    
    # Obtaining an instance of the builtin type 'list' (line 171)
    list_28759 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 17), 'list')
    # Adding type elements to the builtin type 'list' instance (line 171)
    # Adding element type (line 171)
    
    # Call to join(...): (line 173)
    # Processing the call arguments (line 173)
    
    # Call to dirname(...): (line 173)
    # Processing the call arguments (line 173)
    # Getting the type of '__file__' (line 173)
    file___28766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 37), '__file__', False)
    # Processing the call keyword arguments (line 173)
    kwargs_28767 = {}
    # Getting the type of 'os' (line 173)
    os_28763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 21), 'os', False)
    # Obtaining the member 'path' of a type (line 173)
    path_28764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 21), os_28763, 'path')
    # Obtaining the member 'dirname' of a type (line 173)
    dirname_28765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 21), path_28764, 'dirname')
    # Calling dirname(args, kwargs) (line 173)
    dirname_call_result_28768 = invoke(stypy.reporting.localization.Localization(__file__, 173, 21), dirname_28765, *[file___28766], **kwargs_28767)
    
    str_28769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 48), 'str', 'xxmodule.c')
    # Processing the call keyword arguments (line 173)
    kwargs_28770 = {}
    # Getting the type of 'os' (line 173)
    os_28760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 173)
    path_28761 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), os_28760, 'path')
    # Obtaining the member 'join' of a type (line 173)
    join_28762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 173, 8), path_28761, 'join')
    # Calling join(args, kwargs) (line 173)
    join_call_result_28771 = invoke(stypy.reporting.localization.Localization(__file__, 173, 8), join_28762, *[dirname_call_result_28768, str_28769], **kwargs_28770)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_28759, join_call_result_28771)
    # Adding element type (line 171)
    
    # Call to join(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'srcdir' (line 175)
    srcdir_28775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 21), 'srcdir', False)
    str_28776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 29), 'str', 'Modules')
    str_28777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 40), 'str', 'xxmodule.c')
    # Processing the call keyword arguments (line 175)
    kwargs_28778 = {}
    # Getting the type of 'os' (line 175)
    os_28772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 175)
    path_28773 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), os_28772, 'path')
    # Obtaining the member 'join' of a type (line 175)
    join_28774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 8), path_28773, 'join')
    # Calling join(args, kwargs) (line 175)
    join_call_result_28779 = invoke(stypy.reporting.localization.Localization(__file__, 175, 8), join_28774, *[srcdir_28775, str_28776, str_28777], **kwargs_28778)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_28759, join_call_result_28779)
    # Adding element type (line 171)
    
    # Call to join(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'srcdir' (line 179)
    srcdir_28783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 21), 'srcdir', False)
    str_28784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 29), 'str', '..')
    str_28785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 35), 'str', '..')
    str_28786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 41), 'str', '..')
    str_28787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 47), 'str', 'Modules')
    str_28788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 58), 'str', 'xxmodule.c')
    # Processing the call keyword arguments (line 179)
    kwargs_28789 = {}
    # Getting the type of 'os' (line 179)
    os_28780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'os', False)
    # Obtaining the member 'path' of a type (line 179)
    path_28781 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), os_28780, 'path')
    # Obtaining the member 'join' of a type (line 179)
    join_28782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 8), path_28781, 'join')
    # Calling join(args, kwargs) (line 179)
    join_call_result_28790 = invoke(stypy.reporting.localization.Localization(__file__, 179, 8), join_28782, *[srcdir_28783, str_28784, str_28785, str_28786, str_28787, str_28788], **kwargs_28789)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 171, 17), list_28759, join_call_result_28790)
    
    # Assigning a type to the variable 'candidates' (line 171)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 4), 'candidates', list_28759)
    
    # Getting the type of 'candidates' (line 181)
    candidates_28791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 16), 'candidates')
    # Testing the type of a for loop iterable (line 181)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 181, 4), candidates_28791)
    # Getting the type of the for loop variable (line 181)
    for_loop_var_28792 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 181, 4), candidates_28791)
    # Assigning a type to the variable 'path' (line 181)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'path', for_loop_var_28792)
    # SSA begins for a for statement (line 181)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Call to exists(...): (line 182)
    # Processing the call arguments (line 182)
    # Getting the type of 'path' (line 182)
    path_28796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 26), 'path', False)
    # Processing the call keyword arguments (line 182)
    kwargs_28797 = {}
    # Getting the type of 'os' (line 182)
    os_28793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 182)
    path_28794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 11), os_28793, 'path')
    # Obtaining the member 'exists' of a type (line 182)
    exists_28795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 11), path_28794, 'exists')
    # Calling exists(args, kwargs) (line 182)
    exists_call_result_28798 = invoke(stypy.reporting.localization.Localization(__file__, 182, 11), exists_28795, *[path_28796], **kwargs_28797)
    
    # Testing the type of an if condition (line 182)
    if_condition_28799 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 182, 8), exists_call_result_28798)
    # Assigning a type to the variable 'if_condition_28799' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'if_condition_28799', if_condition_28799)
    # SSA begins for if statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'path' (line 183)
    path_28800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 19), 'path')
    # Assigning a type to the variable 'stypy_return_type' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'stypy_return_type', path_28800)
    # SSA join for if statement (line 182)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_get_xxmodule_path(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_xxmodule_path' in the type store
    # Getting the type of 'stypy_return_type' (line 167)
    stypy_return_type_28801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28801)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_xxmodule_path'
    return stypy_return_type_28801

# Assigning a type to the variable '_get_xxmodule_path' (line 167)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 0), '_get_xxmodule_path', _get_xxmodule_path)

@norecursion
def fixup_build_ext(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'fixup_build_ext'
    module_type_store = module_type_store.open_function_context('fixup_build_ext', 186, 0, False)
    
    # Passed parameters checking function
    fixup_build_ext.stypy_localization = localization
    fixup_build_ext.stypy_type_of_self = None
    fixup_build_ext.stypy_type_store = module_type_store
    fixup_build_ext.stypy_function_name = 'fixup_build_ext'
    fixup_build_ext.stypy_param_names_list = ['cmd']
    fixup_build_ext.stypy_varargs_param_name = None
    fixup_build_ext.stypy_kwargs_param_name = None
    fixup_build_ext.stypy_call_defaults = defaults
    fixup_build_ext.stypy_call_varargs = varargs
    fixup_build_ext.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'fixup_build_ext', ['cmd'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'fixup_build_ext', localization, ['cmd'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'fixup_build_ext(...)' code ##################

    str_28802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 206, (-1)), 'str', "Function needed to make build_ext tests pass.\n\n    When Python was build with --enable-shared on Unix, -L. is not good\n    enough to find the libpython<blah>.so.  This is because regrtest runs\n    it under a tempdir, not in the top level where the .so lives.  By the\n    time we've gotten here, Python's already been chdir'd to the tempdir.\n\n    When Python was built with in debug mode on Windows, build_ext commands\n    need their debug attribute set, and it is not done automatically for\n    some reason.\n\n    This function handles both of these things.  Example use:\n\n        cmd = build_ext(dist)\n        support.fixup_build_ext(cmd)\n        cmd.ensure_finalized()\n\n    Unlike most other Unix platforms, Mac OS X embeds absolute paths\n    to shared libraries into executables, so the fixup is not needed there.\n    ")
    
    
    # Getting the type of 'os' (line 207)
    os_28803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 7), 'os')
    # Obtaining the member 'name' of a type (line 207)
    name_28804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 7), os_28803, 'name')
    str_28805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 18), 'str', 'nt')
    # Applying the binary operator '==' (line 207)
    result_eq_28806 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 7), '==', name_28804, str_28805)
    
    # Testing the type of an if condition (line 207)
    if_condition_28807 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 4), result_eq_28806)
    # Assigning a type to the variable 'if_condition_28807' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'if_condition_28807', if_condition_28807)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Attribute (line 208):
    
    # Assigning a Call to a Attribute (line 208):
    
    # Call to endswith(...): (line 208)
    # Processing the call arguments (line 208)
    str_28811 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 208, 44), 'str', '_d.exe')
    # Processing the call keyword arguments (line 208)
    kwargs_28812 = {}
    # Getting the type of 'sys' (line 208)
    sys_28808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'sys', False)
    # Obtaining the member 'executable' of a type (line 208)
    executable_28809 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), sys_28808, 'executable')
    # Obtaining the member 'endswith' of a type (line 208)
    endswith_28810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), executable_28809, 'endswith')
    # Calling endswith(args, kwargs) (line 208)
    endswith_call_result_28813 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), endswith_28810, *[str_28811], **kwargs_28812)
    
    # Getting the type of 'cmd' (line 208)
    cmd_28814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'cmd')
    # Setting the type of the member 'debug' of a type (line 208)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 8), cmd_28814, 'debug', endswith_call_result_28813)
    # SSA branch for the else part of an if statement (line 207)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to get_config_var(...): (line 209)
    # Processing the call arguments (line 209)
    str_28817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 34), 'str', 'Py_ENABLE_SHARED')
    # Processing the call keyword arguments (line 209)
    kwargs_28818 = {}
    # Getting the type of 'sysconfig' (line 209)
    sysconfig_28815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 9), 'sysconfig', False)
    # Obtaining the member 'get_config_var' of a type (line 209)
    get_config_var_28816 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 9), sysconfig_28815, 'get_config_var')
    # Calling get_config_var(args, kwargs) (line 209)
    get_config_var_call_result_28819 = invoke(stypy.reporting.localization.Localization(__file__, 209, 9), get_config_var_28816, *[str_28817], **kwargs_28818)
    
    # Testing the type of an if condition (line 209)
    if_condition_28820 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 9), get_config_var_call_result_28819)
    # Assigning a type to the variable 'if_condition_28820' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 9), 'if_condition_28820', if_condition_28820)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 213):
    
    # Assigning a Call to a Name (line 213):
    
    # Call to get_config_var(...): (line 213)
    # Processing the call arguments (line 213)
    str_28823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 45), 'str', 'RUNSHARED')
    # Processing the call keyword arguments (line 213)
    kwargs_28824 = {}
    # Getting the type of 'sysconfig' (line 213)
    sysconfig_28821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 20), 'sysconfig', False)
    # Obtaining the member 'get_config_var' of a type (line 213)
    get_config_var_28822 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 20), sysconfig_28821, 'get_config_var')
    # Calling get_config_var(args, kwargs) (line 213)
    get_config_var_call_result_28825 = invoke(stypy.reporting.localization.Localization(__file__, 213, 20), get_config_var_28822, *[str_28823], **kwargs_28824)
    
    # Assigning a type to the variable 'runshared' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'runshared', get_config_var_call_result_28825)
    
    # Type idiom detected: calculating its left and rigth part (line 214)
    # Getting the type of 'runshared' (line 214)
    runshared_28826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'runshared')
    # Getting the type of 'None' (line 214)
    None_28827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 24), 'None')
    
    (may_be_28828, more_types_in_union_28829) = may_be_none(runshared_28826, None_28827)

    if may_be_28828:

        if more_types_in_union_28829:
            # Runtime conditional SSA (line 214)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a List to a Attribute (line 215):
        
        # Assigning a List to a Attribute (line 215):
        
        # Obtaining an instance of the builtin type 'list' (line 215)
        list_28830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 31), 'list')
        # Adding type elements to the builtin type 'list' instance (line 215)
        # Adding element type (line 215)
        str_28831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 32), 'str', '.')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 31), list_28830, str_28831)
        
        # Getting the type of 'cmd' (line 215)
        cmd_28832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 12), 'cmd')
        # Setting the type of the member 'library_dirs' of a type (line 215)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 12), cmd_28832, 'library_dirs', list_28830)

        if more_types_in_union_28829:
            # Runtime conditional SSA for else branch (line 214)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_28828) or more_types_in_union_28829):
        
        
        # Getting the type of 'sys' (line 217)
        sys_28833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 15), 'sys')
        # Obtaining the member 'platform' of a type (line 217)
        platform_28834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 15), sys_28833, 'platform')
        str_28835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 31), 'str', 'darwin')
        # Applying the binary operator '==' (line 217)
        result_eq_28836 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 15), '==', platform_28834, str_28835)
        
        # Testing the type of an if condition (line 217)
        if_condition_28837 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 12), result_eq_28836)
        # Assigning a type to the variable 'if_condition_28837' (line 217)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'if_condition_28837', if_condition_28837)
        # SSA begins for if statement (line 217)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a List to a Attribute (line 218):
        
        # Assigning a List to a Attribute (line 218):
        
        # Obtaining an instance of the builtin type 'list' (line 218)
        list_28838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 218, 35), 'list')
        # Adding type elements to the builtin type 'list' instance (line 218)
        
        # Getting the type of 'cmd' (line 218)
        cmd_28839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 16), 'cmd')
        # Setting the type of the member 'library_dirs' of a type (line 218)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 16), cmd_28839, 'library_dirs', list_28838)
        # SSA branch for the else part of an if statement (line 217)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Tuple (line 220):
        
        # Assigning a Subscript to a Name (line 220):
        
        # Obtaining the type of the subscript
        int_28840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 16), 'int')
        
        # Call to partition(...): (line 220)
        # Processing the call arguments (line 220)
        str_28843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 58), 'str', '=')
        # Processing the call keyword arguments (line 220)
        kwargs_28844 = {}
        # Getting the type of 'runshared' (line 220)
        runshared_28841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'runshared', False)
        # Obtaining the member 'partition' of a type (line 220)
        partition_28842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), runshared_28841, 'partition')
        # Calling partition(args, kwargs) (line 220)
        partition_call_result_28845 = invoke(stypy.reporting.localization.Localization(__file__, 220, 38), partition_28842, *[str_28843], **kwargs_28844)
        
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___28846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), partition_call_result_28845, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_28847 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), getitem___28846, int_28840)
        
        # Assigning a type to the variable 'tuple_var_assignment_28408' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'tuple_var_assignment_28408', subscript_call_result_28847)
        
        # Assigning a Subscript to a Name (line 220):
        
        # Obtaining the type of the subscript
        int_28848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 16), 'int')
        
        # Call to partition(...): (line 220)
        # Processing the call arguments (line 220)
        str_28851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 58), 'str', '=')
        # Processing the call keyword arguments (line 220)
        kwargs_28852 = {}
        # Getting the type of 'runshared' (line 220)
        runshared_28849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'runshared', False)
        # Obtaining the member 'partition' of a type (line 220)
        partition_28850 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), runshared_28849, 'partition')
        # Calling partition(args, kwargs) (line 220)
        partition_call_result_28853 = invoke(stypy.reporting.localization.Localization(__file__, 220, 38), partition_28850, *[str_28851], **kwargs_28852)
        
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___28854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), partition_call_result_28853, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_28855 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), getitem___28854, int_28848)
        
        # Assigning a type to the variable 'tuple_var_assignment_28409' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'tuple_var_assignment_28409', subscript_call_result_28855)
        
        # Assigning a Subscript to a Name (line 220):
        
        # Obtaining the type of the subscript
        int_28856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 16), 'int')
        
        # Call to partition(...): (line 220)
        # Processing the call arguments (line 220)
        str_28859 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 220, 58), 'str', '=')
        # Processing the call keyword arguments (line 220)
        kwargs_28860 = {}
        # Getting the type of 'runshared' (line 220)
        runshared_28857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 38), 'runshared', False)
        # Obtaining the member 'partition' of a type (line 220)
        partition_28858 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 38), runshared_28857, 'partition')
        # Calling partition(args, kwargs) (line 220)
        partition_call_result_28861 = invoke(stypy.reporting.localization.Localization(__file__, 220, 38), partition_28858, *[str_28859], **kwargs_28860)
        
        # Obtaining the member '__getitem__' of a type (line 220)
        getitem___28862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), partition_call_result_28861, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 220)
        subscript_call_result_28863 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), getitem___28862, int_28856)
        
        # Assigning a type to the variable 'tuple_var_assignment_28410' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'tuple_var_assignment_28410', subscript_call_result_28863)
        
        # Assigning a Name to a Name (line 220):
        # Getting the type of 'tuple_var_assignment_28408' (line 220)
        tuple_var_assignment_28408_28864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'tuple_var_assignment_28408')
        # Assigning a type to the variable 'name' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'name', tuple_var_assignment_28408_28864)
        
        # Assigning a Name to a Name (line 220):
        # Getting the type of 'tuple_var_assignment_28409' (line 220)
        tuple_var_assignment_28409_28865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'tuple_var_assignment_28409')
        # Assigning a type to the variable 'equals' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 22), 'equals', tuple_var_assignment_28409_28865)
        
        # Assigning a Name to a Name (line 220):
        # Getting the type of 'tuple_var_assignment_28410' (line 220)
        tuple_var_assignment_28410_28866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'tuple_var_assignment_28410')
        # Assigning a type to the variable 'value' (line 220)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 30), 'value', tuple_var_assignment_28410_28866)
        
        # Assigning a ListComp to a Attribute (line 221):
        
        # Assigning a ListComp to a Attribute (line 221):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 221)
        # Processing the call arguments (line 221)
        # Getting the type of 'os' (line 221)
        os_28871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 59), 'os', False)
        # Obtaining the member 'pathsep' of a type (line 221)
        pathsep_28872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 59), os_28871, 'pathsep')
        # Processing the call keyword arguments (line 221)
        kwargs_28873 = {}
        # Getting the type of 'value' (line 221)
        value_28869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 47), 'value', False)
        # Obtaining the member 'split' of a type (line 221)
        split_28870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 47), value_28869, 'split')
        # Calling split(args, kwargs) (line 221)
        split_call_result_28874 = invoke(stypy.reporting.localization.Localization(__file__, 221, 47), split_28870, *[pathsep_28872], **kwargs_28873)
        
        comprehension_28875 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 36), split_call_result_28874)
        # Assigning a type to the variable 'd' (line 221)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 36), 'd', comprehension_28875)
        # Getting the type of 'd' (line 221)
        d_28868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 74), 'd')
        # Getting the type of 'd' (line 221)
        d_28867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 36), 'd')
        list_28876 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 36), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 36), list_28876, d_28867)
        # Getting the type of 'cmd' (line 221)
        cmd_28877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 16), 'cmd')
        # Setting the type of the member 'library_dirs' of a type (line 221)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 16), cmd_28877, 'library_dirs', list_28876)
        # SSA join for if statement (line 217)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_28828 and more_types_in_union_28829):
            # SSA join for if statement (line 214)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'fixup_build_ext(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'fixup_build_ext' in the type store
    # Getting the type of 'stypy_return_type' (line 186)
    stypy_return_type_28878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_28878)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'fixup_build_ext'
    return stypy_return_type_28878

# Assigning a type to the variable 'fixup_build_ext' (line 186)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 0), 'fixup_build_ext', fixup_build_ext)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
