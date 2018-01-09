
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.core
2: 
3: The only module that needs to be imported to use the Distutils; provides
4: the 'setup' function (which is to be called from the setup script).  Also
5: indirectly provides the Distribution and Command classes, although they are
6: really defined in distutils.dist and distutils.cmd.
7: '''
8: 
9: __revision__ = "$Id$"
10: 
11: import sys
12: import os
13: 
14: from distutils.debug import DEBUG
15: from distutils.errors import (DistutilsSetupError, DistutilsArgError,
16:                               DistutilsError, CCompilerError)
17: 
18: # Mainly import these so setup scripts can "from distutils.core import" them.
19: from distutils.dist import Distribution
20: from distutils.cmd import Command
21: from distutils.config import PyPIRCCommand
22: from distutils.extension import Extension
23: 
24: # This is a barebones help message generated displayed when the user
25: # runs the setup script with no arguments at all.  More useful help
26: # is generated with various --help options: global help, list commands,
27: # and per-command help.
28: USAGE = '''\
29: usage: %(script)s [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]
30:    or: %(script)s --help [cmd1 cmd2 ...]
31:    or: %(script)s --help-commands
32:    or: %(script)s cmd --help
33: '''
34: 
35: def gen_usage(script_name):
36:     script = os.path.basename(script_name)
37:     return USAGE % {'script': script}
38: 
39: 
40: # Some mild magic to control the behaviour of 'setup()' from 'run_setup()'.
41: _setup_stop_after = None
42: _setup_distribution = None
43: 
44: # Legal keyword arguments for the setup() function
45: setup_keywords = ('distclass', 'script_name', 'script_args', 'options',
46:                   'name', 'version', 'author', 'author_email',
47:                   'maintainer', 'maintainer_email', 'url', 'license',
48:                   'description', 'long_description', 'keywords',
49:                   'platforms', 'classifiers', 'download_url',
50:                   'requires', 'provides', 'obsoletes',
51:                   )
52: 
53: # Legal keyword arguments for the Extension constructor
54: extension_keywords = ('name', 'sources', 'include_dirs',
55:                       'define_macros', 'undef_macros',
56:                       'library_dirs', 'libraries', 'runtime_library_dirs',
57:                       'extra_objects', 'extra_compile_args', 'extra_link_args',
58:                       'swig_opts', 'export_symbols', 'depends', 'language')
59: 
60: def setup(**attrs):
61:     '''The gateway to the Distutils: do everything your setup script needs
62:     to do, in a highly flexible and user-driven way.  Briefly: create a
63:     Distribution instance; find and parse config files; parse the command
64:     line; run each Distutils command found there, customized by the options
65:     supplied to 'setup()' (as keyword arguments), in config files, and on
66:     the command line.
67: 
68:     The Distribution instance might be an instance of a class supplied via
69:     the 'distclass' keyword argument to 'setup'; if no such class is
70:     supplied, then the Distribution class (in dist.py) is instantiated.
71:     All other arguments to 'setup' (except for 'cmdclass') are used to set
72:     attributes of the Distribution instance.
73: 
74:     The 'cmdclass' argument, if supplied, is a dictionary mapping command
75:     names to command classes.  Each command encountered on the command line
76:     will be turned into a command class, which is in turn instantiated; any
77:     class found in 'cmdclass' is used in place of the default, which is
78:     (for command 'foo_bar') class 'foo_bar' in module
79:     'distutils.command.foo_bar'.  The command class must provide a
80:     'user_options' attribute which is a list of option specifiers for
81:     'distutils.fancy_getopt'.  Any command-line options between the current
82:     and the next command are used to set attributes of the current command
83:     object.
84: 
85:     When the entire command-line has been successfully parsed, calls the
86:     'run()' method on each command object in turn.  This method will be
87:     driven entirely by the Distribution object (which each command object
88:     has a reference to, thanks to its constructor), and the
89:     command-specific options that became attributes of each command
90:     object.
91:     '''
92: 
93:     global _setup_stop_after, _setup_distribution
94: 
95:     # Determine the distribution class -- either caller-supplied or
96:     # our Distribution (see below).
97:     klass = attrs.get('distclass')
98:     if klass:
99:         del attrs['distclass']
100:     else:
101:         klass = Distribution
102: 
103:     if 'script_name' not in attrs:
104:         attrs['script_name'] = os.path.basename(sys.argv[0])
105:     if 'script_args' not in attrs:
106:         attrs['script_args'] = sys.argv[1:]
107: 
108:     # Create the Distribution instance, using the remaining arguments
109:     # (ie. everything except distclass) to initialize it
110:     try:
111:         _setup_distribution = dist = klass(attrs)
112:     except DistutilsSetupError, msg:
113:         if 'name' in attrs:
114:             raise SystemExit, "error in %s setup command: %s" % \
115:                   (attrs['name'], msg)
116:         else:
117:             raise SystemExit, "error in setup command: %s" % msg
118: 
119:     if _setup_stop_after == "init":
120:         return dist
121: 
122:     # Find and parse the config file(s): they will override options from
123:     # the setup script, but be overridden by the command line.
124:     dist.parse_config_files()
125: 
126:     if DEBUG:
127:         print "options (after parsing config files):"
128:         dist.dump_option_dicts()
129: 
130:     if _setup_stop_after == "config":
131:         return dist
132: 
133:     # Parse the command line and override config files; any
134:     # command-line errors are the end user's fault, so turn them into
135:     # SystemExit to suppress tracebacks.
136:     try:
137:         ok = dist.parse_command_line()
138:     except DistutilsArgError, msg:
139:         raise SystemExit, gen_usage(dist.script_name) + "\nerror: %s" % msg
140: 
141:     if DEBUG:
142:         print "options (after parsing command line):"
143:         dist.dump_option_dicts()
144: 
145:     if _setup_stop_after == "commandline":
146:         return dist
147: 
148:     # And finally, run all the commands found on the command line.
149:     if ok:
150:         try:
151:             dist.run_commands()
152:         except KeyboardInterrupt:
153:             raise SystemExit, "interrupted"
154:         except (IOError, os.error), exc:
155:             if DEBUG:
156:                 sys.stderr.write("error: %s\n" % (exc,))
157:                 raise
158:             else:
159:                 raise SystemExit, "error: %s" % (exc,)
160: 
161:         except (DistutilsError,
162:                 CCompilerError), msg:
163:             if DEBUG:
164:                 raise
165:             else:
166:                 raise SystemExit, "error: " + str(msg)
167: 
168:     return dist
169: 
170: 
171: def run_setup(script_name, script_args=None, stop_after="run"):
172:     '''Run a setup script in a somewhat controlled environment, and
173:     return the Distribution instance that drives things.  This is useful
174:     if you need to find out the distribution meta-data (passed as
175:     keyword args from 'script' to 'setup()', or the contents of the
176:     config files or command-line.
177: 
178:     'script_name' is a file that will be run with 'execfile()';
179:     'sys.argv[0]' will be replaced with 'script' for the duration of the
180:     call.  'script_args' is a list of strings; if supplied,
181:     'sys.argv[1:]' will be replaced by 'script_args' for the duration of
182:     the call.
183: 
184:     'stop_after' tells 'setup()' when to stop processing; possible
185:     values:
186:       init
187:         stop after the Distribution instance has been created and
188:         populated with the keyword arguments to 'setup()'
189:       config
190:         stop after config files have been parsed (and their data
191:         stored in the Distribution instance)
192:       commandline
193:         stop after the command-line ('sys.argv[1:]' or 'script_args')
194:         have been parsed (and the data stored in the Distribution)
195:       run [default]
196:         stop after all commands have been run (the same as if 'setup()'
197:         had been called in the usual way
198: 
199:     Returns the Distribution instance, which provides all information
200:     used to drive the Distutils.
201:     '''
202:     if stop_after not in ('init', 'config', 'commandline', 'run'):
203:         raise ValueError, "invalid value for 'stop_after': %r" % (stop_after,)
204: 
205:     global _setup_stop_after, _setup_distribution
206:     _setup_stop_after = stop_after
207: 
208:     save_argv = sys.argv
209:     g = {'__file__': script_name}
210:     l = {}
211:     try:
212:         try:
213:             sys.argv[0] = script_name
214:             if script_args is not None:
215:                 sys.argv[1:] = script_args
216:             f = open(script_name)
217:             try:
218:                 exec f.read() in g, l
219:             finally:
220:                 f.close()
221:         finally:
222:             sys.argv = save_argv
223:             _setup_stop_after = None
224:     except SystemExit:
225:         # Hmm, should we do something if exiting with a non-zero code
226:         # (ie. error)?
227:         pass
228:     except:
229:         raise
230: 
231:     if _setup_distribution is None:
232:         raise RuntimeError, \
233:               ("'distutils.core.setup()' was never called -- "
234:                "perhaps '%s' is not a Distutils setup script?") % \
235:               script_name
236: 
237:     # I wonder if the setup script's namespace -- g and l -- would be of
238:     # any interest to callers?
239:     return _setup_distribution
240: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_306418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', "distutils.core\n\nThe only module that needs to be imported to use the Distutils; provides\nthe 'setup' function (which is to be called from the setup script).  Also\nindirectly provides the Distribution and Command classes, although they are\nreally defined in distutils.dist and distutils.cmd.\n")

# Assigning a Str to a Name (line 9):
str_306419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__revision__', str_306419)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import sys' statement (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import os' statement (line 12)
import os

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.debug import DEBUG' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306420 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.debug')

if (type(import_306420) is not StypyTypeError):

    if (import_306420 != 'pyd_module'):
        __import__(import_306420)
        sys_modules_306421 = sys.modules[import_306420]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.debug', sys_modules_306421.module_type_store, module_type_store, ['DEBUG'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_306421, sys_modules_306421.module_type_store, module_type_store)
    else:
        from distutils.debug import DEBUG

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.debug', None, module_type_store, ['DEBUG'], [DEBUG])

else:
    # Assigning a type to the variable 'distutils.debug' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.debug', import_306420)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.errors import DistutilsSetupError, DistutilsArgError, DistutilsError, CCompilerError' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306422 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors')

if (type(import_306422) is not StypyTypeError):

    if (import_306422 != 'pyd_module'):
        __import__(import_306422)
        sys_modules_306423 = sys.modules[import_306422]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', sys_modules_306423.module_type_store, module_type_store, ['DistutilsSetupError', 'DistutilsArgError', 'DistutilsError', 'CCompilerError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_306423, sys_modules_306423.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsSetupError, DistutilsArgError, DistutilsError, CCompilerError

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', None, module_type_store, ['DistutilsSetupError', 'DistutilsArgError', 'DistutilsError', 'CCompilerError'], [DistutilsSetupError, DistutilsArgError, DistutilsError, CCompilerError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.errors', import_306422)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'from distutils.dist import Distribution' statement (line 19)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306424 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.dist')

if (type(import_306424) is not StypyTypeError):

    if (import_306424 != 'pyd_module'):
        __import__(import_306424)
        sys_modules_306425 = sys.modules[import_306424]
        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.dist', sys_modules_306425.module_type_store, module_type_store, ['Distribution'])
        nest_module(stypy.reporting.localization.Localization(__file__, 19, 0), __file__, sys_modules_306425, sys_modules_306425.module_type_store, module_type_store)
    else:
        from distutils.dist import Distribution

        import_from_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.dist', None, module_type_store, ['Distribution'], [Distribution])

else:
    # Assigning a type to the variable 'distutils.dist' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'distutils.dist', import_306424)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils.cmd import Command' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306426 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.cmd')

if (type(import_306426) is not StypyTypeError):

    if (import_306426 != 'pyd_module'):
        __import__(import_306426)
        sys_modules_306427 = sys.modules[import_306426]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.cmd', sys_modules_306427.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_306427, sys_modules_306427.module_type_store, module_type_store)
    else:
        from distutils.cmd import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.cmd', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.cmd' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.cmd', import_306426)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from distutils.config import PyPIRCCommand' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306428 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.config')

if (type(import_306428) is not StypyTypeError):

    if (import_306428 != 'pyd_module'):
        __import__(import_306428)
        sys_modules_306429 = sys.modules[import_306428]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.config', sys_modules_306429.module_type_store, module_type_store, ['PyPIRCCommand'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_306429, sys_modules_306429.module_type_store, module_type_store)
    else:
        from distutils.config import PyPIRCCommand

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.config', None, module_type_store, ['PyPIRCCommand'], [PyPIRCCommand])

else:
    # Assigning a type to the variable 'distutils.config' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.config', import_306428)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from distutils.extension import Extension' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_306430 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.extension')

if (type(import_306430) is not StypyTypeError):

    if (import_306430 != 'pyd_module'):
        __import__(import_306430)
        sys_modules_306431 = sys.modules[import_306430]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.extension', sys_modules_306431.module_type_store, module_type_store, ['Extension'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_306431, sys_modules_306431.module_type_store, module_type_store)
    else:
        from distutils.extension import Extension

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.extension', None, module_type_store, ['Extension'], [Extension])

else:
    # Assigning a type to the variable 'distutils.extension' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.extension', import_306430)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')


# Assigning a Str to a Name (line 28):
str_306432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', 'usage: %(script)s [global_opts] cmd1 [cmd1_opts] [cmd2 [cmd2_opts] ...]\n   or: %(script)s --help [cmd1 cmd2 ...]\n   or: %(script)s --help-commands\n   or: %(script)s cmd --help\n')
# Assigning a type to the variable 'USAGE' (line 28)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'USAGE', str_306432)

@norecursion
def gen_usage(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'gen_usage'
    module_type_store = module_type_store.open_function_context('gen_usage', 35, 0, False)
    
    # Passed parameters checking function
    gen_usage.stypy_localization = localization
    gen_usage.stypy_type_of_self = None
    gen_usage.stypy_type_store = module_type_store
    gen_usage.stypy_function_name = 'gen_usage'
    gen_usage.stypy_param_names_list = ['script_name']
    gen_usage.stypy_varargs_param_name = None
    gen_usage.stypy_kwargs_param_name = None
    gen_usage.stypy_call_defaults = defaults
    gen_usage.stypy_call_varargs = varargs
    gen_usage.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'gen_usage', ['script_name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'gen_usage', localization, ['script_name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'gen_usage(...)' code ##################

    
    # Assigning a Call to a Name (line 36):
    
    # Call to basename(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'script_name' (line 36)
    script_name_306436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 30), 'script_name', False)
    # Processing the call keyword arguments (line 36)
    kwargs_306437 = {}
    # Getting the type of 'os' (line 36)
    os_306433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 36)
    path_306434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), os_306433, 'path')
    # Obtaining the member 'basename' of a type (line 36)
    basename_306435 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), path_306434, 'basename')
    # Calling basename(args, kwargs) (line 36)
    basename_call_result_306438 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), basename_306435, *[script_name_306436], **kwargs_306437)
    
    # Assigning a type to the variable 'script' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'script', basename_call_result_306438)
    # Getting the type of 'USAGE' (line 37)
    USAGE_306439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'USAGE')
    
    # Obtaining an instance of the builtin type 'dict' (line 37)
    dict_306440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 19), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 37)
    # Adding element type (key, value) (line 37)
    str_306441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 20), 'str', 'script')
    # Getting the type of 'script' (line 37)
    script_306442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 30), 'script')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 19), dict_306440, (str_306441, script_306442))
    
    # Applying the binary operator '%' (line 37)
    result_mod_306443 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 11), '%', USAGE_306439, dict_306440)
    
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', result_mod_306443)
    
    # ################# End of 'gen_usage(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'gen_usage' in the type store
    # Getting the type of 'stypy_return_type' (line 35)
    stypy_return_type_306444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_306444)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'gen_usage'
    return stypy_return_type_306444

# Assigning a type to the variable 'gen_usage' (line 35)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 0), 'gen_usage', gen_usage)

# Assigning a Name to a Name (line 41):
# Getting the type of 'None' (line 41)
None_306445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 20), 'None')
# Assigning a type to the variable '_setup_stop_after' (line 41)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 0), '_setup_stop_after', None_306445)

# Assigning a Name to a Name (line 42):
# Getting the type of 'None' (line 42)
None_306446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 22), 'None')
# Assigning a type to the variable '_setup_distribution' (line 42)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 0), '_setup_distribution', None_306446)

# Assigning a Tuple to a Name (line 45):

# Obtaining an instance of the builtin type 'tuple' (line 45)
tuple_306447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 45)
# Adding element type (line 45)
str_306448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 18), 'str', 'distclass')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306448)
# Adding element type (line 45)
str_306449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 31), 'str', 'script_name')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306449)
# Adding element type (line 45)
str_306450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 46), 'str', 'script_args')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306450)
# Adding element type (line 45)
str_306451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 61), 'str', 'options')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306451)
# Adding element type (line 45)
str_306452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 18), 'str', 'name')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306452)
# Adding element type (line 45)
str_306453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 26), 'str', 'version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306453)
# Adding element type (line 45)
str_306454 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 37), 'str', 'author')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306454)
# Adding element type (line 45)
str_306455 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 47), 'str', 'author_email')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306455)
# Adding element type (line 45)
str_306456 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'str', 'maintainer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306456)
# Adding element type (line 45)
str_306457 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 32), 'str', 'maintainer_email')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306457)
# Adding element type (line 45)
str_306458 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 52), 'str', 'url')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306458)
# Adding element type (line 45)
str_306459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 59), 'str', 'license')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306459)
# Adding element type (line 45)
str_306460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 18), 'str', 'description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306460)
# Adding element type (line 45)
str_306461 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 33), 'str', 'long_description')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306461)
# Adding element type (line 45)
str_306462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 53), 'str', 'keywords')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306462)
# Adding element type (line 45)
str_306463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 18), 'str', 'platforms')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306463)
# Adding element type (line 45)
str_306464 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 31), 'str', 'classifiers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306464)
# Adding element type (line 45)
str_306465 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 46), 'str', 'download_url')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306465)
# Adding element type (line 45)
str_306466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 18), 'str', 'requires')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306466)
# Adding element type (line 45)
str_306467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 30), 'str', 'provides')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306467)
# Adding element type (line 45)
str_306468 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 42), 'str', 'obsoletes')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 18), tuple_306447, str_306468)

# Assigning a type to the variable 'setup_keywords' (line 45)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 0), 'setup_keywords', tuple_306447)

# Assigning a Tuple to a Name (line 54):

# Obtaining an instance of the builtin type 'tuple' (line 54)
tuple_306469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 22), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 54)
# Adding element type (line 54)
str_306470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 22), 'str', 'name')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306470)
# Adding element type (line 54)
str_306471 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 30), 'str', 'sources')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306471)
# Adding element type (line 54)
str_306472 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 41), 'str', 'include_dirs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306472)
# Adding element type (line 54)
str_306473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'str', 'define_macros')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306473)
# Adding element type (line 54)
str_306474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 39), 'str', 'undef_macros')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306474)
# Adding element type (line 54)
str_306475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 22), 'str', 'library_dirs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306475)
# Adding element type (line 54)
str_306476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 38), 'str', 'libraries')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306476)
# Adding element type (line 54)
str_306477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 51), 'str', 'runtime_library_dirs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306477)
# Adding element type (line 54)
str_306478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'str', 'extra_objects')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306478)
# Adding element type (line 54)
str_306479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 39), 'str', 'extra_compile_args')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306479)
# Adding element type (line 54)
str_306480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 61), 'str', 'extra_link_args')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306480)
# Adding element type (line 54)
str_306481 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 22), 'str', 'swig_opts')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306481)
# Adding element type (line 54)
str_306482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 35), 'str', 'export_symbols')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306482)
# Adding element type (line 54)
str_306483 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 53), 'str', 'depends')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306483)
# Adding element type (line 54)
str_306484 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 58, 64), 'str', 'language')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 22), tuple_306469, str_306484)

# Assigning a type to the variable 'extension_keywords' (line 54)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 0), 'extension_keywords', tuple_306469)

@norecursion
def setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'setup'
    module_type_store = module_type_store.open_function_context('setup', 60, 0, False)
    
    # Passed parameters checking function
    setup.stypy_localization = localization
    setup.stypy_type_of_self = None
    setup.stypy_type_store = module_type_store
    setup.stypy_function_name = 'setup'
    setup.stypy_param_names_list = []
    setup.stypy_varargs_param_name = None
    setup.stypy_kwargs_param_name = 'attrs'
    setup.stypy_call_defaults = defaults
    setup.stypy_call_varargs = varargs
    setup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'setup', [], None, 'attrs', defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'setup', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'setup(...)' code ##################

    str_306485 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'str', "The gateway to the Distutils: do everything your setup script needs\n    to do, in a highly flexible and user-driven way.  Briefly: create a\n    Distribution instance; find and parse config files; parse the command\n    line; run each Distutils command found there, customized by the options\n    supplied to 'setup()' (as keyword arguments), in config files, and on\n    the command line.\n\n    The Distribution instance might be an instance of a class supplied via\n    the 'distclass' keyword argument to 'setup'; if no such class is\n    supplied, then the Distribution class (in dist.py) is instantiated.\n    All other arguments to 'setup' (except for 'cmdclass') are used to set\n    attributes of the Distribution instance.\n\n    The 'cmdclass' argument, if supplied, is a dictionary mapping command\n    names to command classes.  Each command encountered on the command line\n    will be turned into a command class, which is in turn instantiated; any\n    class found in 'cmdclass' is used in place of the default, which is\n    (for command 'foo_bar') class 'foo_bar' in module\n    'distutils.command.foo_bar'.  The command class must provide a\n    'user_options' attribute which is a list of option specifiers for\n    'distutils.fancy_getopt'.  Any command-line options between the current\n    and the next command are used to set attributes of the current command\n    object.\n\n    When the entire command-line has been successfully parsed, calls the\n    'run()' method on each command object in turn.  This method will be\n    driven entirely by the Distribution object (which each command object\n    has a reference to, thanks to its constructor), and the\n    command-specific options that became attributes of each command\n    object.\n    ")
    # Marking variables as global (line 93)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 93, 4), '_setup_stop_after')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 93, 4), '_setup_distribution')
    
    # Assigning a Call to a Name (line 97):
    
    # Call to get(...): (line 97)
    # Processing the call arguments (line 97)
    str_306488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, 22), 'str', 'distclass')
    # Processing the call keyword arguments (line 97)
    kwargs_306489 = {}
    # Getting the type of 'attrs' (line 97)
    attrs_306486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'attrs', False)
    # Obtaining the member 'get' of a type (line 97)
    get_306487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), attrs_306486, 'get')
    # Calling get(args, kwargs) (line 97)
    get_call_result_306490 = invoke(stypy.reporting.localization.Localization(__file__, 97, 12), get_306487, *[str_306488], **kwargs_306489)
    
    # Assigning a type to the variable 'klass' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'klass', get_call_result_306490)
    
    # Getting the type of 'klass' (line 98)
    klass_306491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 7), 'klass')
    # Testing the type of an if condition (line 98)
    if_condition_306492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 98, 4), klass_306491)
    # Assigning a type to the variable 'if_condition_306492' (line 98)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 4), 'if_condition_306492', if_condition_306492)
    # SSA begins for if statement (line 98)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Deleting a member
    # Getting the type of 'attrs' (line 99)
    attrs_306493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'attrs')
    
    # Obtaining the type of the subscript
    str_306494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 18), 'str', 'distclass')
    # Getting the type of 'attrs' (line 99)
    attrs_306495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'attrs')
    # Obtaining the member '__getitem__' of a type (line 99)
    getitem___306496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 12), attrs_306495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 99)
    subscript_call_result_306497 = invoke(stypy.reporting.localization.Localization(__file__, 99, 12), getitem___306496, str_306494)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 99, 8), attrs_306493, subscript_call_result_306497)
    # SSA branch for the else part of an if statement (line 98)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Name to a Name (line 101):
    # Getting the type of 'Distribution' (line 101)
    Distribution_306498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'Distribution')
    # Assigning a type to the variable 'klass' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'klass', Distribution_306498)
    # SSA join for if statement (line 98)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_306499 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 7), 'str', 'script_name')
    # Getting the type of 'attrs' (line 103)
    attrs_306500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 28), 'attrs')
    # Applying the binary operator 'notin' (line 103)
    result_contains_306501 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 7), 'notin', str_306499, attrs_306500)
    
    # Testing the type of an if condition (line 103)
    if_condition_306502 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 4), result_contains_306501)
    # Assigning a type to the variable 'if_condition_306502' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 4), 'if_condition_306502', if_condition_306502)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Subscript (line 104):
    
    # Call to basename(...): (line 104)
    # Processing the call arguments (line 104)
    
    # Obtaining the type of the subscript
    int_306506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 57), 'int')
    # Getting the type of 'sys' (line 104)
    sys_306507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 48), 'sys', False)
    # Obtaining the member 'argv' of a type (line 104)
    argv_306508 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 48), sys_306507, 'argv')
    # Obtaining the member '__getitem__' of a type (line 104)
    getitem___306509 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 48), argv_306508, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 104)
    subscript_call_result_306510 = invoke(stypy.reporting.localization.Localization(__file__, 104, 48), getitem___306509, int_306506)
    
    # Processing the call keyword arguments (line 104)
    kwargs_306511 = {}
    # Getting the type of 'os' (line 104)
    os_306503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 31), 'os', False)
    # Obtaining the member 'path' of a type (line 104)
    path_306504 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 31), os_306503, 'path')
    # Obtaining the member 'basename' of a type (line 104)
    basename_306505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 31), path_306504, 'basename')
    # Calling basename(args, kwargs) (line 104)
    basename_call_result_306512 = invoke(stypy.reporting.localization.Localization(__file__, 104, 31), basename_306505, *[subscript_call_result_306510], **kwargs_306511)
    
    # Getting the type of 'attrs' (line 104)
    attrs_306513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'attrs')
    str_306514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 14), 'str', 'script_name')
    # Storing an element on a container (line 104)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 104, 8), attrs_306513, (str_306514, basename_call_result_306512))
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    str_306515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 7), 'str', 'script_args')
    # Getting the type of 'attrs' (line 105)
    attrs_306516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 28), 'attrs')
    # Applying the binary operator 'notin' (line 105)
    result_contains_306517 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 7), 'notin', str_306515, attrs_306516)
    
    # Testing the type of an if condition (line 105)
    if_condition_306518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 4), result_contains_306517)
    # Assigning a type to the variable 'if_condition_306518' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'if_condition_306518', if_condition_306518)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Subscript to a Subscript (line 106):
    
    # Obtaining the type of the subscript
    int_306519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 40), 'int')
    slice_306520 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 106, 31), int_306519, None, None)
    # Getting the type of 'sys' (line 106)
    sys_306521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 31), 'sys')
    # Obtaining the member 'argv' of a type (line 106)
    argv_306522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 31), sys_306521, 'argv')
    # Obtaining the member '__getitem__' of a type (line 106)
    getitem___306523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 31), argv_306522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 106)
    subscript_call_result_306524 = invoke(stypy.reporting.localization.Localization(__file__, 106, 31), getitem___306523, slice_306520)
    
    # Getting the type of 'attrs' (line 106)
    attrs_306525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'attrs')
    str_306526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 14), 'str', 'script_args')
    # Storing an element on a container (line 106)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 8), attrs_306525, (str_306526, subscript_call_result_306524))
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 110)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Multiple assignment of 2 elements.
    
    # Call to klass(...): (line 111)
    # Processing the call arguments (line 111)
    # Getting the type of 'attrs' (line 111)
    attrs_306528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 43), 'attrs', False)
    # Processing the call keyword arguments (line 111)
    kwargs_306529 = {}
    # Getting the type of 'klass' (line 111)
    klass_306527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 37), 'klass', False)
    # Calling klass(args, kwargs) (line 111)
    klass_call_result_306530 = invoke(stypy.reporting.localization.Localization(__file__, 111, 37), klass_306527, *[attrs_306528], **kwargs_306529)
    
    # Assigning a type to the variable 'dist' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'dist', klass_call_result_306530)
    # Getting the type of 'dist' (line 111)
    dist_306531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 30), 'dist')
    # Assigning a type to the variable '_setup_distribution' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), '_setup_distribution', dist_306531)
    # SSA branch for the except part of a try statement (line 110)
    # SSA branch for the except 'DistutilsSetupError' branch of a try statement (line 110)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'DistutilsSetupError' (line 112)
    DistutilsSetupError_306532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'DistutilsSetupError')
    # Assigning a type to the variable 'msg' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'msg', DistutilsSetupError_306532)
    
    
    str_306533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 11), 'str', 'name')
    # Getting the type of 'attrs' (line 113)
    attrs_306534 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 21), 'attrs')
    # Applying the binary operator 'in' (line 113)
    result_contains_306535 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 11), 'in', str_306533, attrs_306534)
    
    # Testing the type of an if condition (line 113)
    if_condition_306536 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 8), result_contains_306535)
    # Assigning a type to the variable 'if_condition_306536' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'if_condition_306536', if_condition_306536)
    # SSA begins for if statement (line 113)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'SystemExit' (line 114)
    SystemExit_306537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'SystemExit')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 114, 12), SystemExit_306537, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 113)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'SystemExit' (line 117)
    SystemExit_306538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 18), 'SystemExit')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 117, 12), SystemExit_306538, 'raise parameter', BaseException)
    # SSA join for if statement (line 113)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 110)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of '_setup_stop_after' (line 119)
    _setup_stop_after_306539 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 7), '_setup_stop_after')
    str_306540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 28), 'str', 'init')
    # Applying the binary operator '==' (line 119)
    result_eq_306541 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 7), '==', _setup_stop_after_306539, str_306540)
    
    # Testing the type of an if condition (line 119)
    if_condition_306542 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 4), result_eq_306541)
    # Assigning a type to the variable 'if_condition_306542' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'if_condition_306542', if_condition_306542)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'dist' (line 120)
    dist_306543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'dist')
    # Assigning a type to the variable 'stypy_return_type' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'stypy_return_type', dist_306543)
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to parse_config_files(...): (line 124)
    # Processing the call keyword arguments (line 124)
    kwargs_306546 = {}
    # Getting the type of 'dist' (line 124)
    dist_306544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'dist', False)
    # Obtaining the member 'parse_config_files' of a type (line 124)
    parse_config_files_306545 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 4), dist_306544, 'parse_config_files')
    # Calling parse_config_files(args, kwargs) (line 124)
    parse_config_files_call_result_306547 = invoke(stypy.reporting.localization.Localization(__file__, 124, 4), parse_config_files_306545, *[], **kwargs_306546)
    
    
    # Getting the type of 'DEBUG' (line 126)
    DEBUG_306548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 7), 'DEBUG')
    # Testing the type of an if condition (line 126)
    if_condition_306549 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 4), DEBUG_306548)
    # Assigning a type to the variable 'if_condition_306549' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'if_condition_306549', if_condition_306549)
    # SSA begins for if statement (line 126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_306550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 14), 'str', 'options (after parsing config files):')
    
    # Call to dump_option_dicts(...): (line 128)
    # Processing the call keyword arguments (line 128)
    kwargs_306553 = {}
    # Getting the type of 'dist' (line 128)
    dist_306551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'dist', False)
    # Obtaining the member 'dump_option_dicts' of a type (line 128)
    dump_option_dicts_306552 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 128, 8), dist_306551, 'dump_option_dicts')
    # Calling dump_option_dicts(args, kwargs) (line 128)
    dump_option_dicts_call_result_306554 = invoke(stypy.reporting.localization.Localization(__file__, 128, 8), dump_option_dicts_306552, *[], **kwargs_306553)
    
    # SSA join for if statement (line 126)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of '_setup_stop_after' (line 130)
    _setup_stop_after_306555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 7), '_setup_stop_after')
    str_306556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 28), 'str', 'config')
    # Applying the binary operator '==' (line 130)
    result_eq_306557 = python_operator(stypy.reporting.localization.Localization(__file__, 130, 7), '==', _setup_stop_after_306555, str_306556)
    
    # Testing the type of an if condition (line 130)
    if_condition_306558 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 4), result_eq_306557)
    # Assigning a type to the variable 'if_condition_306558' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 4), 'if_condition_306558', if_condition_306558)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'dist' (line 131)
    dist_306559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'dist')
    # Assigning a type to the variable 'stypy_return_type' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', dist_306559)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 136)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 137):
    
    # Call to parse_command_line(...): (line 137)
    # Processing the call keyword arguments (line 137)
    kwargs_306562 = {}
    # Getting the type of 'dist' (line 137)
    dist_306560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 13), 'dist', False)
    # Obtaining the member 'parse_command_line' of a type (line 137)
    parse_command_line_306561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 13), dist_306560, 'parse_command_line')
    # Calling parse_command_line(args, kwargs) (line 137)
    parse_command_line_call_result_306563 = invoke(stypy.reporting.localization.Localization(__file__, 137, 13), parse_command_line_306561, *[], **kwargs_306562)
    
    # Assigning a type to the variable 'ok' (line 137)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 137, 8), 'ok', parse_command_line_call_result_306563)
    # SSA branch for the except part of a try statement (line 136)
    # SSA branch for the except 'DistutilsArgError' branch of a try statement (line 136)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'DistutilsArgError' (line 138)
    DistutilsArgError_306564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'DistutilsArgError')
    # Assigning a type to the variable 'msg' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'msg', DistutilsArgError_306564)
    # Getting the type of 'SystemExit' (line 139)
    SystemExit_306565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 14), 'SystemExit')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 8), SystemExit_306565, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 136)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'DEBUG' (line 141)
    DEBUG_306566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 7), 'DEBUG')
    # Testing the type of an if condition (line 141)
    if_condition_306567 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 4), DEBUG_306566)
    # Assigning a type to the variable 'if_condition_306567' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'if_condition_306567', if_condition_306567)
    # SSA begins for if statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    str_306568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 14), 'str', 'options (after parsing command line):')
    
    # Call to dump_option_dicts(...): (line 143)
    # Processing the call keyword arguments (line 143)
    kwargs_306571 = {}
    # Getting the type of 'dist' (line 143)
    dist_306569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 8), 'dist', False)
    # Obtaining the member 'dump_option_dicts' of a type (line 143)
    dump_option_dicts_306570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 8), dist_306569, 'dump_option_dicts')
    # Calling dump_option_dicts(args, kwargs) (line 143)
    dump_option_dicts_call_result_306572 = invoke(stypy.reporting.localization.Localization(__file__, 143, 8), dump_option_dicts_306570, *[], **kwargs_306571)
    
    # SSA join for if statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of '_setup_stop_after' (line 145)
    _setup_stop_after_306573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 7), '_setup_stop_after')
    str_306574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 28), 'str', 'commandline')
    # Applying the binary operator '==' (line 145)
    result_eq_306575 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 7), '==', _setup_stop_after_306573, str_306574)
    
    # Testing the type of an if condition (line 145)
    if_condition_306576 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 4), result_eq_306575)
    # Assigning a type to the variable 'if_condition_306576' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'if_condition_306576', if_condition_306576)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'dist' (line 146)
    dist_306577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 15), 'dist')
    # Assigning a type to the variable 'stypy_return_type' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 8), 'stypy_return_type', dist_306577)
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'ok' (line 149)
    ok_306578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 7), 'ok')
    # Testing the type of an if condition (line 149)
    if_condition_306579 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 4), ok_306578)
    # Assigning a type to the variable 'if_condition_306579' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 4), 'if_condition_306579', if_condition_306579)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to run_commands(...): (line 151)
    # Processing the call keyword arguments (line 151)
    kwargs_306582 = {}
    # Getting the type of 'dist' (line 151)
    dist_306580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'dist', False)
    # Obtaining the member 'run_commands' of a type (line 151)
    run_commands_306581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), dist_306580, 'run_commands')
    # Calling run_commands(args, kwargs) (line 151)
    run_commands_call_result_306583 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), run_commands_306581, *[], **kwargs_306582)
    
    # SSA branch for the except part of a try statement (line 150)
    # SSA branch for the except 'KeyboardInterrupt' branch of a try statement (line 150)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'SystemExit' (line 153)
    SystemExit_306584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 18), 'SystemExit')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 153, 12), SystemExit_306584, 'raise parameter', BaseException)
    # SSA branch for the except 'Tuple' branch of a try statement (line 150)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 154)
    tuple_306585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 154)
    # Adding element type (line 154)
    # Getting the type of 'IOError' (line 154)
    IOError_306586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'IOError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), tuple_306585, IOError_306586)
    # Adding element type (line 154)
    # Getting the type of 'os' (line 154)
    os_306587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 25), 'os')
    # Obtaining the member 'error' of a type (line 154)
    error_306588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 25), os_306587, 'error')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 154, 16), tuple_306585, error_306588)
    
    # Assigning a type to the variable 'exc' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'exc', tuple_306585)
    
    # Getting the type of 'DEBUG' (line 155)
    DEBUG_306589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 15), 'DEBUG')
    # Testing the type of an if condition (line 155)
    if_condition_306590 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 12), DEBUG_306589)
    # Assigning a type to the variable 'if_condition_306590' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'if_condition_306590', if_condition_306590)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to write(...): (line 156)
    # Processing the call arguments (line 156)
    str_306594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 33), 'str', 'error: %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 156)
    tuple_306595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 156)
    # Adding element type (line 156)
    # Getting the type of 'exc' (line 156)
    exc_306596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 50), 'exc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 156, 50), tuple_306595, exc_306596)
    
    # Applying the binary operator '%' (line 156)
    result_mod_306597 = python_operator(stypy.reporting.localization.Localization(__file__, 156, 33), '%', str_306594, tuple_306595)
    
    # Processing the call keyword arguments (line 156)
    kwargs_306598 = {}
    # Getting the type of 'sys' (line 156)
    sys_306591 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 156)
    stderr_306592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), sys_306591, 'stderr')
    # Obtaining the member 'write' of a type (line 156)
    write_306593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), stderr_306592, 'write')
    # Calling write(args, kwargs) (line 156)
    write_call_result_306599 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), write_306593, *[result_mod_306597], **kwargs_306598)
    
    # SSA branch for the else part of an if statement (line 155)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'SystemExit' (line 159)
    SystemExit_306600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 22), 'SystemExit')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 159, 16), SystemExit_306600, 'raise parameter', BaseException)
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except 'Tuple' branch of a try statement (line 150)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 161)
    tuple_306601 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 161)
    # Adding element type (line 161)
    # Getting the type of 'DistutilsError' (line 161)
    DistutilsError_306602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'DistutilsError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 16), tuple_306601, DistutilsError_306602)
    # Adding element type (line 161)
    # Getting the type of 'CCompilerError' (line 162)
    CCompilerError_306603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 16), 'CCompilerError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 161, 16), tuple_306601, CCompilerError_306603)
    
    # Assigning a type to the variable 'msg' (line 161)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'msg', tuple_306601)
    
    # Getting the type of 'DEBUG' (line 163)
    DEBUG_306604 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 15), 'DEBUG')
    # Testing the type of an if condition (line 163)
    if_condition_306605 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 12), DEBUG_306604)
    # Assigning a type to the variable 'if_condition_306605' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'if_condition_306605', if_condition_306605)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA branch for the else part of an if statement (line 163)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'SystemExit' (line 166)
    SystemExit_306606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'SystemExit')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 166, 16), SystemExit_306606, 'raise parameter', BaseException)
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'dist' (line 168)
    dist_306607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 11), 'dist')
    # Assigning a type to the variable 'stypy_return_type' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'stypy_return_type', dist_306607)
    
    # ################# End of 'setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'setup' in the type store
    # Getting the type of 'stypy_return_type' (line 60)
    stypy_return_type_306608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_306608)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'setup'
    return stypy_return_type_306608

# Assigning a type to the variable 'setup' (line 60)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'setup', setup)

@norecursion
def run_setup(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 171)
    None_306609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 39), 'None')
    str_306610 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 56), 'str', 'run')
    defaults = [None_306609, str_306610]
    # Create a new context for function 'run_setup'
    module_type_store = module_type_store.open_function_context('run_setup', 171, 0, False)
    
    # Passed parameters checking function
    run_setup.stypy_localization = localization
    run_setup.stypy_type_of_self = None
    run_setup.stypy_type_store = module_type_store
    run_setup.stypy_function_name = 'run_setup'
    run_setup.stypy_param_names_list = ['script_name', 'script_args', 'stop_after']
    run_setup.stypy_varargs_param_name = None
    run_setup.stypy_kwargs_param_name = None
    run_setup.stypy_call_defaults = defaults
    run_setup.stypy_call_varargs = varargs
    run_setup.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'run_setup', ['script_name', 'script_args', 'stop_after'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'run_setup', localization, ['script_name', 'script_args', 'stop_after'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'run_setup(...)' code ##################

    str_306611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, (-1)), 'str', "Run a setup script in a somewhat controlled environment, and\n    return the Distribution instance that drives things.  This is useful\n    if you need to find out the distribution meta-data (passed as\n    keyword args from 'script' to 'setup()', or the contents of the\n    config files or command-line.\n\n    'script_name' is a file that will be run with 'execfile()';\n    'sys.argv[0]' will be replaced with 'script' for the duration of the\n    call.  'script_args' is a list of strings; if supplied,\n    'sys.argv[1:]' will be replaced by 'script_args' for the duration of\n    the call.\n\n    'stop_after' tells 'setup()' when to stop processing; possible\n    values:\n      init\n        stop after the Distribution instance has been created and\n        populated with the keyword arguments to 'setup()'\n      config\n        stop after config files have been parsed (and their data\n        stored in the Distribution instance)\n      commandline\n        stop after the command-line ('sys.argv[1:]' or 'script_args')\n        have been parsed (and the data stored in the Distribution)\n      run [default]\n        stop after all commands have been run (the same as if 'setup()'\n        had been called in the usual way\n\n    Returns the Distribution instance, which provides all information\n    used to drive the Distutils.\n    ")
    
    
    # Getting the type of 'stop_after' (line 202)
    stop_after_306612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 7), 'stop_after')
    
    # Obtaining an instance of the builtin type 'tuple' (line 202)
    tuple_306613 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 26), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 202)
    # Adding element type (line 202)
    str_306614 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 26), 'str', 'init')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), tuple_306613, str_306614)
    # Adding element type (line 202)
    str_306615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 34), 'str', 'config')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), tuple_306613, str_306615)
    # Adding element type (line 202)
    str_306616 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 44), 'str', 'commandline')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), tuple_306613, str_306616)
    # Adding element type (line 202)
    str_306617 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 59), 'str', 'run')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 26), tuple_306613, str_306617)
    
    # Applying the binary operator 'notin' (line 202)
    result_contains_306618 = python_operator(stypy.reporting.localization.Localization(__file__, 202, 7), 'notin', stop_after_306612, tuple_306613)
    
    # Testing the type of an if condition (line 202)
    if_condition_306619 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 202, 4), result_contains_306618)
    # Assigning a type to the variable 'if_condition_306619' (line 202)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 202, 4), 'if_condition_306619', if_condition_306619)
    # SSA begins for if statement (line 202)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ValueError' (line 203)
    ValueError_306620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 14), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 203, 8), ValueError_306620, 'raise parameter', BaseException)
    # SSA join for if statement (line 202)
    module_type_store = module_type_store.join_ssa_context()
    
    # Marking variables as global (line 205)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 205, 4), '_setup_stop_after')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 205, 4), '_setup_distribution')
    
    # Assigning a Name to a Name (line 206):
    # Getting the type of 'stop_after' (line 206)
    stop_after_306621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 24), 'stop_after')
    # Assigning a type to the variable '_setup_stop_after' (line 206)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 4), '_setup_stop_after', stop_after_306621)
    
    # Assigning a Attribute to a Name (line 208):
    # Getting the type of 'sys' (line 208)
    sys_306622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 16), 'sys')
    # Obtaining the member 'argv' of a type (line 208)
    argv_306623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 16), sys_306622, 'argv')
    # Assigning a type to the variable 'save_argv' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 4), 'save_argv', argv_306623)
    
    # Assigning a Dict to a Name (line 209):
    
    # Obtaining an instance of the builtin type 'dict' (line 209)
    dict_306624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 209)
    # Adding element type (key, value) (line 209)
    str_306625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 9), 'str', '__file__')
    # Getting the type of 'script_name' (line 209)
    script_name_306626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'script_name')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 209, 8), dict_306624, (str_306625, script_name_306626))
    
    # Assigning a type to the variable 'g' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 4), 'g', dict_306624)
    
    # Assigning a Dict to a Name (line 210):
    
    # Obtaining an instance of the builtin type 'dict' (line 210)
    dict_306627 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, 8), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 210)
    
    # Assigning a type to the variable 'l' (line 210)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 210, 4), 'l', dict_306627)
    
    
    # SSA begins for try-except statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Try-finally block (line 212)
    
    # Assigning a Name to a Subscript (line 213):
    # Getting the type of 'script_name' (line 213)
    script_name_306628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 26), 'script_name')
    # Getting the type of 'sys' (line 213)
    sys_306629 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 12), 'sys')
    # Obtaining the member 'argv' of a type (line 213)
    argv_306630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 12), sys_306629, 'argv')
    int_306631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 21), 'int')
    # Storing an element on a container (line 213)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 213, 12), argv_306630, (int_306631, script_name_306628))
    
    # Type idiom detected: calculating its left and rigth part (line 214)
    # Getting the type of 'script_args' (line 214)
    script_args_306632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 12), 'script_args')
    # Getting the type of 'None' (line 214)
    None_306633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 34), 'None')
    
    (may_be_306634, more_types_in_union_306635) = may_not_be_none(script_args_306632, None_306633)

    if may_be_306634:

        if more_types_in_union_306635:
            # Runtime conditional SSA (line 214)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Name to a Subscript (line 215):
        # Getting the type of 'script_args' (line 215)
        script_args_306636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 31), 'script_args')
        # Getting the type of 'sys' (line 215)
        sys_306637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 16), 'sys')
        # Obtaining the member 'argv' of a type (line 215)
        argv_306638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 16), sys_306637, 'argv')
        int_306639 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 25), 'int')
        slice_306640 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 215, 16), int_306639, None, None)
        # Storing an element on a container (line 215)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 215, 16), argv_306638, (slice_306640, script_args_306636))

        if more_types_in_union_306635:
            # SSA join for if statement (line 214)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 216):
    
    # Call to open(...): (line 216)
    # Processing the call arguments (line 216)
    # Getting the type of 'script_name' (line 216)
    script_name_306642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'script_name', False)
    # Processing the call keyword arguments (line 216)
    kwargs_306643 = {}
    # Getting the type of 'open' (line 216)
    open_306641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 16), 'open', False)
    # Calling open(args, kwargs) (line 216)
    open_call_result_306644 = invoke(stypy.reporting.localization.Localization(__file__, 216, 16), open_306641, *[script_name_306642], **kwargs_306643)
    
    # Assigning a type to the variable 'f' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'f', open_call_result_306644)
    
    # Try-finally block (line 217)
    # Dynamic code evaluation using an exec statement
    
    # Call to read(...): (line 218)
    # Processing the call keyword arguments (line 218)
    kwargs_306647 = {}
    # Getting the type of 'f' (line 218)
    f_306645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 21), 'f', False)
    # Obtaining the member 'read' of a type (line 218)
    read_306646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 21), f_306645, 'read')
    # Calling read(args, kwargs) (line 218)
    read_call_result_306648 = invoke(stypy.reporting.localization.Localization(__file__, 218, 21), read_306646, *[], **kwargs_306647)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 218, 16), read_call_result_306648, 'exec parameter', 'StringType', 'FileType', 'CodeType')
    enable_usage_of_dynamic_types_warning(stypy.reporting.localization.Localization(__file__, 218, 16))
    
    # finally branch of the try-finally block (line 217)
    
    # Call to close(...): (line 220)
    # Processing the call keyword arguments (line 220)
    kwargs_306651 = {}
    # Getting the type of 'f' (line 220)
    f_306649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'f', False)
    # Obtaining the member 'close' of a type (line 220)
    close_306650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), f_306649, 'close')
    # Calling close(args, kwargs) (line 220)
    close_call_result_306652 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), close_306650, *[], **kwargs_306651)
    
    
    
    # finally branch of the try-finally block (line 212)
    
    # Assigning a Name to a Attribute (line 222):
    # Getting the type of 'save_argv' (line 222)
    save_argv_306653 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 23), 'save_argv')
    # Getting the type of 'sys' (line 222)
    sys_306654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 12), 'sys')
    # Setting the type of the member 'argv' of a type (line 222)
    module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 12), sys_306654, 'argv', save_argv_306653)
    
    # Assigning a Name to a Name (line 223):
    # Getting the type of 'None' (line 223)
    None_306655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 32), 'None')
    # Assigning a type to the variable '_setup_stop_after' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 12), '_setup_stop_after', None_306655)
    
    # SSA branch for the except part of a try statement (line 211)
    # SSA branch for the except 'SystemExit' branch of a try statement (line 211)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA branch for the except '<any exception>' branch of a try statement (line 211)
    module_type_store.open_ssa_branch('except')
    # SSA join for try-except statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 231)
    # Getting the type of '_setup_distribution' (line 231)
    _setup_distribution_306656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 7), '_setup_distribution')
    # Getting the type of 'None' (line 231)
    None_306657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 30), 'None')
    
    (may_be_306658, more_types_in_union_306659) = may_be_none(_setup_distribution_306656, None_306657)

    if may_be_306658:

        if more_types_in_union_306659:
            # Runtime conditional SSA (line 231)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Getting the type of 'RuntimeError' (line 232)
        RuntimeError_306660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 14), 'RuntimeError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 232, 8), RuntimeError_306660, 'raise parameter', BaseException)

        if more_types_in_union_306659:
            # SSA join for if statement (line 231)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of '_setup_distribution' (line 239)
    _setup_distribution_306661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 11), '_setup_distribution')
    # Assigning a type to the variable 'stypy_return_type' (line 239)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 239, 4), 'stypy_return_type', _setup_distribution_306661)
    
    # ################# End of 'run_setup(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'run_setup' in the type store
    # Getting the type of 'stypy_return_type' (line 171)
    stypy_return_type_306662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_306662)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'run_setup'
    return stypy_return_type_306662

# Assigning a type to the variable 'run_setup' (line 171)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 0), 'run_setup', run_setup)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
