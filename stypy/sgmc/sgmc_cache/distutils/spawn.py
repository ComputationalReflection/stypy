
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.spawn
2: 
3: Provides the 'spawn()' function, a front-end to various platform-
4: specific functions for launching another program in a sub-process.
5: Also provides the 'find_executable()' to search the path for a given
6: executable name.
7: '''
8: 
9: __revision__ = "$Id$"
10: 
11: import sys
12: import os
13: 
14: from distutils.errors import DistutilsPlatformError, DistutilsExecError
15: from distutils.debug import DEBUG
16: from distutils import log
17: 
18: def spawn(cmd, search_path=1, verbose=0, dry_run=0):
19:     '''Run another program, specified as a command list 'cmd', in a new process.
20: 
21:     'cmd' is just the argument list for the new process, ie.
22:     cmd[0] is the program to run and cmd[1:] are the rest of its arguments.
23:     There is no way to run a program with a name different from that of its
24:     executable.
25: 
26:     If 'search_path' is true (the default), the system's executable
27:     search path will be used to find the program; otherwise, cmd[0]
28:     must be the exact path to the executable.  If 'dry_run' is true,
29:     the command will not actually be run.
30: 
31:     Raise DistutilsExecError if running the program fails in any way; just
32:     return on success.
33:     '''
34:     # cmd is documented as a list, but just in case some code passes a tuple
35:     # in, protect our %-formatting code against horrible death
36:     cmd = list(cmd)
37:     if os.name == 'posix':
38:         _spawn_posix(cmd, search_path, dry_run=dry_run)
39:     elif os.name == 'nt':
40:         _spawn_nt(cmd, search_path, dry_run=dry_run)
41:     elif os.name == 'os2':
42:         _spawn_os2(cmd, search_path, dry_run=dry_run)
43:     else:
44:         raise DistutilsPlatformError, \
45:               "don't know how to spawn programs on platform '%s'" % os.name
46: 
47: def _nt_quote_args(args):
48:     '''Quote command-line arguments for DOS/Windows conventions.
49: 
50:     Just wraps every argument which contains blanks in double quotes, and
51:     returns a new argument list.
52:     '''
53:     # XXX this doesn't seem very robust to me -- but if the Windows guys
54:     # say it'll work, I guess I'll have to accept it.  (What if an arg
55:     # contains quotes?  What other magic characters, other than spaces,
56:     # have to be escaped?  Is there an escaping mechanism other than
57:     # quoting?)
58:     for i, arg in enumerate(args):
59:         if ' ' in arg:
60:             args[i] = '"%s"' % arg
61:     return args
62: 
63: def _spawn_nt(cmd, search_path=1, verbose=0, dry_run=0):
64:     executable = cmd[0]
65:     cmd = _nt_quote_args(cmd)
66:     if search_path:
67:         # either we find one or it stays the same
68:         executable = find_executable(executable) or executable
69:     log.info(' '.join([executable] + cmd[1:]))
70:     if not dry_run:
71:         # spawn for NT requires a full path to the .exe
72:         try:
73:             rc = os.spawnv(os.P_WAIT, executable, cmd)
74:         except OSError, exc:
75:             # this seems to happen when the command isn't found
76:             if not DEBUG:
77:                 cmd = executable
78:             raise DistutilsExecError, \
79:                   "command %r failed: %s" % (cmd, exc[-1])
80:         if rc != 0:
81:             # and this reflects the command running but failing
82:             if not DEBUG:
83:                 cmd = executable
84:             raise DistutilsExecError, \
85:                   "command %r failed with exit status %d" % (cmd, rc)
86: 
87: def _spawn_os2(cmd, search_path=1, verbose=0, dry_run=0):
88:     executable = cmd[0]
89:     if search_path:
90:         # either we find one or it stays the same
91:         executable = find_executable(executable) or executable
92:     log.info(' '.join([executable] + cmd[1:]))
93:     if not dry_run:
94:         # spawnv for OS/2 EMX requires a full path to the .exe
95:         try:
96:             rc = os.spawnv(os.P_WAIT, executable, cmd)
97:         except OSError, exc:
98:             # this seems to happen when the command isn't found
99:             if not DEBUG:
100:                 cmd = executable
101:             raise DistutilsExecError, \
102:                   "command %r failed: %s" % (cmd, exc[-1])
103:         if rc != 0:
104:             # and this reflects the command running but failing
105:             if not DEBUG:
106:                 cmd = executable
107:             log.debug("command %r failed with exit status %d" % (cmd, rc))
108:             raise DistutilsExecError, \
109:                   "command %r failed with exit status %d" % (cmd, rc)
110: 
111: if sys.platform == 'darwin':
112:     from distutils import sysconfig
113:     _cfg_target = None
114:     _cfg_target_split = None
115: 
116: def _spawn_posix(cmd, search_path=1, verbose=0, dry_run=0):
117:     log.info(' '.join(cmd))
118:     if dry_run:
119:         return
120:     executable = cmd[0]
121:     exec_fn = search_path and os.execvp or os.execv
122:     env = None
123:     if sys.platform == 'darwin':
124:         global _cfg_target, _cfg_target_split
125:         if _cfg_target is None:
126:             _cfg_target = sysconfig.get_config_var(
127:                                   'MACOSX_DEPLOYMENT_TARGET') or ''
128:             if _cfg_target:
129:                 _cfg_target_split = [int(x) for x in _cfg_target.split('.')]
130:         if _cfg_target:
131:             # ensure that the deployment target of build process is not less
132:             # than that used when the interpreter was built. This ensures
133:             # extension modules are built with correct compatibility values
134:             cur_target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', _cfg_target)
135:             if _cfg_target_split > [int(x) for x in cur_target.split('.')]:
136:                 my_msg = ('$MACOSX_DEPLOYMENT_TARGET mismatch: '
137:                           'now "%s" but "%s" during configure'
138:                                 % (cur_target, _cfg_target))
139:                 raise DistutilsPlatformError(my_msg)
140:             env = dict(os.environ,
141:                        MACOSX_DEPLOYMENT_TARGET=cur_target)
142:             exec_fn = search_path and os.execvpe or os.execve
143:     pid = os.fork()
144: 
145:     if pid == 0:  # in the child
146:         try:
147:             if env is None:
148:                 exec_fn(executable, cmd)
149:             else:
150:                 exec_fn(executable, cmd, env)
151:         except OSError, e:
152:             if not DEBUG:
153:                 cmd = executable
154:             sys.stderr.write("unable to execute %r: %s\n" %
155:                              (cmd, e.strerror))
156:             os._exit(1)
157: 
158:         if not DEBUG:
159:             cmd = executable
160:         sys.stderr.write("unable to execute %r for unknown reasons" % cmd)
161:         os._exit(1)
162:     else:   # in the parent
163:         # Loop until the child either exits or is terminated by a signal
164:         # (ie. keep waiting if it's merely stopped)
165:         while 1:
166:             try:
167:                 pid, status = os.waitpid(pid, 0)
168:             except OSError, exc:
169:                 import errno
170:                 if exc.errno == errno.EINTR:
171:                     continue
172:                 if not DEBUG:
173:                     cmd = executable
174:                 raise DistutilsExecError, \
175:                       "command %r failed: %s" % (cmd, exc[-1])
176:             if os.WIFSIGNALED(status):
177:                 if not DEBUG:
178:                     cmd = executable
179:                 raise DistutilsExecError, \
180:                       "command %r terminated by signal %d" % \
181:                       (cmd, os.WTERMSIG(status))
182: 
183:             elif os.WIFEXITED(status):
184:                 exit_status = os.WEXITSTATUS(status)
185:                 if exit_status == 0:
186:                     return   # hey, it succeeded!
187:                 else:
188:                     if not DEBUG:
189:                         cmd = executable
190:                     raise DistutilsExecError, \
191:                           "command %r failed with exit status %d" % \
192:                           (cmd, exit_status)
193: 
194:             elif os.WIFSTOPPED(status):
195:                 continue
196: 
197:             else:
198:                 if not DEBUG:
199:                     cmd = executable
200:                 raise DistutilsExecError, \
201:                       "unknown error executing %r: termination status %d" % \
202:                       (cmd, status)
203: 
204: def find_executable(executable, path=None):
205:     '''Tries to find 'executable' in the directories listed in 'path'.
206: 
207:     A string listing directories separated by 'os.pathsep'; defaults to
208:     os.environ['PATH'].  Returns the complete filename or None if not found.
209:     '''
210:     if path is None:
211:         path = os.environ['PATH']
212:     paths = path.split(os.pathsep)
213:     base, ext = os.path.splitext(executable)
214: 
215:     if (sys.platform == 'win32' or os.name == 'os2') and (ext != '.exe'):
216:         executable = executable + '.exe'
217: 
218:     if not os.path.isfile(executable):
219:         for p in paths:
220:             f = os.path.join(p, executable)
221:             if os.path.isfile(f):
222:                 # the file exists, we have a shot at spawn working
223:                 return f
224:         return None
225:     else:
226:         return executable
227: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_6715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, (-1)), 'str', "distutils.spawn\n\nProvides the 'spawn()' function, a front-end to various platform-\nspecific functions for launching another program in a sub-process.\nAlso provides the 'find_executable()' to search the path for a given\nexecutable name.\n")

# Assigning a Str to a Name (line 9):

# Assigning a Str to a Name (line 9):
str_6716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), '__revision__', str_6716)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'import sys' statement (line 11)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'import os' statement (line 12)
import os

import_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.errors import DistutilsPlatformError, DistutilsExecError' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_6717 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors')

if (type(import_6717) is not StypyTypeError):

    if (import_6717 != 'pyd_module'):
        __import__(import_6717)
        sys_modules_6718 = sys.modules[import_6717]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', sys_modules_6718.module_type_store, module_type_store, ['DistutilsPlatformError', 'DistutilsExecError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_6718, sys_modules_6718.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError, DistutilsExecError

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError', 'DistutilsExecError'], [DistutilsPlatformError, DistutilsExecError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.errors', import_6717)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils.debug import DEBUG' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_6719 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.debug')

if (type(import_6719) is not StypyTypeError):

    if (import_6719 != 'pyd_module'):
        __import__(import_6719)
        sys_modules_6720 = sys.modules[import_6719]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.debug', sys_modules_6720.module_type_store, module_type_store, ['DEBUG'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 0), __file__, sys_modules_6720, sys_modules_6720.module_type_store, module_type_store)
    else:
        from distutils.debug import DEBUG

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.debug', None, module_type_store, ['DEBUG'], [DEBUG])

else:
    # Assigning a type to the variable 'distutils.debug' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils.debug', import_6719)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 0))

# 'from distutils import log' statement (line 16)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 16, 0), 'distutils', None, module_type_store, ['log'], [log])


@norecursion
def spawn(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_6721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 27), 'int')
    int_6722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 38), 'int')
    int_6723 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 49), 'int')
    defaults = [int_6721, int_6722, int_6723]
    # Create a new context for function 'spawn'
    module_type_store = module_type_store.open_function_context('spawn', 18, 0, False)
    
    # Passed parameters checking function
    spawn.stypy_localization = localization
    spawn.stypy_type_of_self = None
    spawn.stypy_type_store = module_type_store
    spawn.stypy_function_name = 'spawn'
    spawn.stypy_param_names_list = ['cmd', 'search_path', 'verbose', 'dry_run']
    spawn.stypy_varargs_param_name = None
    spawn.stypy_kwargs_param_name = None
    spawn.stypy_call_defaults = defaults
    spawn.stypy_call_varargs = varargs
    spawn.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'spawn', ['cmd', 'search_path', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'spawn', localization, ['cmd', 'search_path', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'spawn(...)' code ##################

    str_6724 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', "Run another program, specified as a command list 'cmd', in a new process.\n\n    'cmd' is just the argument list for the new process, ie.\n    cmd[0] is the program to run and cmd[1:] are the rest of its arguments.\n    There is no way to run a program with a name different from that of its\n    executable.\n\n    If 'search_path' is true (the default), the system's executable\n    search path will be used to find the program; otherwise, cmd[0]\n    must be the exact path to the executable.  If 'dry_run' is true,\n    the command will not actually be run.\n\n    Raise DistutilsExecError if running the program fails in any way; just\n    return on success.\n    ")
    
    # Assigning a Call to a Name (line 36):
    
    # Assigning a Call to a Name (line 36):
    
    # Call to list(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'cmd' (line 36)
    cmd_6726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'cmd', False)
    # Processing the call keyword arguments (line 36)
    kwargs_6727 = {}
    # Getting the type of 'list' (line 36)
    list_6725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 10), 'list', False)
    # Calling list(args, kwargs) (line 36)
    list_call_result_6728 = invoke(stypy.reporting.localization.Localization(__file__, 36, 10), list_6725, *[cmd_6726], **kwargs_6727)
    
    # Assigning a type to the variable 'cmd' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'cmd', list_call_result_6728)
    
    
    # Getting the type of 'os' (line 37)
    os_6729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 7), 'os')
    # Obtaining the member 'name' of a type (line 37)
    name_6730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 7), os_6729, 'name')
    str_6731 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 18), 'str', 'posix')
    # Applying the binary operator '==' (line 37)
    result_eq_6732 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 7), '==', name_6730, str_6731)
    
    # Testing the type of an if condition (line 37)
    if_condition_6733 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 4), result_eq_6732)
    # Assigning a type to the variable 'if_condition_6733' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'if_condition_6733', if_condition_6733)
    # SSA begins for if statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _spawn_posix(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'cmd' (line 38)
    cmd_6735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 21), 'cmd', False)
    # Getting the type of 'search_path' (line 38)
    search_path_6736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'search_path', False)
    # Processing the call keyword arguments (line 38)
    # Getting the type of 'dry_run' (line 38)
    dry_run_6737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 47), 'dry_run', False)
    keyword_6738 = dry_run_6737
    kwargs_6739 = {'dry_run': keyword_6738}
    # Getting the type of '_spawn_posix' (line 38)
    _spawn_posix_6734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), '_spawn_posix', False)
    # Calling _spawn_posix(args, kwargs) (line 38)
    _spawn_posix_call_result_6740 = invoke(stypy.reporting.localization.Localization(__file__, 38, 8), _spawn_posix_6734, *[cmd_6735, search_path_6736], **kwargs_6739)
    
    # SSA branch for the else part of an if statement (line 37)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 39)
    os_6741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'os')
    # Obtaining the member 'name' of a type (line 39)
    name_6742 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 9), os_6741, 'name')
    str_6743 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 20), 'str', 'nt')
    # Applying the binary operator '==' (line 39)
    result_eq_6744 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 9), '==', name_6742, str_6743)
    
    # Testing the type of an if condition (line 39)
    if_condition_6745 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 39, 9), result_eq_6744)
    # Assigning a type to the variable 'if_condition_6745' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 9), 'if_condition_6745', if_condition_6745)
    # SSA begins for if statement (line 39)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _spawn_nt(...): (line 40)
    # Processing the call arguments (line 40)
    # Getting the type of 'cmd' (line 40)
    cmd_6747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 18), 'cmd', False)
    # Getting the type of 'search_path' (line 40)
    search_path_6748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 23), 'search_path', False)
    # Processing the call keyword arguments (line 40)
    # Getting the type of 'dry_run' (line 40)
    dry_run_6749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 44), 'dry_run', False)
    keyword_6750 = dry_run_6749
    kwargs_6751 = {'dry_run': keyword_6750}
    # Getting the type of '_spawn_nt' (line 40)
    _spawn_nt_6746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), '_spawn_nt', False)
    # Calling _spawn_nt(args, kwargs) (line 40)
    _spawn_nt_call_result_6752 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), _spawn_nt_6746, *[cmd_6747, search_path_6748], **kwargs_6751)
    
    # SSA branch for the else part of an if statement (line 39)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'os' (line 41)
    os_6753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 9), 'os')
    # Obtaining the member 'name' of a type (line 41)
    name_6754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 9), os_6753, 'name')
    str_6755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 20), 'str', 'os2')
    # Applying the binary operator '==' (line 41)
    result_eq_6756 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 9), '==', name_6754, str_6755)
    
    # Testing the type of an if condition (line 41)
    if_condition_6757 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 9), result_eq_6756)
    # Assigning a type to the variable 'if_condition_6757' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 9), 'if_condition_6757', if_condition_6757)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _spawn_os2(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'cmd' (line 42)
    cmd_6759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 19), 'cmd', False)
    # Getting the type of 'search_path' (line 42)
    search_path_6760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 24), 'search_path', False)
    # Processing the call keyword arguments (line 42)
    # Getting the type of 'dry_run' (line 42)
    dry_run_6761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 45), 'dry_run', False)
    keyword_6762 = dry_run_6761
    kwargs_6763 = {'dry_run': keyword_6762}
    # Getting the type of '_spawn_os2' (line 42)
    _spawn_os2_6758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), '_spawn_os2', False)
    # Calling _spawn_os2(args, kwargs) (line 42)
    _spawn_os2_call_result_6764 = invoke(stypy.reporting.localization.Localization(__file__, 42, 8), _spawn_os2_6758, *[cmd_6759, search_path_6760], **kwargs_6763)
    
    # SSA branch for the else part of an if statement (line 41)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'DistutilsPlatformError' (line 44)
    DistutilsPlatformError_6765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 14), 'DistutilsPlatformError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 44, 8), DistutilsPlatformError_6765, 'raise parameter', BaseException)
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 39)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'spawn(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'spawn' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_6766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6766)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'spawn'
    return stypy_return_type_6766

# Assigning a type to the variable 'spawn' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'spawn', spawn)

@norecursion
def _nt_quote_args(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_nt_quote_args'
    module_type_store = module_type_store.open_function_context('_nt_quote_args', 47, 0, False)
    
    # Passed parameters checking function
    _nt_quote_args.stypy_localization = localization
    _nt_quote_args.stypy_type_of_self = None
    _nt_quote_args.stypy_type_store = module_type_store
    _nt_quote_args.stypy_function_name = '_nt_quote_args'
    _nt_quote_args.stypy_param_names_list = ['args']
    _nt_quote_args.stypy_varargs_param_name = None
    _nt_quote_args.stypy_kwargs_param_name = None
    _nt_quote_args.stypy_call_defaults = defaults
    _nt_quote_args.stypy_call_varargs = varargs
    _nt_quote_args.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_nt_quote_args', ['args'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_nt_quote_args', localization, ['args'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_nt_quote_args(...)' code ##################

    str_6767 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, (-1)), 'str', 'Quote command-line arguments for DOS/Windows conventions.\n\n    Just wraps every argument which contains blanks in double quotes, and\n    returns a new argument list.\n    ')
    
    
    # Call to enumerate(...): (line 58)
    # Processing the call arguments (line 58)
    # Getting the type of 'args' (line 58)
    args_6769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 28), 'args', False)
    # Processing the call keyword arguments (line 58)
    kwargs_6770 = {}
    # Getting the type of 'enumerate' (line 58)
    enumerate_6768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 18), 'enumerate', False)
    # Calling enumerate(args, kwargs) (line 58)
    enumerate_call_result_6771 = invoke(stypy.reporting.localization.Localization(__file__, 58, 18), enumerate_6768, *[args_6769], **kwargs_6770)
    
    # Testing the type of a for loop iterable (line 58)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 58, 4), enumerate_call_result_6771)
    # Getting the type of the for loop variable (line 58)
    for_loop_var_6772 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 58, 4), enumerate_call_result_6771)
    # Assigning a type to the variable 'i' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'i', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 4), for_loop_var_6772))
    # Assigning a type to the variable 'arg' (line 58)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 58, 4), for_loop_var_6772))
    # SSA begins for a for statement (line 58)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    str_6773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 11), 'str', ' ')
    # Getting the type of 'arg' (line 59)
    arg_6774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 18), 'arg')
    # Applying the binary operator 'in' (line 59)
    result_contains_6775 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), 'in', str_6773, arg_6774)
    
    # Testing the type of an if condition (line 59)
    if_condition_6776 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_contains_6775)
    # Assigning a type to the variable 'if_condition_6776' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_6776', if_condition_6776)
    # SSA begins for if statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Subscript (line 60):
    
    # Assigning a BinOp to a Subscript (line 60):
    str_6777 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 22), 'str', '"%s"')
    # Getting the type of 'arg' (line 60)
    arg_6778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 31), 'arg')
    # Applying the binary operator '%' (line 60)
    result_mod_6779 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 22), '%', str_6777, arg_6778)
    
    # Getting the type of 'args' (line 60)
    args_6780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'args')
    # Getting the type of 'i' (line 60)
    i_6781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 17), 'i')
    # Storing an element on a container (line 60)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 60, 12), args_6780, (i_6781, result_mod_6779))
    # SSA join for if statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'args' (line 61)
    args_6782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 11), 'args')
    # Assigning a type to the variable 'stypy_return_type' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type', args_6782)
    
    # ################# End of '_nt_quote_args(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_nt_quote_args' in the type store
    # Getting the type of 'stypy_return_type' (line 47)
    stypy_return_type_6783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6783)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_nt_quote_args'
    return stypy_return_type_6783

# Assigning a type to the variable '_nt_quote_args' (line 47)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), '_nt_quote_args', _nt_quote_args)

@norecursion
def _spawn_nt(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_6784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'int')
    int_6785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 42), 'int')
    int_6786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 53), 'int')
    defaults = [int_6784, int_6785, int_6786]
    # Create a new context for function '_spawn_nt'
    module_type_store = module_type_store.open_function_context('_spawn_nt', 63, 0, False)
    
    # Passed parameters checking function
    _spawn_nt.stypy_localization = localization
    _spawn_nt.stypy_type_of_self = None
    _spawn_nt.stypy_type_store = module_type_store
    _spawn_nt.stypy_function_name = '_spawn_nt'
    _spawn_nt.stypy_param_names_list = ['cmd', 'search_path', 'verbose', 'dry_run']
    _spawn_nt.stypy_varargs_param_name = None
    _spawn_nt.stypy_kwargs_param_name = None
    _spawn_nt.stypy_call_defaults = defaults
    _spawn_nt.stypy_call_varargs = varargs
    _spawn_nt.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_spawn_nt', ['cmd', 'search_path', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_spawn_nt', localization, ['cmd', 'search_path', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_spawn_nt(...)' code ##################

    
    # Assigning a Subscript to a Name (line 64):
    
    # Assigning a Subscript to a Name (line 64):
    
    # Obtaining the type of the subscript
    int_6787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 21), 'int')
    # Getting the type of 'cmd' (line 64)
    cmd_6788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 17), 'cmd')
    # Obtaining the member '__getitem__' of a type (line 64)
    getitem___6789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 17), cmd_6788, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 64)
    subscript_call_result_6790 = invoke(stypy.reporting.localization.Localization(__file__, 64, 17), getitem___6789, int_6787)
    
    # Assigning a type to the variable 'executable' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 4), 'executable', subscript_call_result_6790)
    
    # Assigning a Call to a Name (line 65):
    
    # Assigning a Call to a Name (line 65):
    
    # Call to _nt_quote_args(...): (line 65)
    # Processing the call arguments (line 65)
    # Getting the type of 'cmd' (line 65)
    cmd_6792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 25), 'cmd', False)
    # Processing the call keyword arguments (line 65)
    kwargs_6793 = {}
    # Getting the type of '_nt_quote_args' (line 65)
    _nt_quote_args_6791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 10), '_nt_quote_args', False)
    # Calling _nt_quote_args(args, kwargs) (line 65)
    _nt_quote_args_call_result_6794 = invoke(stypy.reporting.localization.Localization(__file__, 65, 10), _nt_quote_args_6791, *[cmd_6792], **kwargs_6793)
    
    # Assigning a type to the variable 'cmd' (line 65)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'cmd', _nt_quote_args_call_result_6794)
    
    # Getting the type of 'search_path' (line 66)
    search_path_6795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 7), 'search_path')
    # Testing the type of an if condition (line 66)
    if_condition_6796 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 4), search_path_6795)
    # Assigning a type to the variable 'if_condition_6796' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'if_condition_6796', if_condition_6796)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BoolOp to a Name (line 68):
    
    # Assigning a BoolOp to a Name (line 68):
    
    # Evaluating a boolean operation
    
    # Call to find_executable(...): (line 68)
    # Processing the call arguments (line 68)
    # Getting the type of 'executable' (line 68)
    executable_6798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 37), 'executable', False)
    # Processing the call keyword arguments (line 68)
    kwargs_6799 = {}
    # Getting the type of 'find_executable' (line 68)
    find_executable_6797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 21), 'find_executable', False)
    # Calling find_executable(args, kwargs) (line 68)
    find_executable_call_result_6800 = invoke(stypy.reporting.localization.Localization(__file__, 68, 21), find_executable_6797, *[executable_6798], **kwargs_6799)
    
    # Getting the type of 'executable' (line 68)
    executable_6801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 52), 'executable')
    # Applying the binary operator 'or' (line 68)
    result_or_keyword_6802 = python_operator(stypy.reporting.localization.Localization(__file__, 68, 21), 'or', find_executable_call_result_6800, executable_6801)
    
    # Assigning a type to the variable 'executable' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'executable', result_or_keyword_6802)
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to info(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Call to join(...): (line 69)
    # Processing the call arguments (line 69)
    
    # Obtaining an instance of the builtin type 'list' (line 69)
    list_6807 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 69)
    # Adding element type (line 69)
    # Getting the type of 'executable' (line 69)
    executable_6808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 23), 'executable', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 22), list_6807, executable_6808)
    
    
    # Obtaining the type of the subscript
    int_6809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 41), 'int')
    slice_6810 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 69, 37), int_6809, None, None)
    # Getting the type of 'cmd' (line 69)
    cmd_6811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 37), 'cmd', False)
    # Obtaining the member '__getitem__' of a type (line 69)
    getitem___6812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 37), cmd_6811, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 69)
    subscript_call_result_6813 = invoke(stypy.reporting.localization.Localization(__file__, 69, 37), getitem___6812, slice_6810)
    
    # Applying the binary operator '+' (line 69)
    result_add_6814 = python_operator(stypy.reporting.localization.Localization(__file__, 69, 22), '+', list_6807, subscript_call_result_6813)
    
    # Processing the call keyword arguments (line 69)
    kwargs_6815 = {}
    str_6805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 13), 'str', ' ')
    # Obtaining the member 'join' of a type (line 69)
    join_6806 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 13), str_6805, 'join')
    # Calling join(args, kwargs) (line 69)
    join_call_result_6816 = invoke(stypy.reporting.localization.Localization(__file__, 69, 13), join_6806, *[result_add_6814], **kwargs_6815)
    
    # Processing the call keyword arguments (line 69)
    kwargs_6817 = {}
    # Getting the type of 'log' (line 69)
    log_6803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 69)
    info_6804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 4), log_6803, 'info')
    # Calling info(args, kwargs) (line 69)
    info_call_result_6818 = invoke(stypy.reporting.localization.Localization(__file__, 69, 4), info_6804, *[join_call_result_6816], **kwargs_6817)
    
    
    
    # Getting the type of 'dry_run' (line 70)
    dry_run_6819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 'dry_run')
    # Applying the 'not' unary operator (line 70)
    result_not__6820 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 7), 'not', dry_run_6819)
    
    # Testing the type of an if condition (line 70)
    if_condition_6821 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 4), result_not__6820)
    # Assigning a type to the variable 'if_condition_6821' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'if_condition_6821', if_condition_6821)
    # SSA begins for if statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 73):
    
    # Assigning a Call to a Name (line 73):
    
    # Call to spawnv(...): (line 73)
    # Processing the call arguments (line 73)
    # Getting the type of 'os' (line 73)
    os_6824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 27), 'os', False)
    # Obtaining the member 'P_WAIT' of a type (line 73)
    P_WAIT_6825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 27), os_6824, 'P_WAIT')
    # Getting the type of 'executable' (line 73)
    executable_6826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 38), 'executable', False)
    # Getting the type of 'cmd' (line 73)
    cmd_6827 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 50), 'cmd', False)
    # Processing the call keyword arguments (line 73)
    kwargs_6828 = {}
    # Getting the type of 'os' (line 73)
    os_6822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 17), 'os', False)
    # Obtaining the member 'spawnv' of a type (line 73)
    spawnv_6823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 17), os_6822, 'spawnv')
    # Calling spawnv(args, kwargs) (line 73)
    spawnv_call_result_6829 = invoke(stypy.reporting.localization.Localization(__file__, 73, 17), spawnv_6823, *[P_WAIT_6825, executable_6826, cmd_6827], **kwargs_6828)
    
    # Assigning a type to the variable 'rc' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'rc', spawnv_call_result_6829)
    # SSA branch for the except part of a try statement (line 72)
    # SSA branch for the except 'OSError' branch of a try statement (line 72)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'OSError' (line 74)
    OSError_6830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'OSError')
    # Assigning a type to the variable 'exc' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 8), 'exc', OSError_6830)
    
    
    # Getting the type of 'DEBUG' (line 76)
    DEBUG_6831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'DEBUG')
    # Applying the 'not' unary operator (line 76)
    result_not__6832 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 15), 'not', DEBUG_6831)
    
    # Testing the type of an if condition (line 76)
    if_condition_6833 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 12), result_not__6832)
    # Assigning a type to the variable 'if_condition_6833' (line 76)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'if_condition_6833', if_condition_6833)
    # SSA begins for if statement (line 76)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 77):
    
    # Assigning a Name to a Name (line 77):
    # Getting the type of 'executable' (line 77)
    executable_6834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 22), 'executable')
    # Assigning a type to the variable 'cmd' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 16), 'cmd', executable_6834)
    # SSA join for if statement (line 76)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'DistutilsExecError' (line 78)
    DistutilsExecError_6835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 18), 'DistutilsExecError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 78, 12), DistutilsExecError_6835, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rc' (line 80)
    rc_6836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'rc')
    int_6837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 17), 'int')
    # Applying the binary operator '!=' (line 80)
    result_ne_6838 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 11), '!=', rc_6836, int_6837)
    
    # Testing the type of an if condition (line 80)
    if_condition_6839 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 80, 8), result_ne_6838)
    # Assigning a type to the variable 'if_condition_6839' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'if_condition_6839', if_condition_6839)
    # SSA begins for if statement (line 80)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'DEBUG' (line 82)
    DEBUG_6840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 19), 'DEBUG')
    # Applying the 'not' unary operator (line 82)
    result_not__6841 = python_operator(stypy.reporting.localization.Localization(__file__, 82, 15), 'not', DEBUG_6840)
    
    # Testing the type of an if condition (line 82)
    if_condition_6842 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 82, 12), result_not__6841)
    # Assigning a type to the variable 'if_condition_6842' (line 82)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 12), 'if_condition_6842', if_condition_6842)
    # SSA begins for if statement (line 82)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 83):
    
    # Assigning a Name to a Name (line 83):
    # Getting the type of 'executable' (line 83)
    executable_6843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 22), 'executable')
    # Assigning a type to the variable 'cmd' (line 83)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'cmd', executable_6843)
    # SSA join for if statement (line 82)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'DistutilsExecError' (line 84)
    DistutilsExecError_6844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 18), 'DistutilsExecError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 84, 12), DistutilsExecError_6844, 'raise parameter', BaseException)
    # SSA join for if statement (line 80)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_spawn_nt(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_spawn_nt' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_6845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6845)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_spawn_nt'
    return stypy_return_type_6845

# Assigning a type to the variable '_spawn_nt' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), '_spawn_nt', _spawn_nt)

@norecursion
def _spawn_os2(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_6846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 32), 'int')
    int_6847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 43), 'int')
    int_6848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 87, 54), 'int')
    defaults = [int_6846, int_6847, int_6848]
    # Create a new context for function '_spawn_os2'
    module_type_store = module_type_store.open_function_context('_spawn_os2', 87, 0, False)
    
    # Passed parameters checking function
    _spawn_os2.stypy_localization = localization
    _spawn_os2.stypy_type_of_self = None
    _spawn_os2.stypy_type_store = module_type_store
    _spawn_os2.stypy_function_name = '_spawn_os2'
    _spawn_os2.stypy_param_names_list = ['cmd', 'search_path', 'verbose', 'dry_run']
    _spawn_os2.stypy_varargs_param_name = None
    _spawn_os2.stypy_kwargs_param_name = None
    _spawn_os2.stypy_call_defaults = defaults
    _spawn_os2.stypy_call_varargs = varargs
    _spawn_os2.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_spawn_os2', ['cmd', 'search_path', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_spawn_os2', localization, ['cmd', 'search_path', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_spawn_os2(...)' code ##################

    
    # Assigning a Subscript to a Name (line 88):
    
    # Assigning a Subscript to a Name (line 88):
    
    # Obtaining the type of the subscript
    int_6849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 21), 'int')
    # Getting the type of 'cmd' (line 88)
    cmd_6850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 17), 'cmd')
    # Obtaining the member '__getitem__' of a type (line 88)
    getitem___6851 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 17), cmd_6850, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 88)
    subscript_call_result_6852 = invoke(stypy.reporting.localization.Localization(__file__, 88, 17), getitem___6851, int_6849)
    
    # Assigning a type to the variable 'executable' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'executable', subscript_call_result_6852)
    
    # Getting the type of 'search_path' (line 89)
    search_path_6853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 7), 'search_path')
    # Testing the type of an if condition (line 89)
    if_condition_6854 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 89, 4), search_path_6853)
    # Assigning a type to the variable 'if_condition_6854' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'if_condition_6854', if_condition_6854)
    # SSA begins for if statement (line 89)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BoolOp to a Name (line 91):
    
    # Assigning a BoolOp to a Name (line 91):
    
    # Evaluating a boolean operation
    
    # Call to find_executable(...): (line 91)
    # Processing the call arguments (line 91)
    # Getting the type of 'executable' (line 91)
    executable_6856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 37), 'executable', False)
    # Processing the call keyword arguments (line 91)
    kwargs_6857 = {}
    # Getting the type of 'find_executable' (line 91)
    find_executable_6855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'find_executable', False)
    # Calling find_executable(args, kwargs) (line 91)
    find_executable_call_result_6858 = invoke(stypy.reporting.localization.Localization(__file__, 91, 21), find_executable_6855, *[executable_6856], **kwargs_6857)
    
    # Getting the type of 'executable' (line 91)
    executable_6859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 52), 'executable')
    # Applying the binary operator 'or' (line 91)
    result_or_keyword_6860 = python_operator(stypy.reporting.localization.Localization(__file__, 91, 21), 'or', find_executable_call_result_6858, executable_6859)
    
    # Assigning a type to the variable 'executable' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'executable', result_or_keyword_6860)
    # SSA join for if statement (line 89)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to info(...): (line 92)
    # Processing the call arguments (line 92)
    
    # Call to join(...): (line 92)
    # Processing the call arguments (line 92)
    
    # Obtaining an instance of the builtin type 'list' (line 92)
    list_6865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 22), 'list')
    # Adding type elements to the builtin type 'list' instance (line 92)
    # Adding element type (line 92)
    # Getting the type of 'executable' (line 92)
    executable_6866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 23), 'executable', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 22), list_6865, executable_6866)
    
    
    # Obtaining the type of the subscript
    int_6867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 41), 'int')
    slice_6868 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 92, 37), int_6867, None, None)
    # Getting the type of 'cmd' (line 92)
    cmd_6869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 37), 'cmd', False)
    # Obtaining the member '__getitem__' of a type (line 92)
    getitem___6870 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 37), cmd_6869, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 92)
    subscript_call_result_6871 = invoke(stypy.reporting.localization.Localization(__file__, 92, 37), getitem___6870, slice_6868)
    
    # Applying the binary operator '+' (line 92)
    result_add_6872 = python_operator(stypy.reporting.localization.Localization(__file__, 92, 22), '+', list_6865, subscript_call_result_6871)
    
    # Processing the call keyword arguments (line 92)
    kwargs_6873 = {}
    str_6863 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 13), 'str', ' ')
    # Obtaining the member 'join' of a type (line 92)
    join_6864 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 13), str_6863, 'join')
    # Calling join(args, kwargs) (line 92)
    join_call_result_6874 = invoke(stypy.reporting.localization.Localization(__file__, 92, 13), join_6864, *[result_add_6872], **kwargs_6873)
    
    # Processing the call keyword arguments (line 92)
    kwargs_6875 = {}
    # Getting the type of 'log' (line 92)
    log_6861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 92)
    info_6862 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 4), log_6861, 'info')
    # Calling info(args, kwargs) (line 92)
    info_call_result_6876 = invoke(stypy.reporting.localization.Localization(__file__, 92, 4), info_6862, *[join_call_result_6874], **kwargs_6875)
    
    
    
    # Getting the type of 'dry_run' (line 93)
    dry_run_6877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 11), 'dry_run')
    # Applying the 'not' unary operator (line 93)
    result_not__6878 = python_operator(stypy.reporting.localization.Localization(__file__, 93, 7), 'not', dry_run_6877)
    
    # Testing the type of an if condition (line 93)
    if_condition_6879 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 93, 4), result_not__6878)
    # Assigning a type to the variable 'if_condition_6879' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'if_condition_6879', if_condition_6879)
    # SSA begins for if statement (line 93)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 95)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to spawnv(...): (line 96)
    # Processing the call arguments (line 96)
    # Getting the type of 'os' (line 96)
    os_6882 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 27), 'os', False)
    # Obtaining the member 'P_WAIT' of a type (line 96)
    P_WAIT_6883 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 27), os_6882, 'P_WAIT')
    # Getting the type of 'executable' (line 96)
    executable_6884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 38), 'executable', False)
    # Getting the type of 'cmd' (line 96)
    cmd_6885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 50), 'cmd', False)
    # Processing the call keyword arguments (line 96)
    kwargs_6886 = {}
    # Getting the type of 'os' (line 96)
    os_6880 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 17), 'os', False)
    # Obtaining the member 'spawnv' of a type (line 96)
    spawnv_6881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 17), os_6880, 'spawnv')
    # Calling spawnv(args, kwargs) (line 96)
    spawnv_call_result_6887 = invoke(stypy.reporting.localization.Localization(__file__, 96, 17), spawnv_6881, *[P_WAIT_6883, executable_6884, cmd_6885], **kwargs_6886)
    
    # Assigning a type to the variable 'rc' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'rc', spawnv_call_result_6887)
    # SSA branch for the except part of a try statement (line 95)
    # SSA branch for the except 'OSError' branch of a try statement (line 95)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'OSError' (line 97)
    OSError_6888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 15), 'OSError')
    # Assigning a type to the variable 'exc' (line 97)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'exc', OSError_6888)
    
    
    # Getting the type of 'DEBUG' (line 99)
    DEBUG_6889 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 19), 'DEBUG')
    # Applying the 'not' unary operator (line 99)
    result_not__6890 = python_operator(stypy.reporting.localization.Localization(__file__, 99, 15), 'not', DEBUG_6889)
    
    # Testing the type of an if condition (line 99)
    if_condition_6891 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 12), result_not__6890)
    # Assigning a type to the variable 'if_condition_6891' (line 99)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 12), 'if_condition_6891', if_condition_6891)
    # SSA begins for if statement (line 99)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 100):
    
    # Assigning a Name to a Name (line 100):
    # Getting the type of 'executable' (line 100)
    executable_6892 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 22), 'executable')
    # Assigning a type to the variable 'cmd' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'cmd', executable_6892)
    # SSA join for if statement (line 99)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'DistutilsExecError' (line 101)
    DistutilsExecError_6893 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 18), 'DistutilsExecError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 101, 12), DistutilsExecError_6893, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 95)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'rc' (line 103)
    rc_6894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'rc')
    int_6895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 17), 'int')
    # Applying the binary operator '!=' (line 103)
    result_ne_6896 = python_operator(stypy.reporting.localization.Localization(__file__, 103, 11), '!=', rc_6894, int_6895)
    
    # Testing the type of an if condition (line 103)
    if_condition_6897 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), result_ne_6896)
    # Assigning a type to the variable 'if_condition_6897' (line 103)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_6897', if_condition_6897)
    # SSA begins for if statement (line 103)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'DEBUG' (line 105)
    DEBUG_6898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 19), 'DEBUG')
    # Applying the 'not' unary operator (line 105)
    result_not__6899 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 15), 'not', DEBUG_6898)
    
    # Testing the type of an if condition (line 105)
    if_condition_6900 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 105, 12), result_not__6899)
    # Assigning a type to the variable 'if_condition_6900' (line 105)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'if_condition_6900', if_condition_6900)
    # SSA begins for if statement (line 105)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 106):
    
    # Assigning a Name to a Name (line 106):
    # Getting the type of 'executable' (line 106)
    executable_6901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 22), 'executable')
    # Assigning a type to the variable 'cmd' (line 106)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 16), 'cmd', executable_6901)
    # SSA join for if statement (line 105)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to debug(...): (line 107)
    # Processing the call arguments (line 107)
    str_6904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 22), 'str', 'command %r failed with exit status %d')
    
    # Obtaining an instance of the builtin type 'tuple' (line 107)
    tuple_6905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 65), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 107)
    # Adding element type (line 107)
    # Getting the type of 'cmd' (line 107)
    cmd_6906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 65), 'cmd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 65), tuple_6905, cmd_6906)
    # Adding element type (line 107)
    # Getting the type of 'rc' (line 107)
    rc_6907 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 70), 'rc', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 107, 65), tuple_6905, rc_6907)
    
    # Applying the binary operator '%' (line 107)
    result_mod_6908 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 22), '%', str_6904, tuple_6905)
    
    # Processing the call keyword arguments (line 107)
    kwargs_6909 = {}
    # Getting the type of 'log' (line 107)
    log_6902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'log', False)
    # Obtaining the member 'debug' of a type (line 107)
    debug_6903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), log_6902, 'debug')
    # Calling debug(args, kwargs) (line 107)
    debug_call_result_6910 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), debug_6903, *[result_mod_6908], **kwargs_6909)
    
    # Getting the type of 'DistutilsExecError' (line 108)
    DistutilsExecError_6911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 18), 'DistutilsExecError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 108, 12), DistutilsExecError_6911, 'raise parameter', BaseException)
    # SSA join for if statement (line 103)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 93)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_spawn_os2(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_spawn_os2' in the type store
    # Getting the type of 'stypy_return_type' (line 87)
    stypy_return_type_6912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_6912)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_spawn_os2'
    return stypy_return_type_6912

# Assigning a type to the variable '_spawn_os2' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), '_spawn_os2', _spawn_os2)


# Getting the type of 'sys' (line 111)
sys_6913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 3), 'sys')
# Obtaining the member 'platform' of a type (line 111)
platform_6914 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 3), sys_6913, 'platform')
str_6915 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 111, 19), 'str', 'darwin')
# Applying the binary operator '==' (line 111)
result_eq_6916 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 3), '==', platform_6914, str_6915)

# Testing the type of an if condition (line 111)
if_condition_6917 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 111, 0), result_eq_6916)
# Assigning a type to the variable 'if_condition_6917' (line 111)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 0), 'if_condition_6917', if_condition_6917)
# SSA begins for if statement (line 111)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 112, 4))

# 'from distutils import sysconfig' statement (line 112)
try:
    from distutils import sysconfig

except:
    sysconfig = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 112, 4), 'distutils', None, module_type_store, ['sysconfig'], [sysconfig])


# Assigning a Name to a Name (line 113):

# Assigning a Name to a Name (line 113):
# Getting the type of 'None' (line 113)
None_6918 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 18), 'None')
# Assigning a type to the variable '_cfg_target' (line 113)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), '_cfg_target', None_6918)

# Assigning a Name to a Name (line 114):

# Assigning a Name to a Name (line 114):
# Getting the type of 'None' (line 114)
None_6919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'None')
# Assigning a type to the variable '_cfg_target_split' (line 114)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 4), '_cfg_target_split', None_6919)
# SSA join for if statement (line 111)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _spawn_posix(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_6920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 34), 'int')
    int_6921 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 45), 'int')
    int_6922 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 56), 'int')
    defaults = [int_6920, int_6921, int_6922]
    # Create a new context for function '_spawn_posix'
    module_type_store = module_type_store.open_function_context('_spawn_posix', 116, 0, False)
    
    # Passed parameters checking function
    _spawn_posix.stypy_localization = localization
    _spawn_posix.stypy_type_of_self = None
    _spawn_posix.stypy_type_store = module_type_store
    _spawn_posix.stypy_function_name = '_spawn_posix'
    _spawn_posix.stypy_param_names_list = ['cmd', 'search_path', 'verbose', 'dry_run']
    _spawn_posix.stypy_varargs_param_name = None
    _spawn_posix.stypy_kwargs_param_name = None
    _spawn_posix.stypy_call_defaults = defaults
    _spawn_posix.stypy_call_varargs = varargs
    _spawn_posix.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_spawn_posix', ['cmd', 'search_path', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_spawn_posix', localization, ['cmd', 'search_path', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_spawn_posix(...)' code ##################

    
    # Call to info(...): (line 117)
    # Processing the call arguments (line 117)
    
    # Call to join(...): (line 117)
    # Processing the call arguments (line 117)
    # Getting the type of 'cmd' (line 117)
    cmd_6927 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 22), 'cmd', False)
    # Processing the call keyword arguments (line 117)
    kwargs_6928 = {}
    str_6925 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 13), 'str', ' ')
    # Obtaining the member 'join' of a type (line 117)
    join_6926 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 13), str_6925, 'join')
    # Calling join(args, kwargs) (line 117)
    join_call_result_6929 = invoke(stypy.reporting.localization.Localization(__file__, 117, 13), join_6926, *[cmd_6927], **kwargs_6928)
    
    # Processing the call keyword arguments (line 117)
    kwargs_6930 = {}
    # Getting the type of 'log' (line 117)
    log_6923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 117)
    info_6924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 4), log_6923, 'info')
    # Calling info(args, kwargs) (line 117)
    info_call_result_6931 = invoke(stypy.reporting.localization.Localization(__file__, 117, 4), info_6924, *[join_call_result_6929], **kwargs_6930)
    
    
    # Getting the type of 'dry_run' (line 118)
    dry_run_6932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'dry_run')
    # Testing the type of an if condition (line 118)
    if_condition_6933 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), dry_run_6932)
    # Assigning a type to the variable 'if_condition_6933' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_6933', if_condition_6933)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 120):
    
    # Assigning a Subscript to a Name (line 120):
    
    # Obtaining the type of the subscript
    int_6934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 21), 'int')
    # Getting the type of 'cmd' (line 120)
    cmd_6935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 17), 'cmd')
    # Obtaining the member '__getitem__' of a type (line 120)
    getitem___6936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 17), cmd_6935, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 120)
    subscript_call_result_6937 = invoke(stypy.reporting.localization.Localization(__file__, 120, 17), getitem___6936, int_6934)
    
    # Assigning a type to the variable 'executable' (line 120)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'executable', subscript_call_result_6937)
    
    # Assigning a BoolOp to a Name (line 121):
    
    # Assigning a BoolOp to a Name (line 121):
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    # Getting the type of 'search_path' (line 121)
    search_path_6938 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 14), 'search_path')
    # Getting the type of 'os' (line 121)
    os_6939 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 30), 'os')
    # Obtaining the member 'execvp' of a type (line 121)
    execvp_6940 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 30), os_6939, 'execvp')
    # Applying the binary operator 'and' (line 121)
    result_and_keyword_6941 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 14), 'and', search_path_6938, execvp_6940)
    
    # Getting the type of 'os' (line 121)
    os_6942 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 43), 'os')
    # Obtaining the member 'execv' of a type (line 121)
    execv_6943 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 43), os_6942, 'execv')
    # Applying the binary operator 'or' (line 121)
    result_or_keyword_6944 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 14), 'or', result_and_keyword_6941, execv_6943)
    
    # Assigning a type to the variable 'exec_fn' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'exec_fn', result_or_keyword_6944)
    
    # Assigning a Name to a Name (line 122):
    
    # Assigning a Name to a Name (line 122):
    # Getting the type of 'None' (line 122)
    None_6945 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 10), 'None')
    # Assigning a type to the variable 'env' (line 122)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'env', None_6945)
    
    
    # Getting the type of 'sys' (line 123)
    sys_6946 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 7), 'sys')
    # Obtaining the member 'platform' of a type (line 123)
    platform_6947 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 7), sys_6946, 'platform')
    str_6948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 23), 'str', 'darwin')
    # Applying the binary operator '==' (line 123)
    result_eq_6949 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 7), '==', platform_6947, str_6948)
    
    # Testing the type of an if condition (line 123)
    if_condition_6950 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 123, 4), result_eq_6949)
    # Assigning a type to the variable 'if_condition_6950' (line 123)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'if_condition_6950', if_condition_6950)
    # SSA begins for if statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Marking variables as global (line 124)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 124, 8), '_cfg_target')
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 124, 8), '_cfg_target_split')
    
    # Type idiom detected: calculating its left and rigth part (line 125)
    # Getting the type of '_cfg_target' (line 125)
    _cfg_target_6951 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 11), '_cfg_target')
    # Getting the type of 'None' (line 125)
    None_6952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 26), 'None')
    
    (may_be_6953, more_types_in_union_6954) = may_be_none(_cfg_target_6951, None_6952)

    if may_be_6953:

        if more_types_in_union_6954:
            # Runtime conditional SSA (line 125)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a BoolOp to a Name (line 126):
        
        # Assigning a BoolOp to a Name (line 126):
        
        # Evaluating a boolean operation
        
        # Call to get_config_var(...): (line 126)
        # Processing the call arguments (line 126)
        str_6957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 34), 'str', 'MACOSX_DEPLOYMENT_TARGET')
        # Processing the call keyword arguments (line 126)
        kwargs_6958 = {}
        # Getting the type of 'sysconfig' (line 126)
        sysconfig_6955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 26), 'sysconfig', False)
        # Obtaining the member 'get_config_var' of a type (line 126)
        get_config_var_6956 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 26), sysconfig_6955, 'get_config_var')
        # Calling get_config_var(args, kwargs) (line 126)
        get_config_var_call_result_6959 = invoke(stypy.reporting.localization.Localization(__file__, 126, 26), get_config_var_6956, *[str_6957], **kwargs_6958)
        
        str_6960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 127, 65), 'str', '')
        # Applying the binary operator 'or' (line 126)
        result_or_keyword_6961 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 26), 'or', get_config_var_call_result_6959, str_6960)
        
        # Assigning a type to the variable '_cfg_target' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 12), '_cfg_target', result_or_keyword_6961)
        
        # Getting the type of '_cfg_target' (line 128)
        _cfg_target_6962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), '_cfg_target')
        # Testing the type of an if condition (line 128)
        if_condition_6963 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 12), _cfg_target_6962)
        # Assigning a type to the variable 'if_condition_6963' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 12), 'if_condition_6963', if_condition_6963)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a ListComp to a Name (line 129):
        
        # Assigning a ListComp to a Name (line 129):
        # Calculating list comprehension
        # Calculating comprehension expression
        
        # Call to split(...): (line 129)
        # Processing the call arguments (line 129)
        str_6970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 71), 'str', '.')
        # Processing the call keyword arguments (line 129)
        kwargs_6971 = {}
        # Getting the type of '_cfg_target' (line 129)
        _cfg_target_6968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 53), '_cfg_target', False)
        # Obtaining the member 'split' of a type (line 129)
        split_6969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 53), _cfg_target_6968, 'split')
        # Calling split(args, kwargs) (line 129)
        split_call_result_6972 = invoke(stypy.reporting.localization.Localization(__file__, 129, 53), split_6969, *[str_6970], **kwargs_6971)
        
        comprehension_6973 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 37), split_call_result_6972)
        # Assigning a type to the variable 'x' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'x', comprehension_6973)
        
        # Call to int(...): (line 129)
        # Processing the call arguments (line 129)
        # Getting the type of 'x' (line 129)
        x_6965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 41), 'x', False)
        # Processing the call keyword arguments (line 129)
        kwargs_6966 = {}
        # Getting the type of 'int' (line 129)
        int_6964 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'int', False)
        # Calling int(args, kwargs) (line 129)
        int_call_result_6967 = invoke(stypy.reporting.localization.Localization(__file__, 129, 37), int_6964, *[x_6965], **kwargs_6966)
        
        list_6974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 37), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 37), list_6974, int_call_result_6967)
        # Assigning a type to the variable '_cfg_target_split' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), '_cfg_target_split', list_6974)
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_6954:
            # SSA join for if statement (line 125)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of '_cfg_target' (line 130)
    _cfg_target_6975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 11), '_cfg_target')
    # Testing the type of an if condition (line 130)
    if_condition_6976 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 130, 8), _cfg_target_6975)
    # Assigning a type to the variable 'if_condition_6976' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'if_condition_6976', if_condition_6976)
    # SSA begins for if statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 134):
    
    # Assigning a Call to a Name (line 134):
    
    # Call to get(...): (line 134)
    # Processing the call arguments (line 134)
    str_6980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 40), 'str', 'MACOSX_DEPLOYMENT_TARGET')
    # Getting the type of '_cfg_target' (line 134)
    _cfg_target_6981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 68), '_cfg_target', False)
    # Processing the call keyword arguments (line 134)
    kwargs_6982 = {}
    # Getting the type of 'os' (line 134)
    os_6977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 25), 'os', False)
    # Obtaining the member 'environ' of a type (line 134)
    environ_6978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 25), os_6977, 'environ')
    # Obtaining the member 'get' of a type (line 134)
    get_6979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 25), environ_6978, 'get')
    # Calling get(args, kwargs) (line 134)
    get_call_result_6983 = invoke(stypy.reporting.localization.Localization(__file__, 134, 25), get_6979, *[str_6980, _cfg_target_6981], **kwargs_6982)
    
    # Assigning a type to the variable 'cur_target' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'cur_target', get_call_result_6983)
    
    
    # Getting the type of '_cfg_target_split' (line 135)
    _cfg_target_split_6984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 15), '_cfg_target_split')
    # Calculating list comprehension
    # Calculating comprehension expression
    
    # Call to split(...): (line 135)
    # Processing the call arguments (line 135)
    str_6991 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 69), 'str', '.')
    # Processing the call keyword arguments (line 135)
    kwargs_6992 = {}
    # Getting the type of 'cur_target' (line 135)
    cur_target_6989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 52), 'cur_target', False)
    # Obtaining the member 'split' of a type (line 135)
    split_6990 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 52), cur_target_6989, 'split')
    # Calling split(args, kwargs) (line 135)
    split_call_result_6993 = invoke(stypy.reporting.localization.Localization(__file__, 135, 52), split_6990, *[str_6991], **kwargs_6992)
    
    comprehension_6994 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 36), split_call_result_6993)
    # Assigning a type to the variable 'x' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'x', comprehension_6994)
    
    # Call to int(...): (line 135)
    # Processing the call arguments (line 135)
    # Getting the type of 'x' (line 135)
    x_6986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 40), 'x', False)
    # Processing the call keyword arguments (line 135)
    kwargs_6987 = {}
    # Getting the type of 'int' (line 135)
    int_6985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 36), 'int', False)
    # Calling int(args, kwargs) (line 135)
    int_call_result_6988 = invoke(stypy.reporting.localization.Localization(__file__, 135, 36), int_6985, *[x_6986], **kwargs_6987)
    
    list_6995 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 36), 'list')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 36), list_6995, int_call_result_6988)
    # Applying the binary operator '>' (line 135)
    result_gt_6996 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 15), '>', _cfg_target_split_6984, list_6995)
    
    # Testing the type of an if condition (line 135)
    if_condition_6997 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 135, 12), result_gt_6996)
    # Assigning a type to the variable 'if_condition_6997' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 12), 'if_condition_6997', if_condition_6997)
    # SSA begins for if statement (line 135)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 136):
    
    # Assigning a BinOp to a Name (line 136):
    str_6998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 136, 26), 'str', '$MACOSX_DEPLOYMENT_TARGET mismatch: now "%s" but "%s" during configure')
    
    # Obtaining an instance of the builtin type 'tuple' (line 138)
    tuple_6999 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 35), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 138)
    # Adding element type (line 138)
    # Getting the type of 'cur_target' (line 138)
    cur_target_7000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 35), 'cur_target')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), tuple_6999, cur_target_7000)
    # Adding element type (line 138)
    # Getting the type of '_cfg_target' (line 138)
    _cfg_target_7001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 47), '_cfg_target')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 138, 35), tuple_6999, _cfg_target_7001)
    
    # Applying the binary operator '%' (line 136)
    result_mod_7002 = python_operator(stypy.reporting.localization.Localization(__file__, 136, 26), '%', str_6998, tuple_6999)
    
    # Assigning a type to the variable 'my_msg' (line 136)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 136, 16), 'my_msg', result_mod_7002)
    
    # Call to DistutilsPlatformError(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'my_msg' (line 139)
    my_msg_7004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 45), 'my_msg', False)
    # Processing the call keyword arguments (line 139)
    kwargs_7005 = {}
    # Getting the type of 'DistutilsPlatformError' (line 139)
    DistutilsPlatformError_7003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 22), 'DistutilsPlatformError', False)
    # Calling DistutilsPlatformError(args, kwargs) (line 139)
    DistutilsPlatformError_call_result_7006 = invoke(stypy.reporting.localization.Localization(__file__, 139, 22), DistutilsPlatformError_7003, *[my_msg_7004], **kwargs_7005)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 139, 16), DistutilsPlatformError_call_result_7006, 'raise parameter', BaseException)
    # SSA join for if statement (line 135)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 140):
    
    # Assigning a Call to a Name (line 140):
    
    # Call to dict(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'os' (line 140)
    os_7008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 23), 'os', False)
    # Obtaining the member 'environ' of a type (line 140)
    environ_7009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 23), os_7008, 'environ')
    # Processing the call keyword arguments (line 140)
    # Getting the type of 'cur_target' (line 141)
    cur_target_7010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 48), 'cur_target', False)
    keyword_7011 = cur_target_7010
    kwargs_7012 = {'MACOSX_DEPLOYMENT_TARGET': keyword_7011}
    # Getting the type of 'dict' (line 140)
    dict_7007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'dict', False)
    # Calling dict(args, kwargs) (line 140)
    dict_call_result_7013 = invoke(stypy.reporting.localization.Localization(__file__, 140, 18), dict_7007, *[environ_7009], **kwargs_7012)
    
    # Assigning a type to the variable 'env' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'env', dict_call_result_7013)
    
    # Assigning a BoolOp to a Name (line 142):
    
    # Assigning a BoolOp to a Name (line 142):
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    # Getting the type of 'search_path' (line 142)
    search_path_7014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 22), 'search_path')
    # Getting the type of 'os' (line 142)
    os_7015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 38), 'os')
    # Obtaining the member 'execvpe' of a type (line 142)
    execvpe_7016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 38), os_7015, 'execvpe')
    # Applying the binary operator 'and' (line 142)
    result_and_keyword_7017 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 22), 'and', search_path_7014, execvpe_7016)
    
    # Getting the type of 'os' (line 142)
    os_7018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 52), 'os')
    # Obtaining the member 'execve' of a type (line 142)
    execve_7019 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 52), os_7018, 'execve')
    # Applying the binary operator 'or' (line 142)
    result_or_keyword_7020 = python_operator(stypy.reporting.localization.Localization(__file__, 142, 22), 'or', result_and_keyword_7017, execve_7019)
    
    # Assigning a type to the variable 'exec_fn' (line 142)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'exec_fn', result_or_keyword_7020)
    # SSA join for if statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 143):
    
    # Assigning a Call to a Name (line 143):
    
    # Call to fork(...): (line 143)
    # Processing the call keyword arguments (line 143)
    kwargs_7023 = {}
    # Getting the type of 'os' (line 143)
    os_7021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 10), 'os', False)
    # Obtaining the member 'fork' of a type (line 143)
    fork_7022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 143, 10), os_7021, 'fork')
    # Calling fork(args, kwargs) (line 143)
    fork_call_result_7024 = invoke(stypy.reporting.localization.Localization(__file__, 143, 10), fork_7022, *[], **kwargs_7023)
    
    # Assigning a type to the variable 'pid' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'pid', fork_call_result_7024)
    
    
    # Getting the type of 'pid' (line 145)
    pid_7025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 7), 'pid')
    int_7026 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 14), 'int')
    # Applying the binary operator '==' (line 145)
    result_eq_7027 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 7), '==', pid_7025, int_7026)
    
    # Testing the type of an if condition (line 145)
    if_condition_7028 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 145, 4), result_eq_7027)
    # Assigning a type to the variable 'if_condition_7028' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 4), 'if_condition_7028', if_condition_7028)
    # SSA begins for if statement (line 145)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 146)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Type idiom detected: calculating its left and rigth part (line 147)
    # Getting the type of 'env' (line 147)
    env_7029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 15), 'env')
    # Getting the type of 'None' (line 147)
    None_7030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 22), 'None')
    
    (may_be_7031, more_types_in_union_7032) = may_be_none(env_7029, None_7030)

    if may_be_7031:

        if more_types_in_union_7032:
            # Runtime conditional SSA (line 147)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to exec_fn(...): (line 148)
        # Processing the call arguments (line 148)
        # Getting the type of 'executable' (line 148)
        executable_7034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 24), 'executable', False)
        # Getting the type of 'cmd' (line 148)
        cmd_7035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 36), 'cmd', False)
        # Processing the call keyword arguments (line 148)
        kwargs_7036 = {}
        # Getting the type of 'exec_fn' (line 148)
        exec_fn_7033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 16), 'exec_fn', False)
        # Calling exec_fn(args, kwargs) (line 148)
        exec_fn_call_result_7037 = invoke(stypy.reporting.localization.Localization(__file__, 148, 16), exec_fn_7033, *[executable_7034, cmd_7035], **kwargs_7036)
        

        if more_types_in_union_7032:
            # Runtime conditional SSA for else branch (line 147)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_7031) or more_types_in_union_7032):
        
        # Call to exec_fn(...): (line 150)
        # Processing the call arguments (line 150)
        # Getting the type of 'executable' (line 150)
        executable_7039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 24), 'executable', False)
        # Getting the type of 'cmd' (line 150)
        cmd_7040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 36), 'cmd', False)
        # Getting the type of 'env' (line 150)
        env_7041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 41), 'env', False)
        # Processing the call keyword arguments (line 150)
        kwargs_7042 = {}
        # Getting the type of 'exec_fn' (line 150)
        exec_fn_7038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'exec_fn', False)
        # Calling exec_fn(args, kwargs) (line 150)
        exec_fn_call_result_7043 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), exec_fn_7038, *[executable_7039, cmd_7040, env_7041], **kwargs_7042)
        

        if (may_be_7031 and more_types_in_union_7032):
            # SSA join for if statement (line 147)
            module_type_store = module_type_store.join_ssa_context()


    
    # SSA branch for the except part of a try statement (line 146)
    # SSA branch for the except 'OSError' branch of a try statement (line 146)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'OSError' (line 151)
    OSError_7044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 15), 'OSError')
    # Assigning a type to the variable 'e' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'e', OSError_7044)
    
    
    # Getting the type of 'DEBUG' (line 152)
    DEBUG_7045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 19), 'DEBUG')
    # Applying the 'not' unary operator (line 152)
    result_not__7046 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 15), 'not', DEBUG_7045)
    
    # Testing the type of an if condition (line 152)
    if_condition_7047 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 152, 12), result_not__7046)
    # Assigning a type to the variable 'if_condition_7047' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'if_condition_7047', if_condition_7047)
    # SSA begins for if statement (line 152)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 153):
    
    # Assigning a Name to a Name (line 153):
    # Getting the type of 'executable' (line 153)
    executable_7048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 22), 'executable')
    # Assigning a type to the variable 'cmd' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 16), 'cmd', executable_7048)
    # SSA join for if statement (line 152)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 154)
    # Processing the call arguments (line 154)
    str_7052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 29), 'str', 'unable to execute %r: %s\n')
    
    # Obtaining an instance of the builtin type 'tuple' (line 155)
    tuple_7053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 155, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 155)
    # Adding element type (line 155)
    # Getting the type of 'cmd' (line 155)
    cmd_7054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 30), 'cmd', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 30), tuple_7053, cmd_7054)
    # Adding element type (line 155)
    # Getting the type of 'e' (line 155)
    e_7055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 35), 'e', False)
    # Obtaining the member 'strerror' of a type (line 155)
    strerror_7056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 155, 35), e_7055, 'strerror')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 155, 30), tuple_7053, strerror_7056)
    
    # Applying the binary operator '%' (line 154)
    result_mod_7057 = python_operator(stypy.reporting.localization.Localization(__file__, 154, 29), '%', str_7052, tuple_7053)
    
    # Processing the call keyword arguments (line 154)
    kwargs_7058 = {}
    # Getting the type of 'sys' (line 154)
    sys_7049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 12), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 154)
    stderr_7050 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), sys_7049, 'stderr')
    # Obtaining the member 'write' of a type (line 154)
    write_7051 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 12), stderr_7050, 'write')
    # Calling write(args, kwargs) (line 154)
    write_call_result_7059 = invoke(stypy.reporting.localization.Localization(__file__, 154, 12), write_7051, *[result_mod_7057], **kwargs_7058)
    
    
    # Call to _exit(...): (line 156)
    # Processing the call arguments (line 156)
    int_7062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 21), 'int')
    # Processing the call keyword arguments (line 156)
    kwargs_7063 = {}
    # Getting the type of 'os' (line 156)
    os_7060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 12), 'os', False)
    # Obtaining the member '_exit' of a type (line 156)
    _exit_7061 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 12), os_7060, '_exit')
    # Calling _exit(args, kwargs) (line 156)
    _exit_call_result_7064 = invoke(stypy.reporting.localization.Localization(__file__, 156, 12), _exit_7061, *[int_7062], **kwargs_7063)
    
    # SSA join for try-except statement (line 146)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'DEBUG' (line 158)
    DEBUG_7065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'DEBUG')
    # Applying the 'not' unary operator (line 158)
    result_not__7066 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), 'not', DEBUG_7065)
    
    # Testing the type of an if condition (line 158)
    if_condition_7067 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_not__7066)
    # Assigning a type to the variable 'if_condition_7067' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_7067', if_condition_7067)
    # SSA begins for if statement (line 158)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 159):
    
    # Assigning a Name to a Name (line 159):
    # Getting the type of 'executable' (line 159)
    executable_7068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 18), 'executable')
    # Assigning a type to the variable 'cmd' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'cmd', executable_7068)
    # SSA join for if statement (line 158)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to write(...): (line 160)
    # Processing the call arguments (line 160)
    str_7072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 160, 25), 'str', 'unable to execute %r for unknown reasons')
    # Getting the type of 'cmd' (line 160)
    cmd_7073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 70), 'cmd', False)
    # Applying the binary operator '%' (line 160)
    result_mod_7074 = python_operator(stypy.reporting.localization.Localization(__file__, 160, 25), '%', str_7072, cmd_7073)
    
    # Processing the call keyword arguments (line 160)
    kwargs_7075 = {}
    # Getting the type of 'sys' (line 160)
    sys_7069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 8), 'sys', False)
    # Obtaining the member 'stderr' of a type (line 160)
    stderr_7070 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), sys_7069, 'stderr')
    # Obtaining the member 'write' of a type (line 160)
    write_7071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 8), stderr_7070, 'write')
    # Calling write(args, kwargs) (line 160)
    write_call_result_7076 = invoke(stypy.reporting.localization.Localization(__file__, 160, 8), write_7071, *[result_mod_7074], **kwargs_7075)
    
    
    # Call to _exit(...): (line 161)
    # Processing the call arguments (line 161)
    int_7079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 17), 'int')
    # Processing the call keyword arguments (line 161)
    kwargs_7080 = {}
    # Getting the type of 'os' (line 161)
    os_7077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'os', False)
    # Obtaining the member '_exit' of a type (line 161)
    _exit_7078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 161, 8), os_7077, '_exit')
    # Calling _exit(args, kwargs) (line 161)
    _exit_call_result_7081 = invoke(stypy.reporting.localization.Localization(__file__, 161, 8), _exit_7078, *[int_7079], **kwargs_7080)
    
    # SSA branch for the else part of an if statement (line 145)
    module_type_store.open_ssa_branch('else')
    
    int_7082 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 165, 14), 'int')
    # Testing the type of an if condition (line 165)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 165, 8), int_7082)
    # SSA begins for while statement (line 165)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # SSA begins for try-except statement (line 166)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Tuple (line 167):
    
    # Assigning a Subscript to a Name (line 167):
    
    # Obtaining the type of the subscript
    int_7083 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 16), 'int')
    
    # Call to waitpid(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'pid' (line 167)
    pid_7086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 'pid', False)
    int_7087 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 46), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_7088 = {}
    # Getting the type of 'os' (line 167)
    os_7084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'os', False)
    # Obtaining the member 'waitpid' of a type (line 167)
    waitpid_7085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 30), os_7084, 'waitpid')
    # Calling waitpid(args, kwargs) (line 167)
    waitpid_call_result_7089 = invoke(stypy.reporting.localization.Localization(__file__, 167, 30), waitpid_7085, *[pid_7086, int_7087], **kwargs_7088)
    
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___7090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), waitpid_call_result_7089, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_7091 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), getitem___7090, int_7083)
    
    # Assigning a type to the variable 'tuple_var_assignment_6711' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'tuple_var_assignment_6711', subscript_call_result_7091)
    
    # Assigning a Subscript to a Name (line 167):
    
    # Obtaining the type of the subscript
    int_7092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 16), 'int')
    
    # Call to waitpid(...): (line 167)
    # Processing the call arguments (line 167)
    # Getting the type of 'pid' (line 167)
    pid_7095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 41), 'pid', False)
    int_7096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 46), 'int')
    # Processing the call keyword arguments (line 167)
    kwargs_7097 = {}
    # Getting the type of 'os' (line 167)
    os_7093 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'os', False)
    # Obtaining the member 'waitpid' of a type (line 167)
    waitpid_7094 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 30), os_7093, 'waitpid')
    # Calling waitpid(args, kwargs) (line 167)
    waitpid_call_result_7098 = invoke(stypy.reporting.localization.Localization(__file__, 167, 30), waitpid_7094, *[pid_7095, int_7096], **kwargs_7097)
    
    # Obtaining the member '__getitem__' of a type (line 167)
    getitem___7099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 16), waitpid_call_result_7098, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 167)
    subscript_call_result_7100 = invoke(stypy.reporting.localization.Localization(__file__, 167, 16), getitem___7099, int_7092)
    
    # Assigning a type to the variable 'tuple_var_assignment_6712' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'tuple_var_assignment_6712', subscript_call_result_7100)
    
    # Assigning a Name to a Name (line 167):
    # Getting the type of 'tuple_var_assignment_6711' (line 167)
    tuple_var_assignment_6711_7101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'tuple_var_assignment_6711')
    # Assigning a type to the variable 'pid' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'pid', tuple_var_assignment_6711_7101)
    
    # Assigning a Name to a Name (line 167):
    # Getting the type of 'tuple_var_assignment_6712' (line 167)
    tuple_var_assignment_6712_7102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 16), 'tuple_var_assignment_6712')
    # Assigning a type to the variable 'status' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 21), 'status', tuple_var_assignment_6712_7102)
    # SSA branch for the except part of a try statement (line 166)
    # SSA branch for the except 'OSError' branch of a try statement (line 166)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'OSError' (line 168)
    OSError_7103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'OSError')
    # Assigning a type to the variable 'exc' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'exc', OSError_7103)
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 169, 16))
    
    # 'import errno' statement (line 169)
    import errno

    import_module(stypy.reporting.localization.Localization(__file__, 169, 16), 'errno', errno, module_type_store)
    
    
    
    # Getting the type of 'exc' (line 170)
    exc_7104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 19), 'exc')
    # Obtaining the member 'errno' of a type (line 170)
    errno_7105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 19), exc_7104, 'errno')
    # Getting the type of 'errno' (line 170)
    errno_7106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 32), 'errno')
    # Obtaining the member 'EINTR' of a type (line 170)
    EINTR_7107 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 32), errno_7106, 'EINTR')
    # Applying the binary operator '==' (line 170)
    result_eq_7108 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 19), '==', errno_7105, EINTR_7107)
    
    # Testing the type of an if condition (line 170)
    if_condition_7109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 170, 16), result_eq_7108)
    # Assigning a type to the variable 'if_condition_7109' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 16), 'if_condition_7109', if_condition_7109)
    # SSA begins for if statement (line 170)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 170)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'DEBUG' (line 172)
    DEBUG_7110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 23), 'DEBUG')
    # Applying the 'not' unary operator (line 172)
    result_not__7111 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 19), 'not', DEBUG_7110)
    
    # Testing the type of an if condition (line 172)
    if_condition_7112 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 16), result_not__7111)
    # Assigning a type to the variable 'if_condition_7112' (line 172)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 16), 'if_condition_7112', if_condition_7112)
    # SSA begins for if statement (line 172)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 173):
    
    # Assigning a Name to a Name (line 173):
    # Getting the type of 'executable' (line 173)
    executable_7113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 26), 'executable')
    # Assigning a type to the variable 'cmd' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 20), 'cmd', executable_7113)
    # SSA join for if statement (line 172)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'DistutilsExecError' (line 174)
    DistutilsExecError_7114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 22), 'DistutilsExecError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 174, 16), DistutilsExecError_7114, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 166)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to WIFSIGNALED(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'status' (line 176)
    status_7117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 30), 'status', False)
    # Processing the call keyword arguments (line 176)
    kwargs_7118 = {}
    # Getting the type of 'os' (line 176)
    os_7115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 15), 'os', False)
    # Obtaining the member 'WIFSIGNALED' of a type (line 176)
    WIFSIGNALED_7116 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 15), os_7115, 'WIFSIGNALED')
    # Calling WIFSIGNALED(args, kwargs) (line 176)
    WIFSIGNALED_call_result_7119 = invoke(stypy.reporting.localization.Localization(__file__, 176, 15), WIFSIGNALED_7116, *[status_7117], **kwargs_7118)
    
    # Testing the type of an if condition (line 176)
    if_condition_7120 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 12), WIFSIGNALED_call_result_7119)
    # Assigning a type to the variable 'if_condition_7120' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 12), 'if_condition_7120', if_condition_7120)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'DEBUG' (line 177)
    DEBUG_7121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 23), 'DEBUG')
    # Applying the 'not' unary operator (line 177)
    result_not__7122 = python_operator(stypy.reporting.localization.Localization(__file__, 177, 19), 'not', DEBUG_7121)
    
    # Testing the type of an if condition (line 177)
    if_condition_7123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 177, 16), result_not__7122)
    # Assigning a type to the variable 'if_condition_7123' (line 177)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 177, 16), 'if_condition_7123', if_condition_7123)
    # SSA begins for if statement (line 177)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 178):
    
    # Assigning a Name to a Name (line 178):
    # Getting the type of 'executable' (line 178)
    executable_7124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 26), 'executable')
    # Assigning a type to the variable 'cmd' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 20), 'cmd', executable_7124)
    # SSA join for if statement (line 177)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'DistutilsExecError' (line 179)
    DistutilsExecError_7125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 22), 'DistutilsExecError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 179, 16), DistutilsExecError_7125, 'raise parameter', BaseException)
    # SSA branch for the else part of an if statement (line 176)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to WIFEXITED(...): (line 183)
    # Processing the call arguments (line 183)
    # Getting the type of 'status' (line 183)
    status_7128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 30), 'status', False)
    # Processing the call keyword arguments (line 183)
    kwargs_7129 = {}
    # Getting the type of 'os' (line 183)
    os_7126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'os', False)
    # Obtaining the member 'WIFEXITED' of a type (line 183)
    WIFEXITED_7127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 17), os_7126, 'WIFEXITED')
    # Calling WIFEXITED(args, kwargs) (line 183)
    WIFEXITED_call_result_7130 = invoke(stypy.reporting.localization.Localization(__file__, 183, 17), WIFEXITED_7127, *[status_7128], **kwargs_7129)
    
    # Testing the type of an if condition (line 183)
    if_condition_7131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 17), WIFEXITED_call_result_7130)
    # Assigning a type to the variable 'if_condition_7131' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 17), 'if_condition_7131', if_condition_7131)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 184):
    
    # Assigning a Call to a Name (line 184):
    
    # Call to WEXITSTATUS(...): (line 184)
    # Processing the call arguments (line 184)
    # Getting the type of 'status' (line 184)
    status_7134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 45), 'status', False)
    # Processing the call keyword arguments (line 184)
    kwargs_7135 = {}
    # Getting the type of 'os' (line 184)
    os_7132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 30), 'os', False)
    # Obtaining the member 'WEXITSTATUS' of a type (line 184)
    WEXITSTATUS_7133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 30), os_7132, 'WEXITSTATUS')
    # Calling WEXITSTATUS(args, kwargs) (line 184)
    WEXITSTATUS_call_result_7136 = invoke(stypy.reporting.localization.Localization(__file__, 184, 30), WEXITSTATUS_7133, *[status_7134], **kwargs_7135)
    
    # Assigning a type to the variable 'exit_status' (line 184)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 16), 'exit_status', WEXITSTATUS_call_result_7136)
    
    
    # Getting the type of 'exit_status' (line 185)
    exit_status_7137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 19), 'exit_status')
    int_7138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 185, 34), 'int')
    # Applying the binary operator '==' (line 185)
    result_eq_7139 = python_operator(stypy.reporting.localization.Localization(__file__, 185, 19), '==', exit_status_7137, int_7138)
    
    # Testing the type of an if condition (line 185)
    if_condition_7140 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 185, 16), result_eq_7139)
    # Assigning a type to the variable 'if_condition_7140' (line 185)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 16), 'if_condition_7140', if_condition_7140)
    # SSA begins for if statement (line 185)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 20), 'stypy_return_type', types.NoneType)
    # SSA branch for the else part of an if statement (line 185)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'DEBUG' (line 188)
    DEBUG_7141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 27), 'DEBUG')
    # Applying the 'not' unary operator (line 188)
    result_not__7142 = python_operator(stypy.reporting.localization.Localization(__file__, 188, 23), 'not', DEBUG_7141)
    
    # Testing the type of an if condition (line 188)
    if_condition_7143 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 188, 20), result_not__7142)
    # Assigning a type to the variable 'if_condition_7143' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 20), 'if_condition_7143', if_condition_7143)
    # SSA begins for if statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 189):
    
    # Assigning a Name to a Name (line 189):
    # Getting the type of 'executable' (line 189)
    executable_7144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 30), 'executable')
    # Assigning a type to the variable 'cmd' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 24), 'cmd', executable_7144)
    # SSA join for if statement (line 188)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'DistutilsExecError' (line 190)
    DistutilsExecError_7145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 26), 'DistutilsExecError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 190, 20), DistutilsExecError_7145, 'raise parameter', BaseException)
    # SSA join for if statement (line 185)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 183)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to WIFSTOPPED(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'status' (line 194)
    status_7148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 31), 'status', False)
    # Processing the call keyword arguments (line 194)
    kwargs_7149 = {}
    # Getting the type of 'os' (line 194)
    os_7146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'os', False)
    # Obtaining the member 'WIFSTOPPED' of a type (line 194)
    WIFSTOPPED_7147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 194, 17), os_7146, 'WIFSTOPPED')
    # Calling WIFSTOPPED(args, kwargs) (line 194)
    WIFSTOPPED_call_result_7150 = invoke(stypy.reporting.localization.Localization(__file__, 194, 17), WIFSTOPPED_7147, *[status_7148], **kwargs_7149)
    
    # Testing the type of an if condition (line 194)
    if_condition_7151 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 17), WIFSTOPPED_call_result_7150)
    # Assigning a type to the variable 'if_condition_7151' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 17), 'if_condition_7151', if_condition_7151)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA branch for the else part of an if statement (line 194)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'DEBUG' (line 198)
    DEBUG_7152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 23), 'DEBUG')
    # Applying the 'not' unary operator (line 198)
    result_not__7153 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 19), 'not', DEBUG_7152)
    
    # Testing the type of an if condition (line 198)
    if_condition_7154 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 16), result_not__7153)
    # Assigning a type to the variable 'if_condition_7154' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 16), 'if_condition_7154', if_condition_7154)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 199):
    
    # Assigning a Name to a Name (line 199):
    # Getting the type of 'executable' (line 199)
    executable_7155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 26), 'executable')
    # Assigning a type to the variable 'cmd' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'cmd', executable_7155)
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'DistutilsExecError' (line 200)
    DistutilsExecError_7156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'DistutilsExecError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 200, 16), DistutilsExecError_7156, 'raise parameter', BaseException)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 165)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 145)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of '_spawn_posix(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_spawn_posix' in the type store
    # Getting the type of 'stypy_return_type' (line 116)
    stypy_return_type_7157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7157)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_spawn_posix'
    return stypy_return_type_7157

# Assigning a type to the variable '_spawn_posix' (line 116)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 0), '_spawn_posix', _spawn_posix)

@norecursion
def find_executable(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 204)
    None_7158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 37), 'None')
    defaults = [None_7158]
    # Create a new context for function 'find_executable'
    module_type_store = module_type_store.open_function_context('find_executable', 204, 0, False)
    
    # Passed parameters checking function
    find_executable.stypy_localization = localization
    find_executable.stypy_type_of_self = None
    find_executable.stypy_type_store = module_type_store
    find_executable.stypy_function_name = 'find_executable'
    find_executable.stypy_param_names_list = ['executable', 'path']
    find_executable.stypy_varargs_param_name = None
    find_executable.stypy_kwargs_param_name = None
    find_executable.stypy_call_defaults = defaults
    find_executable.stypy_call_varargs = varargs
    find_executable.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'find_executable', ['executable', 'path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'find_executable', localization, ['executable', 'path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'find_executable(...)' code ##################

    str_7159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, (-1)), 'str', "Tries to find 'executable' in the directories listed in 'path'.\n\n    A string listing directories separated by 'os.pathsep'; defaults to\n    os.environ['PATH'].  Returns the complete filename or None if not found.\n    ")
    
    # Type idiom detected: calculating its left and rigth part (line 210)
    # Getting the type of 'path' (line 210)
    path_7160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 7), 'path')
    # Getting the type of 'None' (line 210)
    None_7161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 15), 'None')
    
    (may_be_7162, more_types_in_union_7163) = may_be_none(path_7160, None_7161)

    if may_be_7162:

        if more_types_in_union_7163:
            # Runtime conditional SSA (line 210)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Subscript to a Name (line 211):
        
        # Assigning a Subscript to a Name (line 211):
        
        # Obtaining the type of the subscript
        str_7164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 26), 'str', 'PATH')
        # Getting the type of 'os' (line 211)
        os_7165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'os')
        # Obtaining the member 'environ' of a type (line 211)
        environ_7166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 15), os_7165, 'environ')
        # Obtaining the member '__getitem__' of a type (line 211)
        getitem___7167 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 15), environ_7166, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 211)
        subscript_call_result_7168 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), getitem___7167, str_7164)
        
        # Assigning a type to the variable 'path' (line 211)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 8), 'path', subscript_call_result_7168)

        if more_types_in_union_7163:
            # SSA join for if statement (line 210)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 212):
    
    # Assigning a Call to a Name (line 212):
    
    # Call to split(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'os' (line 212)
    os_7171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 'os', False)
    # Obtaining the member 'pathsep' of a type (line 212)
    pathsep_7172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 23), os_7171, 'pathsep')
    # Processing the call keyword arguments (line 212)
    kwargs_7173 = {}
    # Getting the type of 'path' (line 212)
    path_7169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 12), 'path', False)
    # Obtaining the member 'split' of a type (line 212)
    split_7170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 12), path_7169, 'split')
    # Calling split(args, kwargs) (line 212)
    split_call_result_7174 = invoke(stypy.reporting.localization.Localization(__file__, 212, 12), split_7170, *[pathsep_7172], **kwargs_7173)
    
    # Assigning a type to the variable 'paths' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'paths', split_call_result_7174)
    
    # Assigning a Call to a Tuple (line 213):
    
    # Assigning a Subscript to a Name (line 213):
    
    # Obtaining the type of the subscript
    int_7175 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 4), 'int')
    
    # Call to splitext(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'executable' (line 213)
    executable_7179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'executable', False)
    # Processing the call keyword arguments (line 213)
    kwargs_7180 = {}
    # Getting the type of 'os' (line 213)
    os_7176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 213)
    path_7177 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), os_7176, 'path')
    # Obtaining the member 'splitext' of a type (line 213)
    splitext_7178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), path_7177, 'splitext')
    # Calling splitext(args, kwargs) (line 213)
    splitext_call_result_7181 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), splitext_7178, *[executable_7179], **kwargs_7180)
    
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___7182 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 4), splitext_call_result_7181, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_7183 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), getitem___7182, int_7175)
    
    # Assigning a type to the variable 'tuple_var_assignment_6713' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_var_assignment_6713', subscript_call_result_7183)
    
    # Assigning a Subscript to a Name (line 213):
    
    # Obtaining the type of the subscript
    int_7184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 4), 'int')
    
    # Call to splitext(...): (line 213)
    # Processing the call arguments (line 213)
    # Getting the type of 'executable' (line 213)
    executable_7188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 33), 'executable', False)
    # Processing the call keyword arguments (line 213)
    kwargs_7189 = {}
    # Getting the type of 'os' (line 213)
    os_7185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 213)
    path_7186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), os_7185, 'path')
    # Obtaining the member 'splitext' of a type (line 213)
    splitext_7187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 16), path_7186, 'splitext')
    # Calling splitext(args, kwargs) (line 213)
    splitext_call_result_7190 = invoke(stypy.reporting.localization.Localization(__file__, 213, 16), splitext_7187, *[executable_7188], **kwargs_7189)
    
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___7191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 4), splitext_call_result_7190, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_7192 = invoke(stypy.reporting.localization.Localization(__file__, 213, 4), getitem___7191, int_7184)
    
    # Assigning a type to the variable 'tuple_var_assignment_6714' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_var_assignment_6714', subscript_call_result_7192)
    
    # Assigning a Name to a Name (line 213):
    # Getting the type of 'tuple_var_assignment_6713' (line 213)
    tuple_var_assignment_6713_7193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_var_assignment_6713')
    # Assigning a type to the variable 'base' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'base', tuple_var_assignment_6713_7193)
    
    # Assigning a Name to a Name (line 213):
    # Getting the type of 'tuple_var_assignment_6714' (line 213)
    tuple_var_assignment_6714_7194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 4), 'tuple_var_assignment_6714')
    # Assigning a type to the variable 'ext' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 10), 'ext', tuple_var_assignment_6714_7194)
    
    
    # Evaluating a boolean operation
    
    # Evaluating a boolean operation
    
    # Getting the type of 'sys' (line 215)
    sys_7195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'sys')
    # Obtaining the member 'platform' of a type (line 215)
    platform_7196 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 8), sys_7195, 'platform')
    str_7197 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 24), 'str', 'win32')
    # Applying the binary operator '==' (line 215)
    result_eq_7198 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 8), '==', platform_7196, str_7197)
    
    
    # Getting the type of 'os' (line 215)
    os_7199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 35), 'os')
    # Obtaining the member 'name' of a type (line 215)
    name_7200 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 215, 35), os_7199, 'name')
    str_7201 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 46), 'str', 'os2')
    # Applying the binary operator '==' (line 215)
    result_eq_7202 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 35), '==', name_7200, str_7201)
    
    # Applying the binary operator 'or' (line 215)
    result_or_keyword_7203 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 8), 'or', result_eq_7198, result_eq_7202)
    
    
    # Getting the type of 'ext' (line 215)
    ext_7204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 58), 'ext')
    str_7205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 65), 'str', '.exe')
    # Applying the binary operator '!=' (line 215)
    result_ne_7206 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 58), '!=', ext_7204, str_7205)
    
    # Applying the binary operator 'and' (line 215)
    result_and_keyword_7207 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 7), 'and', result_or_keyword_7203, result_ne_7206)
    
    # Testing the type of an if condition (line 215)
    if_condition_7208 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 4), result_and_keyword_7207)
    # Assigning a type to the variable 'if_condition_7208' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 4), 'if_condition_7208', if_condition_7208)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 216):
    
    # Assigning a BinOp to a Name (line 216):
    # Getting the type of 'executable' (line 216)
    executable_7209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'executable')
    str_7210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 216, 34), 'str', '.exe')
    # Applying the binary operator '+' (line 216)
    result_add_7211 = python_operator(stypy.reporting.localization.Localization(__file__, 216, 21), '+', executable_7209, str_7210)
    
    # Assigning a type to the variable 'executable' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 8), 'executable', result_add_7211)
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isfile(...): (line 218)
    # Processing the call arguments (line 218)
    # Getting the type of 'executable' (line 218)
    executable_7215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 26), 'executable', False)
    # Processing the call keyword arguments (line 218)
    kwargs_7216 = {}
    # Getting the type of 'os' (line 218)
    os_7212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 218)
    path_7213 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), os_7212, 'path')
    # Obtaining the member 'isfile' of a type (line 218)
    isfile_7214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 11), path_7213, 'isfile')
    # Calling isfile(args, kwargs) (line 218)
    isfile_call_result_7217 = invoke(stypy.reporting.localization.Localization(__file__, 218, 11), isfile_7214, *[executable_7215], **kwargs_7216)
    
    # Applying the 'not' unary operator (line 218)
    result_not__7218 = python_operator(stypy.reporting.localization.Localization(__file__, 218, 7), 'not', isfile_call_result_7217)
    
    # Testing the type of an if condition (line 218)
    if_condition_7219 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 218, 4), result_not__7218)
    # Assigning a type to the variable 'if_condition_7219' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 4), 'if_condition_7219', if_condition_7219)
    # SSA begins for if statement (line 218)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'paths' (line 219)
    paths_7220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 17), 'paths')
    # Testing the type of a for loop iterable (line 219)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 219, 8), paths_7220)
    # Getting the type of the for loop variable (line 219)
    for_loop_var_7221 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 219, 8), paths_7220)
    # Assigning a type to the variable 'p' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'p', for_loop_var_7221)
    # SSA begins for a for statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 220):
    
    # Assigning a Call to a Name (line 220):
    
    # Call to join(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'p' (line 220)
    p_7225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 29), 'p', False)
    # Getting the type of 'executable' (line 220)
    executable_7226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 32), 'executable', False)
    # Processing the call keyword arguments (line 220)
    kwargs_7227 = {}
    # Getting the type of 'os' (line 220)
    os_7222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 220)
    path_7223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), os_7222, 'path')
    # Obtaining the member 'join' of a type (line 220)
    join_7224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), path_7223, 'join')
    # Calling join(args, kwargs) (line 220)
    join_call_result_7228 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), join_7224, *[p_7225, executable_7226], **kwargs_7227)
    
    # Assigning a type to the variable 'f' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 12), 'f', join_call_result_7228)
    
    
    # Call to isfile(...): (line 221)
    # Processing the call arguments (line 221)
    # Getting the type of 'f' (line 221)
    f_7232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 30), 'f', False)
    # Processing the call keyword arguments (line 221)
    kwargs_7233 = {}
    # Getting the type of 'os' (line 221)
    os_7229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 221)
    path_7230 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 15), os_7229, 'path')
    # Obtaining the member 'isfile' of a type (line 221)
    isfile_7231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 221, 15), path_7230, 'isfile')
    # Calling isfile(args, kwargs) (line 221)
    isfile_call_result_7234 = invoke(stypy.reporting.localization.Localization(__file__, 221, 15), isfile_7231, *[f_7232], **kwargs_7233)
    
    # Testing the type of an if condition (line 221)
    if_condition_7235 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 12), isfile_call_result_7234)
    # Assigning a type to the variable 'if_condition_7235' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 12), 'if_condition_7235', if_condition_7235)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'f' (line 223)
    f_7236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 23), 'f')
    # Assigning a type to the variable 'stypy_return_type' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 16), 'stypy_return_type', f_7236)
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 224)
    None_7237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'stypy_return_type', None_7237)
    # SSA branch for the else part of an if statement (line 218)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'executable' (line 226)
    executable_7238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 15), 'executable')
    # Assigning a type to the variable 'stypy_return_type' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 8), 'stypy_return_type', executable_7238)
    # SSA join for if statement (line 218)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'find_executable(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'find_executable' in the type store
    # Getting the type of 'stypy_return_type' (line 204)
    stypy_return_type_7239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_7239)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'find_executable'
    return stypy_return_type_7239

# Assigning a type to the variable 'find_executable' (line 204)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 0), 'find_executable', find_executable)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
