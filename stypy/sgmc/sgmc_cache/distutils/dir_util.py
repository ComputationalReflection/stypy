
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.dir_util
2: 
3: Utility functions for manipulating directories and directory trees.'''
4: 
5: __revision__ = "$Id$"
6: 
7: import os
8: import errno
9: from distutils.errors import DistutilsFileError, DistutilsInternalError
10: from distutils import log
11: 
12: # cache for by mkpath() -- in addition to cheapening redundant calls,
13: # eliminates redundant "creating /foo/bar/baz" messages in dry-run mode
14: _path_created = {}
15: 
16: # I don't use os.makedirs because a) it's new to Python 1.5.2, and
17: # b) it blows up if the directory already exists (I want to silently
18: # succeed in that case).
19: def mkpath(name, mode=0777, verbose=1, dry_run=0):
20:     '''Create a directory and any missing ancestor directories.
21: 
22:     If the directory already exists (or if 'name' is the empty string, which
23:     means the current directory, which of course exists), then do nothing.
24:     Raise DistutilsFileError if unable to create some directory along the way
25:     (eg. some sub-path exists, but is a file rather than a directory).
26:     If 'verbose' is true, print a one-line summary of each mkdir to stdout.
27:     Return the list of directories actually created.
28:     '''
29: 
30:     global _path_created
31: 
32:     # Detect a common bug -- name is None
33:     if not isinstance(name, basestring):
34:         raise DistutilsInternalError, \
35:               "mkpath: 'name' must be a string (got %r)" % (name,)
36: 
37:     # XXX what's the better way to handle verbosity? print as we create
38:     # each directory in the path (the current behaviour), or only announce
39:     # the creation of the whole path? (quite easy to do the latter since
40:     # we're not using a recursive algorithm)
41: 
42:     name = os.path.normpath(name)
43:     created_dirs = []
44:     if os.path.isdir(name) or name == '':
45:         return created_dirs
46:     if _path_created.get(os.path.abspath(name)):
47:         return created_dirs
48: 
49:     (head, tail) = os.path.split(name)
50:     tails = [tail]                      # stack of lone dirs to create
51: 
52:     while head and tail and not os.path.isdir(head):
53:         (head, tail) = os.path.split(head)
54:         tails.insert(0, tail)          # push next higher dir onto stack
55: 
56:     # now 'head' contains the deepest directory that already exists
57:     # (that is, the child of 'head' in 'name' is the highest directory
58:     # that does *not* exist)
59:     for d in tails:
60:         #print "head = %s, d = %s: " % (head, d),
61:         head = os.path.join(head, d)
62:         abs_head = os.path.abspath(head)
63: 
64:         if _path_created.get(abs_head):
65:             continue
66: 
67:         if verbose >= 1:
68:             log.info("creating %s", head)
69: 
70:         if not dry_run:
71:             try:
72:                 os.mkdir(head, mode)
73:             except OSError, exc:
74:                 if not (exc.errno == errno.EEXIST and os.path.isdir(head)):
75:                     raise DistutilsFileError(
76:                           "could not create '%s': %s" % (head, exc.args[-1]))
77:             created_dirs.append(head)
78: 
79:         _path_created[abs_head] = 1
80:     return created_dirs
81: 
82: def create_tree(base_dir, files, mode=0777, verbose=1, dry_run=0):
83:     '''Create all the empty directories under 'base_dir' needed to put 'files'
84:     there.
85: 
86:     'base_dir' is just the name of a directory which doesn't necessarily
87:     exist yet; 'files' is a list of filenames to be interpreted relative to
88:     'base_dir'.  'base_dir' + the directory portion of every file in 'files'
89:     will be created if it doesn't already exist.  'mode', 'verbose' and
90:     'dry_run' flags are as for 'mkpath()'.
91:     '''
92:     # First get the list of directories to create
93:     need_dir = {}
94:     for file in files:
95:         need_dir[os.path.join(base_dir, os.path.dirname(file))] = 1
96:     need_dirs = need_dir.keys()
97:     need_dirs.sort()
98: 
99:     # Now create them
100:     for dir in need_dirs:
101:         mkpath(dir, mode, verbose=verbose, dry_run=dry_run)
102: 
103: def copy_tree(src, dst, preserve_mode=1, preserve_times=1,
104:               preserve_symlinks=0, update=0, verbose=1, dry_run=0):
105:     '''Copy an entire directory tree 'src' to a new location 'dst'.
106: 
107:     Both 'src' and 'dst' must be directory names.  If 'src' is not a
108:     directory, raise DistutilsFileError.  If 'dst' does not exist, it is
109:     created with 'mkpath()'.  The end result of the copy is that every
110:     file in 'src' is copied to 'dst', and directories under 'src' are
111:     recursively copied to 'dst'.  Return the list of files that were
112:     copied or might have been copied, using their output name.  The
113:     return value is unaffected by 'update' or 'dry_run': it is simply
114:     the list of all files under 'src', with the names changed to be
115:     under 'dst'.
116: 
117:     'preserve_mode' and 'preserve_times' are the same as for
118:     'copy_file'; note that they only apply to regular files, not to
119:     directories.  If 'preserve_symlinks' is true, symlinks will be
120:     copied as symlinks (on platforms that support them!); otherwise
121:     (the default), the destination of the symlink will be copied.
122:     'update' and 'verbose' are the same as for 'copy_file'.
123:     '''
124:     from distutils.file_util import copy_file
125: 
126:     if not dry_run and not os.path.isdir(src):
127:         raise DistutilsFileError, \
128:               "cannot copy tree '%s': not a directory" % src
129:     try:
130:         names = os.listdir(src)
131:     except os.error, (errno, errstr):
132:         if dry_run:
133:             names = []
134:         else:
135:             raise DistutilsFileError, \
136:                   "error listing files in '%s': %s" % (src, errstr)
137: 
138:     if not dry_run:
139:         mkpath(dst, verbose=verbose)
140: 
141:     outputs = []
142: 
143:     for n in names:
144:         src_name = os.path.join(src, n)
145:         dst_name = os.path.join(dst, n)
146: 
147:         if n.startswith('.nfs'):
148:             # skip NFS rename files
149:             continue
150: 
151:         if preserve_symlinks and os.path.islink(src_name):
152:             link_dest = os.readlink(src_name)
153:             if verbose >= 1:
154:                 log.info("linking %s -> %s", dst_name, link_dest)
155:             if not dry_run:
156:                 os.symlink(link_dest, dst_name)
157:             outputs.append(dst_name)
158: 
159:         elif os.path.isdir(src_name):
160:             outputs.extend(
161:                 copy_tree(src_name, dst_name, preserve_mode,
162:                           preserve_times, preserve_symlinks, update,
163:                           verbose=verbose, dry_run=dry_run))
164:         else:
165:             copy_file(src_name, dst_name, preserve_mode,
166:                       preserve_times, update, verbose=verbose,
167:                       dry_run=dry_run)
168:             outputs.append(dst_name)
169: 
170:     return outputs
171: 
172: def _build_cmdtuple(path, cmdtuples):
173:     '''Helper for remove_tree().'''
174:     for f in os.listdir(path):
175:         real_f = os.path.join(path,f)
176:         if os.path.isdir(real_f) and not os.path.islink(real_f):
177:             _build_cmdtuple(real_f, cmdtuples)
178:         else:
179:             cmdtuples.append((os.remove, real_f))
180:     cmdtuples.append((os.rmdir, path))
181: 
182: def remove_tree(directory, verbose=1, dry_run=0):
183:     '''Recursively remove an entire directory tree.
184: 
185:     Any errors are ignored (apart from being reported to stdout if 'verbose'
186:     is true).
187:     '''
188:     global _path_created
189: 
190:     if verbose >= 1:
191:         log.info("removing '%s' (and everything under it)", directory)
192:     if dry_run:
193:         return
194:     cmdtuples = []
195:     _build_cmdtuple(directory, cmdtuples)
196:     for cmd in cmdtuples:
197:         try:
198:             cmd[0](cmd[1])
199:             # remove dir from cache if it's already there
200:             abspath = os.path.abspath(cmd[1])
201:             if abspath in _path_created:
202:                 del _path_created[abspath]
203:         except (IOError, OSError), exc:
204:             log.warn("error removing %s: %s", directory, exc)
205: 
206: def ensure_relative(path):
207:     '''Take the full path 'path', and make it a relative path.
208: 
209:     This is useful to make 'path' the second argument to os.path.join().
210:     '''
211:     drive, path = os.path.splitdrive(path)
212:     if path[0:1] == os.sep:
213:         path = drive + path[1:]
214:     return path
215: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_7 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', 'distutils.dir_util\n\nUtility functions for manipulating directories and directory trees.')

# Assigning a Str to a Name (line 5):

# Assigning a Str to a Name (line 5):
str_8 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__revision__', str_8)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'import os' statement (line 7)
import os

import_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import errno' statement (line 8)
import errno

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'errno', errno, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsFileError, DistutilsInternalError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_9 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_9) is not StypyTypeError):

    if (import_9 != 'pyd_module'):
        __import__(import_9)
        sys_modules_10 = sys.modules[import_9]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_10.module_type_store, module_type_store, ['DistutilsFileError', 'DistutilsInternalError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_10, sys_modules_10.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsFileError, DistutilsInternalError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsFileError', 'DistutilsInternalError'], [DistutilsFileError, DistutilsInternalError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_9)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils import log' statement (line 10)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils', None, module_type_store, ['log'], [log])


# Assigning a Dict to a Name (line 14):

# Assigning a Dict to a Name (line 14):

# Obtaining an instance of the builtin type 'dict' (line 14)
dict_11 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 14)

# Assigning a type to the variable '_path_created' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), '_path_created', dict_11)

@norecursion
def mkpath(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_12 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'int')
    int_13 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 36), 'int')
    int_14 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 47), 'int')
    defaults = [int_12, int_13, int_14]
    # Create a new context for function 'mkpath'
    module_type_store = module_type_store.open_function_context('mkpath', 19, 0, False)
    
    # Passed parameters checking function
    mkpath.stypy_localization = localization
    mkpath.stypy_type_of_self = None
    mkpath.stypy_type_store = module_type_store
    mkpath.stypy_function_name = 'mkpath'
    mkpath.stypy_param_names_list = ['name', 'mode', 'verbose', 'dry_run']
    mkpath.stypy_varargs_param_name = None
    mkpath.stypy_kwargs_param_name = None
    mkpath.stypy_call_defaults = defaults
    mkpath.stypy_call_varargs = varargs
    mkpath.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'mkpath', ['name', 'mode', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'mkpath', localization, ['name', 'mode', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'mkpath(...)' code ##################

    str_15 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', "Create a directory and any missing ancestor directories.\n\n    If the directory already exists (or if 'name' is the empty string, which\n    means the current directory, which of course exists), then do nothing.\n    Raise DistutilsFileError if unable to create some directory along the way\n    (eg. some sub-path exists, but is a file rather than a directory).\n    If 'verbose' is true, print a one-line summary of each mkdir to stdout.\n    Return the list of directories actually created.\n    ")
    # Marking variables as global (line 30)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 30, 4), '_path_created')
    
    # Type idiom detected: calculating its left and rigth part (line 33)
    # Getting the type of 'basestring' (line 33)
    basestring_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 28), 'basestring')
    # Getting the type of 'name' (line 33)
    name_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 22), 'name')
    
    (may_be_18, more_types_in_union_19) = may_not_be_subtype(basestring_16, name_17)

    if may_be_18:

        if more_types_in_union_19:
            # Runtime conditional SSA (line 33)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        # Assigning a type to the variable 'name' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'name', remove_subtype_from_union(name_17, basestring))
        # Getting the type of 'DistutilsInternalError' (line 34)
        DistutilsInternalError_20 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 14), 'DistutilsInternalError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 34, 8), DistutilsInternalError_20, 'raise parameter', BaseException)

        if more_types_in_union_19:
            # SSA join for if statement (line 33)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Call to a Name (line 42):
    
    # Assigning a Call to a Name (line 42):
    
    # Call to normpath(...): (line 42)
    # Processing the call arguments (line 42)
    # Getting the type of 'name' (line 42)
    name_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'name', False)
    # Processing the call keyword arguments (line 42)
    kwargs_25 = {}
    # Getting the type of 'os' (line 42)
    os_21 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 42)
    path_22 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), os_21, 'path')
    # Obtaining the member 'normpath' of a type (line 42)
    normpath_23 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 11), path_22, 'normpath')
    # Calling normpath(args, kwargs) (line 42)
    normpath_call_result_26 = invoke(stypy.reporting.localization.Localization(__file__, 42, 11), normpath_23, *[name_24], **kwargs_25)
    
    # Assigning a type to the variable 'name' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'name', normpath_call_result_26)
    
    # Assigning a List to a Name (line 43):
    
    # Assigning a List to a Name (line 43):
    
    # Obtaining an instance of the builtin type 'list' (line 43)
    list_27 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'list')
    # Adding type elements to the builtin type 'list' instance (line 43)
    
    # Assigning a type to the variable 'created_dirs' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'created_dirs', list_27)
    
    
    # Evaluating a boolean operation
    
    # Call to isdir(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'name' (line 44)
    name_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 21), 'name', False)
    # Processing the call keyword arguments (line 44)
    kwargs_32 = {}
    # Getting the type of 'os' (line 44)
    os_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 44)
    path_29 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 7), os_28, 'path')
    # Obtaining the member 'isdir' of a type (line 44)
    isdir_30 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 7), path_29, 'isdir')
    # Calling isdir(args, kwargs) (line 44)
    isdir_call_result_33 = invoke(stypy.reporting.localization.Localization(__file__, 44, 7), isdir_30, *[name_31], **kwargs_32)
    
    
    # Getting the type of 'name' (line 44)
    name_34 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 30), 'name')
    str_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 38), 'str', '')
    # Applying the binary operator '==' (line 44)
    result_eq_36 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 30), '==', name_34, str_35)
    
    # Applying the binary operator 'or' (line 44)
    result_or_keyword_37 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 7), 'or', isdir_call_result_33, result_eq_36)
    
    # Testing the type of an if condition (line 44)
    if_condition_38 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 44, 4), result_or_keyword_37)
    # Assigning a type to the variable 'if_condition_38' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'if_condition_38', if_condition_38)
    # SSA begins for if statement (line 44)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'created_dirs' (line 45)
    created_dirs_39 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'created_dirs')
    # Assigning a type to the variable 'stypy_return_type' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'stypy_return_type', created_dirs_39)
    # SSA join for if statement (line 44)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to get(...): (line 46)
    # Processing the call arguments (line 46)
    
    # Call to abspath(...): (line 46)
    # Processing the call arguments (line 46)
    # Getting the type of 'name' (line 46)
    name_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 41), 'name', False)
    # Processing the call keyword arguments (line 46)
    kwargs_46 = {}
    # Getting the type of 'os' (line 46)
    os_42 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 25), 'os', False)
    # Obtaining the member 'path' of a type (line 46)
    path_43 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), os_42, 'path')
    # Obtaining the member 'abspath' of a type (line 46)
    abspath_44 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 25), path_43, 'abspath')
    # Calling abspath(args, kwargs) (line 46)
    abspath_call_result_47 = invoke(stypy.reporting.localization.Localization(__file__, 46, 25), abspath_44, *[name_45], **kwargs_46)
    
    # Processing the call keyword arguments (line 46)
    kwargs_48 = {}
    # Getting the type of '_path_created' (line 46)
    _path_created_40 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 7), '_path_created', False)
    # Obtaining the member 'get' of a type (line 46)
    get_41 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 7), _path_created_40, 'get')
    # Calling get(args, kwargs) (line 46)
    get_call_result_49 = invoke(stypy.reporting.localization.Localization(__file__, 46, 7), get_41, *[abspath_call_result_47], **kwargs_48)
    
    # Testing the type of an if condition (line 46)
    if_condition_50 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 46, 4), get_call_result_49)
    # Assigning a type to the variable 'if_condition_50' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'if_condition_50', if_condition_50)
    # SSA begins for if statement (line 46)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'created_dirs' (line 47)
    created_dirs_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'created_dirs')
    # Assigning a type to the variable 'stypy_return_type' (line 47)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', created_dirs_51)
    # SSA join for if statement (line 46)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Tuple (line 49):
    
    # Assigning a Subscript to a Name (line 49):
    
    # Obtaining the type of the subscript
    int_52 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'int')
    
    # Call to split(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'name' (line 49)
    name_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'name', False)
    # Processing the call keyword arguments (line 49)
    kwargs_57 = {}
    # Getting the type of 'os' (line 49)
    os_53 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 49)
    path_54 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), os_53, 'path')
    # Obtaining the member 'split' of a type (line 49)
    split_55 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), path_54, 'split')
    # Calling split(args, kwargs) (line 49)
    split_call_result_58 = invoke(stypy.reporting.localization.Localization(__file__, 49, 19), split_55, *[name_56], **kwargs_57)
    
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___59 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), split_call_result_58, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_60 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), getitem___59, int_52)
    
    # Assigning a type to the variable 'tuple_var_assignment_1' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'tuple_var_assignment_1', subscript_call_result_60)
    
    # Assigning a Subscript to a Name (line 49):
    
    # Obtaining the type of the subscript
    int_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 4), 'int')
    
    # Call to split(...): (line 49)
    # Processing the call arguments (line 49)
    # Getting the type of 'name' (line 49)
    name_65 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 33), 'name', False)
    # Processing the call keyword arguments (line 49)
    kwargs_66 = {}
    # Getting the type of 'os' (line 49)
    os_62 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 49)
    path_63 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), os_62, 'path')
    # Obtaining the member 'split' of a type (line 49)
    split_64 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 19), path_63, 'split')
    # Calling split(args, kwargs) (line 49)
    split_call_result_67 = invoke(stypy.reporting.localization.Localization(__file__, 49, 19), split_64, *[name_65], **kwargs_66)
    
    # Obtaining the member '__getitem__' of a type (line 49)
    getitem___68 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 4), split_call_result_67, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 49)
    subscript_call_result_69 = invoke(stypy.reporting.localization.Localization(__file__, 49, 4), getitem___68, int_61)
    
    # Assigning a type to the variable 'tuple_var_assignment_2' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'tuple_var_assignment_2', subscript_call_result_69)
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'tuple_var_assignment_1' (line 49)
    tuple_var_assignment_1_70 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'tuple_var_assignment_1')
    # Assigning a type to the variable 'head' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 5), 'head', tuple_var_assignment_1_70)
    
    # Assigning a Name to a Name (line 49):
    # Getting the type of 'tuple_var_assignment_2' (line 49)
    tuple_var_assignment_2_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'tuple_var_assignment_2')
    # Assigning a type to the variable 'tail' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'tail', tuple_var_assignment_2_71)
    
    # Assigning a List to a Name (line 50):
    
    # Assigning a List to a Name (line 50):
    
    # Obtaining an instance of the builtin type 'list' (line 50)
    list_72 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 12), 'list')
    # Adding type elements to the builtin type 'list' instance (line 50)
    # Adding element type (line 50)
    # Getting the type of 'tail' (line 50)
    tail_73 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 13), 'tail')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 12), list_72, tail_73)
    
    # Assigning a type to the variable 'tails' (line 50)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'tails', list_72)
    
    
    # Evaluating a boolean operation
    # Getting the type of 'head' (line 52)
    head_74 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 10), 'head')
    # Getting the type of 'tail' (line 52)
    tail_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'tail')
    # Applying the binary operator 'and' (line 52)
    result_and_keyword_76 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 10), 'and', head_74, tail_75)
    
    
    # Call to isdir(...): (line 52)
    # Processing the call arguments (line 52)
    # Getting the type of 'head' (line 52)
    head_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 46), 'head', False)
    # Processing the call keyword arguments (line 52)
    kwargs_81 = {}
    # Getting the type of 'os' (line 52)
    os_77 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 32), 'os', False)
    # Obtaining the member 'path' of a type (line 52)
    path_78 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 32), os_77, 'path')
    # Obtaining the member 'isdir' of a type (line 52)
    isdir_79 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 32), path_78, 'isdir')
    # Calling isdir(args, kwargs) (line 52)
    isdir_call_result_82 = invoke(stypy.reporting.localization.Localization(__file__, 52, 32), isdir_79, *[head_80], **kwargs_81)
    
    # Applying the 'not' unary operator (line 52)
    result_not__83 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 28), 'not', isdir_call_result_82)
    
    # Applying the binary operator 'and' (line 52)
    result_and_keyword_84 = python_operator(stypy.reporting.localization.Localization(__file__, 52, 10), 'and', result_and_keyword_76, result_not__83)
    
    # Testing the type of an if condition (line 52)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 52, 4), result_and_keyword_84)
    # SSA begins for while statement (line 52)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Tuple (line 53):
    
    # Assigning a Subscript to a Name (line 53):
    
    # Obtaining the type of the subscript
    int_85 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'int')
    
    # Call to split(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'head' (line 53)
    head_89 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'head', False)
    # Processing the call keyword arguments (line 53)
    kwargs_90 = {}
    # Getting the type of 'os' (line 53)
    os_86 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 53)
    path_87 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 23), os_86, 'path')
    # Obtaining the member 'split' of a type (line 53)
    split_88 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 23), path_87, 'split')
    # Calling split(args, kwargs) (line 53)
    split_call_result_91 = invoke(stypy.reporting.localization.Localization(__file__, 53, 23), split_88, *[head_89], **kwargs_90)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___92 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), split_call_result_91, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_93 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), getitem___92, int_85)
    
    # Assigning a type to the variable 'tuple_var_assignment_3' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'tuple_var_assignment_3', subscript_call_result_93)
    
    # Assigning a Subscript to a Name (line 53):
    
    # Obtaining the type of the subscript
    int_94 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 8), 'int')
    
    # Call to split(...): (line 53)
    # Processing the call arguments (line 53)
    # Getting the type of 'head' (line 53)
    head_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 37), 'head', False)
    # Processing the call keyword arguments (line 53)
    kwargs_99 = {}
    # Getting the type of 'os' (line 53)
    os_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 23), 'os', False)
    # Obtaining the member 'path' of a type (line 53)
    path_96 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 23), os_95, 'path')
    # Obtaining the member 'split' of a type (line 53)
    split_97 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 23), path_96, 'split')
    # Calling split(args, kwargs) (line 53)
    split_call_result_100 = invoke(stypy.reporting.localization.Localization(__file__, 53, 23), split_97, *[head_98], **kwargs_99)
    
    # Obtaining the member '__getitem__' of a type (line 53)
    getitem___101 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), split_call_result_100, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 53)
    subscript_call_result_102 = invoke(stypy.reporting.localization.Localization(__file__, 53, 8), getitem___101, int_94)
    
    # Assigning a type to the variable 'tuple_var_assignment_4' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'tuple_var_assignment_4', subscript_call_result_102)
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'tuple_var_assignment_3' (line 53)
    tuple_var_assignment_3_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'tuple_var_assignment_3')
    # Assigning a type to the variable 'head' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 9), 'head', tuple_var_assignment_3_103)
    
    # Assigning a Name to a Name (line 53):
    # Getting the type of 'tuple_var_assignment_4' (line 53)
    tuple_var_assignment_4_104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'tuple_var_assignment_4')
    # Assigning a type to the variable 'tail' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'tail', tuple_var_assignment_4_104)
    
    # Call to insert(...): (line 54)
    # Processing the call arguments (line 54)
    int_107 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 21), 'int')
    # Getting the type of 'tail' (line 54)
    tail_108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 24), 'tail', False)
    # Processing the call keyword arguments (line 54)
    kwargs_109 = {}
    # Getting the type of 'tails' (line 54)
    tails_105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'tails', False)
    # Obtaining the member 'insert' of a type (line 54)
    insert_106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), tails_105, 'insert')
    # Calling insert(args, kwargs) (line 54)
    insert_call_result_110 = invoke(stypy.reporting.localization.Localization(__file__, 54, 8), insert_106, *[int_107, tail_108], **kwargs_109)
    
    # SSA join for while statement (line 52)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'tails' (line 59)
    tails_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 13), 'tails')
    # Testing the type of a for loop iterable (line 59)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 59, 4), tails_111)
    # Getting the type of the for loop variable (line 59)
    for_loop_var_112 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 59, 4), tails_111)
    # Assigning a type to the variable 'd' (line 59)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'd', for_loop_var_112)
    # SSA begins for a for statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 61):
    
    # Assigning a Call to a Name (line 61):
    
    # Call to join(...): (line 61)
    # Processing the call arguments (line 61)
    # Getting the type of 'head' (line 61)
    head_116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 28), 'head', False)
    # Getting the type of 'd' (line 61)
    d_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 34), 'd', False)
    # Processing the call keyword arguments (line 61)
    kwargs_118 = {}
    # Getting the type of 'os' (line 61)
    os_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 15), 'os', False)
    # Obtaining the member 'path' of a type (line 61)
    path_114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), os_113, 'path')
    # Obtaining the member 'join' of a type (line 61)
    join_115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 15), path_114, 'join')
    # Calling join(args, kwargs) (line 61)
    join_call_result_119 = invoke(stypy.reporting.localization.Localization(__file__, 61, 15), join_115, *[head_116, d_117], **kwargs_118)
    
    # Assigning a type to the variable 'head' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'head', join_call_result_119)
    
    # Assigning a Call to a Name (line 62):
    
    # Assigning a Call to a Name (line 62):
    
    # Call to abspath(...): (line 62)
    # Processing the call arguments (line 62)
    # Getting the type of 'head' (line 62)
    head_123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 35), 'head', False)
    # Processing the call keyword arguments (line 62)
    kwargs_124 = {}
    # Getting the type of 'os' (line 62)
    os_120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 62)
    path_121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), os_120, 'path')
    # Obtaining the member 'abspath' of a type (line 62)
    abspath_122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 19), path_121, 'abspath')
    # Calling abspath(args, kwargs) (line 62)
    abspath_call_result_125 = invoke(stypy.reporting.localization.Localization(__file__, 62, 19), abspath_122, *[head_123], **kwargs_124)
    
    # Assigning a type to the variable 'abs_head' (line 62)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'abs_head', abspath_call_result_125)
    
    
    # Call to get(...): (line 64)
    # Processing the call arguments (line 64)
    # Getting the type of 'abs_head' (line 64)
    abs_head_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 29), 'abs_head', False)
    # Processing the call keyword arguments (line 64)
    kwargs_129 = {}
    # Getting the type of '_path_created' (line 64)
    _path_created_126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), '_path_created', False)
    # Obtaining the member 'get' of a type (line 64)
    get_127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), _path_created_126, 'get')
    # Calling get(args, kwargs) (line 64)
    get_call_result_130 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), get_127, *[abs_head_128], **kwargs_129)
    
    # Testing the type of an if condition (line 64)
    if_condition_131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), get_call_result_130)
    # Assigning a type to the variable 'if_condition_131' (line 64)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_131', if_condition_131)
    # SSA begins for if statement (line 64)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 64)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 67)
    verbose_132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 11), 'verbose')
    int_133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 22), 'int')
    # Applying the binary operator '>=' (line 67)
    result_ge_134 = python_operator(stypy.reporting.localization.Localization(__file__, 67, 11), '>=', verbose_132, int_133)
    
    # Testing the type of an if condition (line 67)
    if_condition_135 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 67, 8), result_ge_134)
    # Assigning a type to the variable 'if_condition_135' (line 67)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'if_condition_135', if_condition_135)
    # SSA begins for if statement (line 67)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to info(...): (line 68)
    # Processing the call arguments (line 68)
    str_138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 21), 'str', 'creating %s')
    # Getting the type of 'head' (line 68)
    head_139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'head', False)
    # Processing the call keyword arguments (line 68)
    kwargs_140 = {}
    # Getting the type of 'log' (line 68)
    log_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'log', False)
    # Obtaining the member 'info' of a type (line 68)
    info_137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 12), log_136, 'info')
    # Calling info(args, kwargs) (line 68)
    info_call_result_141 = invoke(stypy.reporting.localization.Localization(__file__, 68, 12), info_137, *[str_138, head_139], **kwargs_140)
    
    # SSA join for if statement (line 67)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'dry_run' (line 70)
    dry_run_142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 15), 'dry_run')
    # Applying the 'not' unary operator (line 70)
    result_not__143 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 11), 'not', dry_run_142)
    
    # Testing the type of an if condition (line 70)
    if_condition_144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 8), result_not__143)
    # Assigning a type to the variable 'if_condition_144' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 8), 'if_condition_144', if_condition_144)
    # SSA begins for if statement (line 70)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 71)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to mkdir(...): (line 72)
    # Processing the call arguments (line 72)
    # Getting the type of 'head' (line 72)
    head_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 25), 'head', False)
    # Getting the type of 'mode' (line 72)
    mode_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 31), 'mode', False)
    # Processing the call keyword arguments (line 72)
    kwargs_149 = {}
    # Getting the type of 'os' (line 72)
    os_145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'os', False)
    # Obtaining the member 'mkdir' of a type (line 72)
    mkdir_146 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 16), os_145, 'mkdir')
    # Calling mkdir(args, kwargs) (line 72)
    mkdir_call_result_150 = invoke(stypy.reporting.localization.Localization(__file__, 72, 16), mkdir_146, *[head_147, mode_148], **kwargs_149)
    
    # SSA branch for the except part of a try statement (line 71)
    # SSA branch for the except 'OSError' branch of a try statement (line 71)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'OSError' (line 73)
    OSError_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 19), 'OSError')
    # Assigning a type to the variable 'exc' (line 73)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'exc', OSError_151)
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'exc' (line 74)
    exc_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 24), 'exc')
    # Obtaining the member 'errno' of a type (line 74)
    errno_153 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 24), exc_152, 'errno')
    # Getting the type of 'errno' (line 74)
    errno_154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 37), 'errno')
    # Obtaining the member 'EEXIST' of a type (line 74)
    EEXIST_155 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 37), errno_154, 'EEXIST')
    # Applying the binary operator '==' (line 74)
    result_eq_156 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 24), '==', errno_153, EEXIST_155)
    
    
    # Call to isdir(...): (line 74)
    # Processing the call arguments (line 74)
    # Getting the type of 'head' (line 74)
    head_160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 68), 'head', False)
    # Processing the call keyword arguments (line 74)
    kwargs_161 = {}
    # Getting the type of 'os' (line 74)
    os_157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 54), 'os', False)
    # Obtaining the member 'path' of a type (line 74)
    path_158 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 54), os_157, 'path')
    # Obtaining the member 'isdir' of a type (line 74)
    isdir_159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 54), path_158, 'isdir')
    # Calling isdir(args, kwargs) (line 74)
    isdir_call_result_162 = invoke(stypy.reporting.localization.Localization(__file__, 74, 54), isdir_159, *[head_160], **kwargs_161)
    
    # Applying the binary operator 'and' (line 74)
    result_and_keyword_163 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 24), 'and', result_eq_156, isdir_call_result_162)
    
    # Applying the 'not' unary operator (line 74)
    result_not__164 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 19), 'not', result_and_keyword_163)
    
    # Testing the type of an if condition (line 74)
    if_condition_165 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 16), result_not__164)
    # Assigning a type to the variable 'if_condition_165' (line 74)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'if_condition_165', if_condition_165)
    # SSA begins for if statement (line 74)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to DistutilsFileError(...): (line 75)
    # Processing the call arguments (line 75)
    str_167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 26), 'str', "could not create '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 76)
    tuple_168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 57), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 76)
    # Adding element type (line 76)
    # Getting the type of 'head' (line 76)
    head_169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 57), 'head', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 57), tuple_168, head_169)
    # Adding element type (line 76)
    
    # Obtaining the type of the subscript
    int_170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 72), 'int')
    # Getting the type of 'exc' (line 76)
    exc_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 63), 'exc', False)
    # Obtaining the member 'args' of a type (line 76)
    args_172 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 63), exc_171, 'args')
    # Obtaining the member '__getitem__' of a type (line 76)
    getitem___173 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 63), args_172, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 76)
    subscript_call_result_174 = invoke(stypy.reporting.localization.Localization(__file__, 76, 63), getitem___173, int_170)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 57), tuple_168, subscript_call_result_174)
    
    # Applying the binary operator '%' (line 76)
    result_mod_175 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 26), '%', str_167, tuple_168)
    
    # Processing the call keyword arguments (line 75)
    kwargs_176 = {}
    # Getting the type of 'DistutilsFileError' (line 75)
    DistutilsFileError_166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 26), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 75)
    DistutilsFileError_call_result_177 = invoke(stypy.reporting.localization.Localization(__file__, 75, 26), DistutilsFileError_166, *[result_mod_175], **kwargs_176)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 75, 20), DistutilsFileError_call_result_177, 'raise parameter', BaseException)
    # SSA join for if statement (line 74)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 71)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 77)
    # Processing the call arguments (line 77)
    # Getting the type of 'head' (line 77)
    head_180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 32), 'head', False)
    # Processing the call keyword arguments (line 77)
    kwargs_181 = {}
    # Getting the type of 'created_dirs' (line 77)
    created_dirs_178 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'created_dirs', False)
    # Obtaining the member 'append' of a type (line 77)
    append_179 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 12), created_dirs_178, 'append')
    # Calling append(args, kwargs) (line 77)
    append_call_result_182 = invoke(stypy.reporting.localization.Localization(__file__, 77, 12), append_179, *[head_180], **kwargs_181)
    
    # SSA join for if statement (line 70)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Subscript (line 79):
    
    # Assigning a Num to a Subscript (line 79):
    int_183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 34), 'int')
    # Getting the type of '_path_created' (line 79)
    _path_created_184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), '_path_created')
    # Getting the type of 'abs_head' (line 79)
    abs_head_185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 22), 'abs_head')
    # Storing an element on a container (line 79)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 8), _path_created_184, (abs_head_185, int_183))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'created_dirs' (line 80)
    created_dirs_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 11), 'created_dirs')
    # Assigning a type to the variable 'stypy_return_type' (line 80)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type', created_dirs_186)
    
    # ################# End of 'mkpath(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'mkpath' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_187)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'mkpath'
    return stypy_return_type_187

# Assigning a type to the variable 'mkpath' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 0), 'mkpath', mkpath)

@norecursion
def create_tree(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_188 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 38), 'int')
    int_189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 52), 'int')
    int_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 82, 63), 'int')
    defaults = [int_188, int_189, int_190]
    # Create a new context for function 'create_tree'
    module_type_store = module_type_store.open_function_context('create_tree', 82, 0, False)
    
    # Passed parameters checking function
    create_tree.stypy_localization = localization
    create_tree.stypy_type_of_self = None
    create_tree.stypy_type_store = module_type_store
    create_tree.stypy_function_name = 'create_tree'
    create_tree.stypy_param_names_list = ['base_dir', 'files', 'mode', 'verbose', 'dry_run']
    create_tree.stypy_varargs_param_name = None
    create_tree.stypy_kwargs_param_name = None
    create_tree.stypy_call_defaults = defaults
    create_tree.stypy_call_varargs = varargs
    create_tree.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'create_tree', ['base_dir', 'files', 'mode', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'create_tree', localization, ['base_dir', 'files', 'mode', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'create_tree(...)' code ##################

    str_191 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 91, (-1)), 'str', "Create all the empty directories under 'base_dir' needed to put 'files'\n    there.\n\n    'base_dir' is just the name of a directory which doesn't necessarily\n    exist yet; 'files' is a list of filenames to be interpreted relative to\n    'base_dir'.  'base_dir' + the directory portion of every file in 'files'\n    will be created if it doesn't already exist.  'mode', 'verbose' and\n    'dry_run' flags are as for 'mkpath()'.\n    ")
    
    # Assigning a Dict to a Name (line 93):
    
    # Assigning a Dict to a Name (line 93):
    
    # Obtaining an instance of the builtin type 'dict' (line 93)
    dict_192 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 15), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 93)
    
    # Assigning a type to the variable 'need_dir' (line 93)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'need_dir', dict_192)
    
    # Getting the type of 'files' (line 94)
    files_193 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 16), 'files')
    # Testing the type of a for loop iterable (line 94)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 94, 4), files_193)
    # Getting the type of the for loop variable (line 94)
    for_loop_var_194 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 94, 4), files_193)
    # Assigning a type to the variable 'file' (line 94)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'file', for_loop_var_194)
    # SSA begins for a for statement (line 94)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Num to a Subscript (line 95):
    
    # Assigning a Num to a Subscript (line 95):
    int_195 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 95, 66), 'int')
    # Getting the type of 'need_dir' (line 95)
    need_dir_196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'need_dir')
    
    # Call to join(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'base_dir' (line 95)
    base_dir_200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 30), 'base_dir', False)
    
    # Call to dirname(...): (line 95)
    # Processing the call arguments (line 95)
    # Getting the type of 'file' (line 95)
    file_204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 56), 'file', False)
    # Processing the call keyword arguments (line 95)
    kwargs_205 = {}
    # Getting the type of 'os' (line 95)
    os_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 40), 'os', False)
    # Obtaining the member 'path' of a type (line 95)
    path_202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 40), os_201, 'path')
    # Obtaining the member 'dirname' of a type (line 95)
    dirname_203 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 40), path_202, 'dirname')
    # Calling dirname(args, kwargs) (line 95)
    dirname_call_result_206 = invoke(stypy.reporting.localization.Localization(__file__, 95, 40), dirname_203, *[file_204], **kwargs_205)
    
    # Processing the call keyword arguments (line 95)
    kwargs_207 = {}
    # Getting the type of 'os' (line 95)
    os_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 95)
    path_198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 17), os_197, 'path')
    # Obtaining the member 'join' of a type (line 95)
    join_199 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 17), path_198, 'join')
    # Calling join(args, kwargs) (line 95)
    join_call_result_208 = invoke(stypy.reporting.localization.Localization(__file__, 95, 17), join_199, *[base_dir_200, dirname_call_result_206], **kwargs_207)
    
    # Storing an element on a container (line 95)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 95, 8), need_dir_196, (join_call_result_208, int_195))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 96):
    
    # Assigning a Call to a Name (line 96):
    
    # Call to keys(...): (line 96)
    # Processing the call keyword arguments (line 96)
    kwargs_211 = {}
    # Getting the type of 'need_dir' (line 96)
    need_dir_209 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 16), 'need_dir', False)
    # Obtaining the member 'keys' of a type (line 96)
    keys_210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 16), need_dir_209, 'keys')
    # Calling keys(args, kwargs) (line 96)
    keys_call_result_212 = invoke(stypy.reporting.localization.Localization(__file__, 96, 16), keys_210, *[], **kwargs_211)
    
    # Assigning a type to the variable 'need_dirs' (line 96)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'need_dirs', keys_call_result_212)
    
    # Call to sort(...): (line 97)
    # Processing the call keyword arguments (line 97)
    kwargs_215 = {}
    # Getting the type of 'need_dirs' (line 97)
    need_dirs_213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'need_dirs', False)
    # Obtaining the member 'sort' of a type (line 97)
    sort_214 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 4), need_dirs_213, 'sort')
    # Calling sort(args, kwargs) (line 97)
    sort_call_result_216 = invoke(stypy.reporting.localization.Localization(__file__, 97, 4), sort_214, *[], **kwargs_215)
    
    
    # Getting the type of 'need_dirs' (line 100)
    need_dirs_217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 15), 'need_dirs')
    # Testing the type of a for loop iterable (line 100)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 100, 4), need_dirs_217)
    # Getting the type of the for loop variable (line 100)
    for_loop_var_218 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 100, 4), need_dirs_217)
    # Assigning a type to the variable 'dir' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'dir', for_loop_var_218)
    # SSA begins for a for statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to mkpath(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'dir' (line 101)
    dir_220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 15), 'dir', False)
    # Getting the type of 'mode' (line 101)
    mode_221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 20), 'mode', False)
    # Processing the call keyword arguments (line 101)
    # Getting the type of 'verbose' (line 101)
    verbose_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 34), 'verbose', False)
    keyword_223 = verbose_222
    # Getting the type of 'dry_run' (line 101)
    dry_run_224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 51), 'dry_run', False)
    keyword_225 = dry_run_224
    kwargs_226 = {'verbose': keyword_223, 'dry_run': keyword_225}
    # Getting the type of 'mkpath' (line 101)
    mkpath_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'mkpath', False)
    # Calling mkpath(args, kwargs) (line 101)
    mkpath_call_result_227 = invoke(stypy.reporting.localization.Localization(__file__, 101, 8), mkpath_219, *[dir_220, mode_221], **kwargs_226)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'create_tree(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'create_tree' in the type store
    # Getting the type of 'stypy_return_type' (line 82)
    stypy_return_type_228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_228)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'create_tree'
    return stypy_return_type_228

# Assigning a type to the variable 'create_tree' (line 82)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 0), 'create_tree', create_tree)

@norecursion
def copy_tree(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_229 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 38), 'int')
    int_230 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 56), 'int')
    int_231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 32), 'int')
    int_232 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 42), 'int')
    int_233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 53), 'int')
    int_234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 64), 'int')
    defaults = [int_229, int_230, int_231, int_232, int_233, int_234]
    # Create a new context for function 'copy_tree'
    module_type_store = module_type_store.open_function_context('copy_tree', 103, 0, False)
    
    # Passed parameters checking function
    copy_tree.stypy_localization = localization
    copy_tree.stypy_type_of_self = None
    copy_tree.stypy_type_store = module_type_store
    copy_tree.stypy_function_name = 'copy_tree'
    copy_tree.stypy_param_names_list = ['src', 'dst', 'preserve_mode', 'preserve_times', 'preserve_symlinks', 'update', 'verbose', 'dry_run']
    copy_tree.stypy_varargs_param_name = None
    copy_tree.stypy_kwargs_param_name = None
    copy_tree.stypy_call_defaults = defaults
    copy_tree.stypy_call_varargs = varargs
    copy_tree.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'copy_tree', ['src', 'dst', 'preserve_mode', 'preserve_times', 'preserve_symlinks', 'update', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'copy_tree', localization, ['src', 'dst', 'preserve_mode', 'preserve_times', 'preserve_symlinks', 'update', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'copy_tree(...)' code ##################

    str_235 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, (-1)), 'str', "Copy an entire directory tree 'src' to a new location 'dst'.\n\n    Both 'src' and 'dst' must be directory names.  If 'src' is not a\n    directory, raise DistutilsFileError.  If 'dst' does not exist, it is\n    created with 'mkpath()'.  The end result of the copy is that every\n    file in 'src' is copied to 'dst', and directories under 'src' are\n    recursively copied to 'dst'.  Return the list of files that were\n    copied or might have been copied, using their output name.  The\n    return value is unaffected by 'update' or 'dry_run': it is simply\n    the list of all files under 'src', with the names changed to be\n    under 'dst'.\n\n    'preserve_mode' and 'preserve_times' are the same as for\n    'copy_file'; note that they only apply to regular files, not to\n    directories.  If 'preserve_symlinks' is true, symlinks will be\n    copied as symlinks (on platforms that support them!); otherwise\n    (the default), the destination of the symlink will be copied.\n    'update' and 'verbose' are the same as for 'copy_file'.\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 124, 4))
    
    # 'from distutils.file_util import copy_file' statement (line 124)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_236 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 124, 4), 'distutils.file_util')

    if (type(import_236) is not StypyTypeError):

        if (import_236 != 'pyd_module'):
            __import__(import_236)
            sys_modules_237 = sys.modules[import_236]
            import_from_module(stypy.reporting.localization.Localization(__file__, 124, 4), 'distutils.file_util', sys_modules_237.module_type_store, module_type_store, ['copy_file'])
            nest_module(stypy.reporting.localization.Localization(__file__, 124, 4), __file__, sys_modules_237, sys_modules_237.module_type_store, module_type_store)
        else:
            from distutils.file_util import copy_file

            import_from_module(stypy.reporting.localization.Localization(__file__, 124, 4), 'distutils.file_util', None, module_type_store, ['copy_file'], [copy_file])

    else:
        # Assigning a type to the variable 'distutils.file_util' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'distutils.file_util', import_236)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'dry_run' (line 126)
    dry_run_238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 11), 'dry_run')
    # Applying the 'not' unary operator (line 126)
    result_not__239 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 7), 'not', dry_run_238)
    
    
    
    # Call to isdir(...): (line 126)
    # Processing the call arguments (line 126)
    # Getting the type of 'src' (line 126)
    src_243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 41), 'src', False)
    # Processing the call keyword arguments (line 126)
    kwargs_244 = {}
    # Getting the type of 'os' (line 126)
    os_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 27), 'os', False)
    # Obtaining the member 'path' of a type (line 126)
    path_241 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 27), os_240, 'path')
    # Obtaining the member 'isdir' of a type (line 126)
    isdir_242 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 27), path_241, 'isdir')
    # Calling isdir(args, kwargs) (line 126)
    isdir_call_result_245 = invoke(stypy.reporting.localization.Localization(__file__, 126, 27), isdir_242, *[src_243], **kwargs_244)
    
    # Applying the 'not' unary operator (line 126)
    result_not__246 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 23), 'not', isdir_call_result_245)
    
    # Applying the binary operator 'and' (line 126)
    result_and_keyword_247 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 7), 'and', result_not__239, result_not__246)
    
    # Testing the type of an if condition (line 126)
    if_condition_248 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 4), result_and_keyword_247)
    # Assigning a type to the variable 'if_condition_248' (line 126)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'if_condition_248', if_condition_248)
    # SSA begins for if statement (line 126)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'DistutilsFileError' (line 127)
    DistutilsFileError_249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 14), 'DistutilsFileError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 127, 8), DistutilsFileError_249, 'raise parameter', BaseException)
    # SSA join for if statement (line 126)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 130):
    
    # Assigning a Call to a Name (line 130):
    
    # Call to listdir(...): (line 130)
    # Processing the call arguments (line 130)
    # Getting the type of 'src' (line 130)
    src_252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 27), 'src', False)
    # Processing the call keyword arguments (line 130)
    kwargs_253 = {}
    # Getting the type of 'os' (line 130)
    os_250 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'os', False)
    # Obtaining the member 'listdir' of a type (line 130)
    listdir_251 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 16), os_250, 'listdir')
    # Calling listdir(args, kwargs) (line 130)
    listdir_call_result_254 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), listdir_251, *[src_252], **kwargs_253)
    
    # Assigning a type to the variable 'names' (line 130)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'names', listdir_call_result_254)
    # SSA branch for the except part of a try statement (line 129)
    # SSA branch for the except 'Attribute' branch of a try statement (line 129)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'os' (line 131)
    os_255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 11), 'os')
    # Obtaining the member 'error' of a type (line 131)
    error_256 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 11), os_255, 'error')
    # Assigning a type to the variable 'errno' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'errno', error_256)
    # Assigning a type to the variable 'errstr' (line 131)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 4), 'errstr', error_256)
    
    # Getting the type of 'dry_run' (line 132)
    dry_run_257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 11), 'dry_run')
    # Testing the type of an if condition (line 132)
    if_condition_258 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 132, 8), dry_run_257)
    # Assigning a type to the variable 'if_condition_258' (line 132)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'if_condition_258', if_condition_258)
    # SSA begins for if statement (line 132)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 133):
    
    # Assigning a List to a Name (line 133):
    
    # Obtaining an instance of the builtin type 'list' (line 133)
    list_259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 133, 20), 'list')
    # Adding type elements to the builtin type 'list' instance (line 133)
    
    # Assigning a type to the variable 'names' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'names', list_259)
    # SSA branch for the else part of an if statement (line 132)
    module_type_store.open_ssa_branch('else')
    # Getting the type of 'DistutilsFileError' (line 135)
    DistutilsFileError_260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 18), 'DistutilsFileError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 12), DistutilsFileError_260, 'raise parameter', BaseException)
    # SSA join for if statement (line 132)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'dry_run' (line 138)
    dry_run_261 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 11), 'dry_run')
    # Applying the 'not' unary operator (line 138)
    result_not__262 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 7), 'not', dry_run_261)
    
    # Testing the type of an if condition (line 138)
    if_condition_263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 4), result_not__262)
    # Assigning a type to the variable 'if_condition_263' (line 138)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 4), 'if_condition_263', if_condition_263)
    # SSA begins for if statement (line 138)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to mkpath(...): (line 139)
    # Processing the call arguments (line 139)
    # Getting the type of 'dst' (line 139)
    dst_265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 15), 'dst', False)
    # Processing the call keyword arguments (line 139)
    # Getting the type of 'verbose' (line 139)
    verbose_266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 28), 'verbose', False)
    keyword_267 = verbose_266
    kwargs_268 = {'verbose': keyword_267}
    # Getting the type of 'mkpath' (line 139)
    mkpath_264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 8), 'mkpath', False)
    # Calling mkpath(args, kwargs) (line 139)
    mkpath_call_result_269 = invoke(stypy.reporting.localization.Localization(__file__, 139, 8), mkpath_264, *[dst_265], **kwargs_268)
    
    # SSA join for if statement (line 138)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 141):
    
    # Assigning a List to a Name (line 141):
    
    # Obtaining an instance of the builtin type 'list' (line 141)
    list_270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 141, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 141)
    
    # Assigning a type to the variable 'outputs' (line 141)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 4), 'outputs', list_270)
    
    # Getting the type of 'names' (line 143)
    names_271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 13), 'names')
    # Testing the type of a for loop iterable (line 143)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 143, 4), names_271)
    # Getting the type of the for loop variable (line 143)
    for_loop_var_272 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 143, 4), names_271)
    # Assigning a type to the variable 'n' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'n', for_loop_var_272)
    # SSA begins for a for statement (line 143)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 144):
    
    # Assigning a Call to a Name (line 144):
    
    # Call to join(...): (line 144)
    # Processing the call arguments (line 144)
    # Getting the type of 'src' (line 144)
    src_276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 32), 'src', False)
    # Getting the type of 'n' (line 144)
    n_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 37), 'n', False)
    # Processing the call keyword arguments (line 144)
    kwargs_278 = {}
    # Getting the type of 'os' (line 144)
    os_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 144)
    path_274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), os_273, 'path')
    # Obtaining the member 'join' of a type (line 144)
    join_275 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 19), path_274, 'join')
    # Calling join(args, kwargs) (line 144)
    join_call_result_279 = invoke(stypy.reporting.localization.Localization(__file__, 144, 19), join_275, *[src_276, n_277], **kwargs_278)
    
    # Assigning a type to the variable 'src_name' (line 144)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'src_name', join_call_result_279)
    
    # Assigning a Call to a Name (line 145):
    
    # Assigning a Call to a Name (line 145):
    
    # Call to join(...): (line 145)
    # Processing the call arguments (line 145)
    # Getting the type of 'dst' (line 145)
    dst_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 32), 'dst', False)
    # Getting the type of 'n' (line 145)
    n_284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 37), 'n', False)
    # Processing the call keyword arguments (line 145)
    kwargs_285 = {}
    # Getting the type of 'os' (line 145)
    os_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 19), 'os', False)
    # Obtaining the member 'path' of a type (line 145)
    path_281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 19), os_280, 'path')
    # Obtaining the member 'join' of a type (line 145)
    join_282 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 19), path_281, 'join')
    # Calling join(args, kwargs) (line 145)
    join_call_result_286 = invoke(stypy.reporting.localization.Localization(__file__, 145, 19), join_282, *[dst_283, n_284], **kwargs_285)
    
    # Assigning a type to the variable 'dst_name' (line 145)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'dst_name', join_call_result_286)
    
    
    # Call to startswith(...): (line 147)
    # Processing the call arguments (line 147)
    str_289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 24), 'str', '.nfs')
    # Processing the call keyword arguments (line 147)
    kwargs_290 = {}
    # Getting the type of 'n' (line 147)
    n_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 11), 'n', False)
    # Obtaining the member 'startswith' of a type (line 147)
    startswith_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 11), n_287, 'startswith')
    # Calling startswith(args, kwargs) (line 147)
    startswith_call_result_291 = invoke(stypy.reporting.localization.Localization(__file__, 147, 11), startswith_288, *[str_289], **kwargs_290)
    
    # Testing the type of an if condition (line 147)
    if_condition_292 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 147, 8), startswith_call_result_291)
    # Assigning a type to the variable 'if_condition_292' (line 147)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 8), 'if_condition_292', if_condition_292)
    # SSA begins for if statement (line 147)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 147)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'preserve_symlinks' (line 151)
    preserve_symlinks_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 11), 'preserve_symlinks')
    
    # Call to islink(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'src_name' (line 151)
    src_name_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 48), 'src_name', False)
    # Processing the call keyword arguments (line 151)
    kwargs_298 = {}
    # Getting the type of 'os' (line 151)
    os_294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 33), 'os', False)
    # Obtaining the member 'path' of a type (line 151)
    path_295 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 33), os_294, 'path')
    # Obtaining the member 'islink' of a type (line 151)
    islink_296 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 33), path_295, 'islink')
    # Calling islink(args, kwargs) (line 151)
    islink_call_result_299 = invoke(stypy.reporting.localization.Localization(__file__, 151, 33), islink_296, *[src_name_297], **kwargs_298)
    
    # Applying the binary operator 'and' (line 151)
    result_and_keyword_300 = python_operator(stypy.reporting.localization.Localization(__file__, 151, 11), 'and', preserve_symlinks_293, islink_call_result_299)
    
    # Testing the type of an if condition (line 151)
    if_condition_301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 151, 8), result_and_keyword_300)
    # Assigning a type to the variable 'if_condition_301' (line 151)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 151, 8), 'if_condition_301', if_condition_301)
    # SSA begins for if statement (line 151)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 152):
    
    # Assigning a Call to a Name (line 152):
    
    # Call to readlink(...): (line 152)
    # Processing the call arguments (line 152)
    # Getting the type of 'src_name' (line 152)
    src_name_304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 36), 'src_name', False)
    # Processing the call keyword arguments (line 152)
    kwargs_305 = {}
    # Getting the type of 'os' (line 152)
    os_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 24), 'os', False)
    # Obtaining the member 'readlink' of a type (line 152)
    readlink_303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 24), os_302, 'readlink')
    # Calling readlink(args, kwargs) (line 152)
    readlink_call_result_306 = invoke(stypy.reporting.localization.Localization(__file__, 152, 24), readlink_303, *[src_name_304], **kwargs_305)
    
    # Assigning a type to the variable 'link_dest' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'link_dest', readlink_call_result_306)
    
    
    # Getting the type of 'verbose' (line 153)
    verbose_307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'verbose')
    int_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 153, 26), 'int')
    # Applying the binary operator '>=' (line 153)
    result_ge_309 = python_operator(stypy.reporting.localization.Localization(__file__, 153, 15), '>=', verbose_307, int_308)
    
    # Testing the type of an if condition (line 153)
    if_condition_310 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 153, 12), result_ge_309)
    # Assigning a type to the variable 'if_condition_310' (line 153)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 12), 'if_condition_310', if_condition_310)
    # SSA begins for if statement (line 153)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to info(...): (line 154)
    # Processing the call arguments (line 154)
    str_313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 25), 'str', 'linking %s -> %s')
    # Getting the type of 'dst_name' (line 154)
    dst_name_314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 45), 'dst_name', False)
    # Getting the type of 'link_dest' (line 154)
    link_dest_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 55), 'link_dest', False)
    # Processing the call keyword arguments (line 154)
    kwargs_316 = {}
    # Getting the type of 'log' (line 154)
    log_311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 154, 16), 'log', False)
    # Obtaining the member 'info' of a type (line 154)
    info_312 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 154, 16), log_311, 'info')
    # Calling info(args, kwargs) (line 154)
    info_call_result_317 = invoke(stypy.reporting.localization.Localization(__file__, 154, 16), info_312, *[str_313, dst_name_314, link_dest_315], **kwargs_316)
    
    # SSA join for if statement (line 153)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'dry_run' (line 155)
    dry_run_318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 19), 'dry_run')
    # Applying the 'not' unary operator (line 155)
    result_not__319 = python_operator(stypy.reporting.localization.Localization(__file__, 155, 15), 'not', dry_run_318)
    
    # Testing the type of an if condition (line 155)
    if_condition_320 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 155, 12), result_not__319)
    # Assigning a type to the variable 'if_condition_320' (line 155)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 155, 12), 'if_condition_320', if_condition_320)
    # SSA begins for if statement (line 155)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to symlink(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'link_dest' (line 156)
    link_dest_323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 27), 'link_dest', False)
    # Getting the type of 'dst_name' (line 156)
    dst_name_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 38), 'dst_name', False)
    # Processing the call keyword arguments (line 156)
    kwargs_325 = {}
    # Getting the type of 'os' (line 156)
    os_321 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 16), 'os', False)
    # Obtaining the member 'symlink' of a type (line 156)
    symlink_322 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 156, 16), os_321, 'symlink')
    # Calling symlink(args, kwargs) (line 156)
    symlink_call_result_326 = invoke(stypy.reporting.localization.Localization(__file__, 156, 16), symlink_322, *[link_dest_323, dst_name_324], **kwargs_325)
    
    # SSA join for if statement (line 155)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 157)
    # Processing the call arguments (line 157)
    # Getting the type of 'dst_name' (line 157)
    dst_name_329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 27), 'dst_name', False)
    # Processing the call keyword arguments (line 157)
    kwargs_330 = {}
    # Getting the type of 'outputs' (line 157)
    outputs_327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'outputs', False)
    # Obtaining the member 'append' of a type (line 157)
    append_328 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 12), outputs_327, 'append')
    # Calling append(args, kwargs) (line 157)
    append_call_result_331 = invoke(stypy.reporting.localization.Localization(__file__, 157, 12), append_328, *[dst_name_329], **kwargs_330)
    
    # SSA branch for the else part of an if statement (line 151)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to isdir(...): (line 159)
    # Processing the call arguments (line 159)
    # Getting the type of 'src_name' (line 159)
    src_name_335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 27), 'src_name', False)
    # Processing the call keyword arguments (line 159)
    kwargs_336 = {}
    # Getting the type of 'os' (line 159)
    os_332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'os', False)
    # Obtaining the member 'path' of a type (line 159)
    path_333 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 13), os_332, 'path')
    # Obtaining the member 'isdir' of a type (line 159)
    isdir_334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 159, 13), path_333, 'isdir')
    # Calling isdir(args, kwargs) (line 159)
    isdir_call_result_337 = invoke(stypy.reporting.localization.Localization(__file__, 159, 13), isdir_334, *[src_name_335], **kwargs_336)
    
    # Testing the type of an if condition (line 159)
    if_condition_338 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 159, 13), isdir_call_result_337)
    # Assigning a type to the variable 'if_condition_338' (line 159)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 13), 'if_condition_338', if_condition_338)
    # SSA begins for if statement (line 159)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to extend(...): (line 160)
    # Processing the call arguments (line 160)
    
    # Call to copy_tree(...): (line 161)
    # Processing the call arguments (line 161)
    # Getting the type of 'src_name' (line 161)
    src_name_342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 26), 'src_name', False)
    # Getting the type of 'dst_name' (line 161)
    dst_name_343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 36), 'dst_name', False)
    # Getting the type of 'preserve_mode' (line 161)
    preserve_mode_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 46), 'preserve_mode', False)
    # Getting the type of 'preserve_times' (line 162)
    preserve_times_345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 26), 'preserve_times', False)
    # Getting the type of 'preserve_symlinks' (line 162)
    preserve_symlinks_346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 42), 'preserve_symlinks', False)
    # Getting the type of 'update' (line 162)
    update_347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 61), 'update', False)
    # Processing the call keyword arguments (line 161)
    # Getting the type of 'verbose' (line 163)
    verbose_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 34), 'verbose', False)
    keyword_349 = verbose_348
    # Getting the type of 'dry_run' (line 163)
    dry_run_350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 51), 'dry_run', False)
    keyword_351 = dry_run_350
    kwargs_352 = {'verbose': keyword_349, 'dry_run': keyword_351}
    # Getting the type of 'copy_tree' (line 161)
    copy_tree_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 16), 'copy_tree', False)
    # Calling copy_tree(args, kwargs) (line 161)
    copy_tree_call_result_353 = invoke(stypy.reporting.localization.Localization(__file__, 161, 16), copy_tree_341, *[src_name_342, dst_name_343, preserve_mode_344, preserve_times_345, preserve_symlinks_346, update_347], **kwargs_352)
    
    # Processing the call keyword arguments (line 160)
    kwargs_354 = {}
    # Getting the type of 'outputs' (line 160)
    outputs_339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'outputs', False)
    # Obtaining the member 'extend' of a type (line 160)
    extend_340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 12), outputs_339, 'extend')
    # Calling extend(args, kwargs) (line 160)
    extend_call_result_355 = invoke(stypy.reporting.localization.Localization(__file__, 160, 12), extend_340, *[copy_tree_call_result_353], **kwargs_354)
    
    # SSA branch for the else part of an if statement (line 159)
    module_type_store.open_ssa_branch('else')
    
    # Call to copy_file(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'src_name' (line 165)
    src_name_357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 22), 'src_name', False)
    # Getting the type of 'dst_name' (line 165)
    dst_name_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 32), 'dst_name', False)
    # Getting the type of 'preserve_mode' (line 165)
    preserve_mode_359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 42), 'preserve_mode', False)
    # Getting the type of 'preserve_times' (line 166)
    preserve_times_360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 22), 'preserve_times', False)
    # Getting the type of 'update' (line 166)
    update_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 38), 'update', False)
    # Processing the call keyword arguments (line 165)
    # Getting the type of 'verbose' (line 166)
    verbose_362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 54), 'verbose', False)
    keyword_363 = verbose_362
    # Getting the type of 'dry_run' (line 167)
    dry_run_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 30), 'dry_run', False)
    keyword_365 = dry_run_364
    kwargs_366 = {'verbose': keyword_363, 'dry_run': keyword_365}
    # Getting the type of 'copy_file' (line 165)
    copy_file_356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'copy_file', False)
    # Calling copy_file(args, kwargs) (line 165)
    copy_file_call_result_367 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), copy_file_356, *[src_name_357, dst_name_358, preserve_mode_359, preserve_times_360, update_361], **kwargs_366)
    
    
    # Call to append(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'dst_name' (line 168)
    dst_name_370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 27), 'dst_name', False)
    # Processing the call keyword arguments (line 168)
    kwargs_371 = {}
    # Getting the type of 'outputs' (line 168)
    outputs_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'outputs', False)
    # Obtaining the member 'append' of a type (line 168)
    append_369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 12), outputs_368, 'append')
    # Calling append(args, kwargs) (line 168)
    append_call_result_372 = invoke(stypy.reporting.localization.Localization(__file__, 168, 12), append_369, *[dst_name_370], **kwargs_371)
    
    # SSA join for if statement (line 159)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 151)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'outputs' (line 170)
    outputs_373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 11), 'outputs')
    # Assigning a type to the variable 'stypy_return_type' (line 170)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 4), 'stypy_return_type', outputs_373)
    
    # ################# End of 'copy_tree(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'copy_tree' in the type store
    # Getting the type of 'stypy_return_type' (line 103)
    stypy_return_type_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_374)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'copy_tree'
    return stypy_return_type_374

# Assigning a type to the variable 'copy_tree' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'copy_tree', copy_tree)

@norecursion
def _build_cmdtuple(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_build_cmdtuple'
    module_type_store = module_type_store.open_function_context('_build_cmdtuple', 172, 0, False)
    
    # Passed parameters checking function
    _build_cmdtuple.stypy_localization = localization
    _build_cmdtuple.stypy_type_of_self = None
    _build_cmdtuple.stypy_type_store = module_type_store
    _build_cmdtuple.stypy_function_name = '_build_cmdtuple'
    _build_cmdtuple.stypy_param_names_list = ['path', 'cmdtuples']
    _build_cmdtuple.stypy_varargs_param_name = None
    _build_cmdtuple.stypy_kwargs_param_name = None
    _build_cmdtuple.stypy_call_defaults = defaults
    _build_cmdtuple.stypy_call_varargs = varargs
    _build_cmdtuple.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_build_cmdtuple', ['path', 'cmdtuples'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_build_cmdtuple', localization, ['path', 'cmdtuples'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_build_cmdtuple(...)' code ##################

    str_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 173, 4), 'str', 'Helper for remove_tree().')
    
    
    # Call to listdir(...): (line 174)
    # Processing the call arguments (line 174)
    # Getting the type of 'path' (line 174)
    path_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 24), 'path', False)
    # Processing the call keyword arguments (line 174)
    kwargs_379 = {}
    # Getting the type of 'os' (line 174)
    os_376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 174, 13), 'os', False)
    # Obtaining the member 'listdir' of a type (line 174)
    listdir_377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 174, 13), os_376, 'listdir')
    # Calling listdir(args, kwargs) (line 174)
    listdir_call_result_380 = invoke(stypy.reporting.localization.Localization(__file__, 174, 13), listdir_377, *[path_378], **kwargs_379)
    
    # Testing the type of a for loop iterable (line 174)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 174, 4), listdir_call_result_380)
    # Getting the type of the for loop variable (line 174)
    for_loop_var_381 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 174, 4), listdir_call_result_380)
    # Assigning a type to the variable 'f' (line 174)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 174, 4), 'f', for_loop_var_381)
    # SSA begins for a for statement (line 174)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Call to a Name (line 175):
    
    # Assigning a Call to a Name (line 175):
    
    # Call to join(...): (line 175)
    # Processing the call arguments (line 175)
    # Getting the type of 'path' (line 175)
    path_385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 30), 'path', False)
    # Getting the type of 'f' (line 175)
    f_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 35), 'f', False)
    # Processing the call keyword arguments (line 175)
    kwargs_387 = {}
    # Getting the type of 'os' (line 175)
    os_382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 175, 17), 'os', False)
    # Obtaining the member 'path' of a type (line 175)
    path_383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 17), os_382, 'path')
    # Obtaining the member 'join' of a type (line 175)
    join_384 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 175, 17), path_383, 'join')
    # Calling join(args, kwargs) (line 175)
    join_call_result_388 = invoke(stypy.reporting.localization.Localization(__file__, 175, 17), join_384, *[path_385, f_386], **kwargs_387)
    
    # Assigning a type to the variable 'real_f' (line 175)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 8), 'real_f', join_call_result_388)
    
    
    # Evaluating a boolean operation
    
    # Call to isdir(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'real_f' (line 176)
    real_f_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 25), 'real_f', False)
    # Processing the call keyword arguments (line 176)
    kwargs_393 = {}
    # Getting the type of 'os' (line 176)
    os_389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 176)
    path_390 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 11), os_389, 'path')
    # Obtaining the member 'isdir' of a type (line 176)
    isdir_391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 11), path_390, 'isdir')
    # Calling isdir(args, kwargs) (line 176)
    isdir_call_result_394 = invoke(stypy.reporting.localization.Localization(__file__, 176, 11), isdir_391, *[real_f_392], **kwargs_393)
    
    
    
    # Call to islink(...): (line 176)
    # Processing the call arguments (line 176)
    # Getting the type of 'real_f' (line 176)
    real_f_398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 56), 'real_f', False)
    # Processing the call keyword arguments (line 176)
    kwargs_399 = {}
    # Getting the type of 'os' (line 176)
    os_395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 41), 'os', False)
    # Obtaining the member 'path' of a type (line 176)
    path_396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 41), os_395, 'path')
    # Obtaining the member 'islink' of a type (line 176)
    islink_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 176, 41), path_396, 'islink')
    # Calling islink(args, kwargs) (line 176)
    islink_call_result_400 = invoke(stypy.reporting.localization.Localization(__file__, 176, 41), islink_397, *[real_f_398], **kwargs_399)
    
    # Applying the 'not' unary operator (line 176)
    result_not__401 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 37), 'not', islink_call_result_400)
    
    # Applying the binary operator 'and' (line 176)
    result_and_keyword_402 = python_operator(stypy.reporting.localization.Localization(__file__, 176, 11), 'and', isdir_call_result_394, result_not__401)
    
    # Testing the type of an if condition (line 176)
    if_condition_403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 176, 8), result_and_keyword_402)
    # Assigning a type to the variable 'if_condition_403' (line 176)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 176, 8), 'if_condition_403', if_condition_403)
    # SSA begins for if statement (line 176)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to _build_cmdtuple(...): (line 177)
    # Processing the call arguments (line 177)
    # Getting the type of 'real_f' (line 177)
    real_f_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 28), 'real_f', False)
    # Getting the type of 'cmdtuples' (line 177)
    cmdtuples_406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 36), 'cmdtuples', False)
    # Processing the call keyword arguments (line 177)
    kwargs_407 = {}
    # Getting the type of '_build_cmdtuple' (line 177)
    _build_cmdtuple_404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 12), '_build_cmdtuple', False)
    # Calling _build_cmdtuple(args, kwargs) (line 177)
    _build_cmdtuple_call_result_408 = invoke(stypy.reporting.localization.Localization(__file__, 177, 12), _build_cmdtuple_404, *[real_f_405, cmdtuples_406], **kwargs_407)
    
    # SSA branch for the else part of an if statement (line 176)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 179)
    # Processing the call arguments (line 179)
    
    # Obtaining an instance of the builtin type 'tuple' (line 179)
    tuple_411 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 179)
    # Adding element type (line 179)
    # Getting the type of 'os' (line 179)
    os_412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 30), 'os', False)
    # Obtaining the member 'remove' of a type (line 179)
    remove_413 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 30), os_412, 'remove')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), tuple_411, remove_413)
    # Adding element type (line 179)
    # Getting the type of 'real_f' (line 179)
    real_f_414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 41), 'real_f', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), tuple_411, real_f_414)
    
    # Processing the call keyword arguments (line 179)
    kwargs_415 = {}
    # Getting the type of 'cmdtuples' (line 179)
    cmdtuples_409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'cmdtuples', False)
    # Obtaining the member 'append' of a type (line 179)
    append_410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 179, 12), cmdtuples_409, 'append')
    # Calling append(args, kwargs) (line 179)
    append_call_result_416 = invoke(stypy.reporting.localization.Localization(__file__, 179, 12), append_410, *[tuple_411], **kwargs_415)
    
    # SSA join for if statement (line 176)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 180)
    # Processing the call arguments (line 180)
    
    # Obtaining an instance of the builtin type 'tuple' (line 180)
    tuple_419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 22), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 180)
    # Adding element type (line 180)
    # Getting the type of 'os' (line 180)
    os_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 22), 'os', False)
    # Obtaining the member 'rmdir' of a type (line 180)
    rmdir_421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 22), os_420, 'rmdir')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 22), tuple_419, rmdir_421)
    # Adding element type (line 180)
    # Getting the type of 'path' (line 180)
    path_422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 32), 'path', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 22), tuple_419, path_422)
    
    # Processing the call keyword arguments (line 180)
    kwargs_423 = {}
    # Getting the type of 'cmdtuples' (line 180)
    cmdtuples_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'cmdtuples', False)
    # Obtaining the member 'append' of a type (line 180)
    append_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 180, 4), cmdtuples_417, 'append')
    # Calling append(args, kwargs) (line 180)
    append_call_result_424 = invoke(stypy.reporting.localization.Localization(__file__, 180, 4), append_418, *[tuple_419], **kwargs_423)
    
    
    # ################# End of '_build_cmdtuple(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_build_cmdtuple' in the type store
    # Getting the type of 'stypy_return_type' (line 172)
    stypy_return_type_425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_425)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_build_cmdtuple'
    return stypy_return_type_425

# Assigning a type to the variable '_build_cmdtuple' (line 172)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 0), '_build_cmdtuple', _build_cmdtuple)

@norecursion
def remove_tree(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 35), 'int')
    int_427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 46), 'int')
    defaults = [int_426, int_427]
    # Create a new context for function 'remove_tree'
    module_type_store = module_type_store.open_function_context('remove_tree', 182, 0, False)
    
    # Passed parameters checking function
    remove_tree.stypy_localization = localization
    remove_tree.stypy_type_of_self = None
    remove_tree.stypy_type_store = module_type_store
    remove_tree.stypy_function_name = 'remove_tree'
    remove_tree.stypy_param_names_list = ['directory', 'verbose', 'dry_run']
    remove_tree.stypy_varargs_param_name = None
    remove_tree.stypy_kwargs_param_name = None
    remove_tree.stypy_call_defaults = defaults
    remove_tree.stypy_call_varargs = varargs
    remove_tree.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'remove_tree', ['directory', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'remove_tree', localization, ['directory', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'remove_tree(...)' code ##################

    str_428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'str', "Recursively remove an entire directory tree.\n\n    Any errors are ignored (apart from being reported to stdout if 'verbose'\n    is true).\n    ")
    # Marking variables as global (line 188)
    module_type_store.declare_global(stypy.reporting.localization.Localization(__file__, 188, 4), '_path_created')
    
    
    # Getting the type of 'verbose' (line 190)
    verbose_429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 7), 'verbose')
    int_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 18), 'int')
    # Applying the binary operator '>=' (line 190)
    result_ge_431 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 7), '>=', verbose_429, int_430)
    
    # Testing the type of an if condition (line 190)
    if_condition_432 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 190, 4), result_ge_431)
    # Assigning a type to the variable 'if_condition_432' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 4), 'if_condition_432', if_condition_432)
    # SSA begins for if statement (line 190)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to info(...): (line 191)
    # Processing the call arguments (line 191)
    str_435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 17), 'str', "removing '%s' (and everything under it)")
    # Getting the type of 'directory' (line 191)
    directory_436 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 60), 'directory', False)
    # Processing the call keyword arguments (line 191)
    kwargs_437 = {}
    # Getting the type of 'log' (line 191)
    log_433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 8), 'log', False)
    # Obtaining the member 'info' of a type (line 191)
    info_434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 8), log_433, 'info')
    # Calling info(args, kwargs) (line 191)
    info_call_result_438 = invoke(stypy.reporting.localization.Localization(__file__, 191, 8), info_434, *[str_435, directory_436], **kwargs_437)
    
    # SSA join for if statement (line 190)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'dry_run' (line 192)
    dry_run_439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 7), 'dry_run')
    # Testing the type of an if condition (line 192)
    if_condition_440 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 4), dry_run_439)
    # Assigning a type to the variable 'if_condition_440' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'if_condition_440', if_condition_440)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a List to a Name (line 194):
    
    # Assigning a List to a Name (line 194):
    
    # Obtaining an instance of the builtin type 'list' (line 194)
    list_441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 16), 'list')
    # Adding type elements to the builtin type 'list' instance (line 194)
    
    # Assigning a type to the variable 'cmdtuples' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 4), 'cmdtuples', list_441)
    
    # Call to _build_cmdtuple(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'directory' (line 195)
    directory_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'directory', False)
    # Getting the type of 'cmdtuples' (line 195)
    cmdtuples_444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 31), 'cmdtuples', False)
    # Processing the call keyword arguments (line 195)
    kwargs_445 = {}
    # Getting the type of '_build_cmdtuple' (line 195)
    _build_cmdtuple_442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 4), '_build_cmdtuple', False)
    # Calling _build_cmdtuple(args, kwargs) (line 195)
    _build_cmdtuple_call_result_446 = invoke(stypy.reporting.localization.Localization(__file__, 195, 4), _build_cmdtuple_442, *[directory_443, cmdtuples_444], **kwargs_445)
    
    
    # Getting the type of 'cmdtuples' (line 196)
    cmdtuples_447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 15), 'cmdtuples')
    # Testing the type of a for loop iterable (line 196)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 196, 4), cmdtuples_447)
    # Getting the type of the for loop variable (line 196)
    for_loop_var_448 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 196, 4), cmdtuples_447)
    # Assigning a type to the variable 'cmd' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 4), 'cmd', for_loop_var_448)
    # SSA begins for a for statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # SSA begins for try-except statement (line 197)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to (...): (line 198)
    # Processing the call arguments (line 198)
    
    # Obtaining the type of the subscript
    int_453 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 23), 'int')
    # Getting the type of 'cmd' (line 198)
    cmd_454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 19), 'cmd', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 19), cmd_454, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_456 = invoke(stypy.reporting.localization.Localization(__file__, 198, 19), getitem___455, int_453)
    
    # Processing the call keyword arguments (line 198)
    kwargs_457 = {}
    
    # Obtaining the type of the subscript
    int_449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 16), 'int')
    # Getting the type of 'cmd' (line 198)
    cmd_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'cmd', False)
    # Obtaining the member '__getitem__' of a type (line 198)
    getitem___451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 12), cmd_450, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 198)
    subscript_call_result_452 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), getitem___451, int_449)
    
    # Calling (args, kwargs) (line 198)
    _call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 198, 12), subscript_call_result_452, *[subscript_call_result_456], **kwargs_457)
    
    
    # Assigning a Call to a Name (line 200):
    
    # Assigning a Call to a Name (line 200):
    
    # Call to abspath(...): (line 200)
    # Processing the call arguments (line 200)
    
    # Obtaining the type of the subscript
    int_462 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 42), 'int')
    # Getting the type of 'cmd' (line 200)
    cmd_463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 38), 'cmd', False)
    # Obtaining the member '__getitem__' of a type (line 200)
    getitem___464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 38), cmd_463, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 200)
    subscript_call_result_465 = invoke(stypy.reporting.localization.Localization(__file__, 200, 38), getitem___464, int_462)
    
    # Processing the call keyword arguments (line 200)
    kwargs_466 = {}
    # Getting the type of 'os' (line 200)
    os_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 22), 'os', False)
    # Obtaining the member 'path' of a type (line 200)
    path_460 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), os_459, 'path')
    # Obtaining the member 'abspath' of a type (line 200)
    abspath_461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 200, 22), path_460, 'abspath')
    # Calling abspath(args, kwargs) (line 200)
    abspath_call_result_467 = invoke(stypy.reporting.localization.Localization(__file__, 200, 22), abspath_461, *[subscript_call_result_465], **kwargs_466)
    
    # Assigning a type to the variable 'abspath' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 12), 'abspath', abspath_call_result_467)
    
    
    # Getting the type of 'abspath' (line 201)
    abspath_468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 15), 'abspath')
    # Getting the type of '_path_created' (line 201)
    _path_created_469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 26), '_path_created')
    # Applying the binary operator 'in' (line 201)
    result_contains_470 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 15), 'in', abspath_468, _path_created_469)
    
    # Testing the type of an if condition (line 201)
    if_condition_471 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 201, 12), result_contains_470)
    # Assigning a type to the variable 'if_condition_471' (line 201)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 201, 12), 'if_condition_471', if_condition_471)
    # SSA begins for if statement (line 201)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Deleting a member
    # Getting the type of '_path_created' (line 202)
    _path_created_472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), '_path_created')
    
    # Obtaining the type of the subscript
    # Getting the type of 'abspath' (line 202)
    abspath_473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 34), 'abspath')
    # Getting the type of '_path_created' (line 202)
    _path_created_474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), '_path_created')
    # Obtaining the member '__getitem__' of a type (line 202)
    getitem___475 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 20), _path_created_474, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 202)
    subscript_call_result_476 = invoke(stypy.reporting.localization.Localization(__file__, 202, 20), getitem___475, abspath_473)
    
    del_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 16), _path_created_472, subscript_call_result_476)
    # SSA join for if statement (line 201)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the except part of a try statement (line 197)
    # SSA branch for the except 'Tuple' branch of a try statement (line 197)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    # Getting the type of 'IOError' (line 203)
    IOError_478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 16), 'IOError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 16), tuple_477, IOError_478)
    # Adding element type (line 203)
    # Getting the type of 'OSError' (line 203)
    OSError_479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 25), 'OSError')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 16), tuple_477, OSError_479)
    
    # Assigning a type to the variable 'exc' (line 203)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 203, 8), 'exc', tuple_477)
    
    # Call to warn(...): (line 204)
    # Processing the call arguments (line 204)
    str_482 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 21), 'str', 'error removing %s: %s')
    # Getting the type of 'directory' (line 204)
    directory_483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 46), 'directory', False)
    # Getting the type of 'exc' (line 204)
    exc_484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 57), 'exc', False)
    # Processing the call keyword arguments (line 204)
    kwargs_485 = {}
    # Getting the type of 'log' (line 204)
    log_480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 12), 'log', False)
    # Obtaining the member 'warn' of a type (line 204)
    warn_481 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 12), log_480, 'warn')
    # Calling warn(args, kwargs) (line 204)
    warn_call_result_486 = invoke(stypy.reporting.localization.Localization(__file__, 204, 12), warn_481, *[str_482, directory_483, exc_484], **kwargs_485)
    
    # SSA join for try-except statement (line 197)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # ################# End of 'remove_tree(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'remove_tree' in the type store
    # Getting the type of 'stypy_return_type' (line 182)
    stypy_return_type_487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_487)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'remove_tree'
    return stypy_return_type_487

# Assigning a type to the variable 'remove_tree' (line 182)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 0), 'remove_tree', remove_tree)

@norecursion
def ensure_relative(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'ensure_relative'
    module_type_store = module_type_store.open_function_context('ensure_relative', 206, 0, False)
    
    # Passed parameters checking function
    ensure_relative.stypy_localization = localization
    ensure_relative.stypy_type_of_self = None
    ensure_relative.stypy_type_store = module_type_store
    ensure_relative.stypy_function_name = 'ensure_relative'
    ensure_relative.stypy_param_names_list = ['path']
    ensure_relative.stypy_varargs_param_name = None
    ensure_relative.stypy_kwargs_param_name = None
    ensure_relative.stypy_call_defaults = defaults
    ensure_relative.stypy_call_varargs = varargs
    ensure_relative.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'ensure_relative', ['path'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'ensure_relative', localization, ['path'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'ensure_relative(...)' code ##################

    str_488 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', "Take the full path 'path', and make it a relative path.\n\n    This is useful to make 'path' the second argument to os.path.join().\n    ")
    
    # Assigning a Call to a Tuple (line 211):
    
    # Assigning a Subscript to a Name (line 211):
    
    # Obtaining the type of the subscript
    int_489 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 4), 'int')
    
    # Call to splitdrive(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'path' (line 211)
    path_493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 37), 'path', False)
    # Processing the call keyword arguments (line 211)
    kwargs_494 = {}
    # Getting the type of 'os' (line 211)
    os_490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 211)
    path_491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 18), os_490, 'path')
    # Obtaining the member 'splitdrive' of a type (line 211)
    splitdrive_492 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 18), path_491, 'splitdrive')
    # Calling splitdrive(args, kwargs) (line 211)
    splitdrive_call_result_495 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), splitdrive_492, *[path_493], **kwargs_494)
    
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___496 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 4), splitdrive_call_result_495, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_497 = invoke(stypy.reporting.localization.Localization(__file__, 211, 4), getitem___496, int_489)
    
    # Assigning a type to the variable 'tuple_var_assignment_5' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'tuple_var_assignment_5', subscript_call_result_497)
    
    # Assigning a Subscript to a Name (line 211):
    
    # Obtaining the type of the subscript
    int_498 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 4), 'int')
    
    # Call to splitdrive(...): (line 211)
    # Processing the call arguments (line 211)
    # Getting the type of 'path' (line 211)
    path_502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 37), 'path', False)
    # Processing the call keyword arguments (line 211)
    kwargs_503 = {}
    # Getting the type of 'os' (line 211)
    os_499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'os', False)
    # Obtaining the member 'path' of a type (line 211)
    path_500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 18), os_499, 'path')
    # Obtaining the member 'splitdrive' of a type (line 211)
    splitdrive_501 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 18), path_500, 'splitdrive')
    # Calling splitdrive(args, kwargs) (line 211)
    splitdrive_call_result_504 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), splitdrive_501, *[path_502], **kwargs_503)
    
    # Obtaining the member '__getitem__' of a type (line 211)
    getitem___505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 4), splitdrive_call_result_504, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 211)
    subscript_call_result_506 = invoke(stypy.reporting.localization.Localization(__file__, 211, 4), getitem___505, int_498)
    
    # Assigning a type to the variable 'tuple_var_assignment_6' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'tuple_var_assignment_6', subscript_call_result_506)
    
    # Assigning a Name to a Name (line 211):
    # Getting the type of 'tuple_var_assignment_5' (line 211)
    tuple_var_assignment_5_507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'tuple_var_assignment_5')
    # Assigning a type to the variable 'drive' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'drive', tuple_var_assignment_5_507)
    
    # Assigning a Name to a Name (line 211):
    # Getting the type of 'tuple_var_assignment_6' (line 211)
    tuple_var_assignment_6_508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'tuple_var_assignment_6')
    # Assigning a type to the variable 'path' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 11), 'path', tuple_var_assignment_6_508)
    
    
    
    # Obtaining the type of the subscript
    int_509 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 12), 'int')
    int_510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 14), 'int')
    slice_511 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 212, 7), int_509, int_510, None)
    # Getting the type of 'path' (line 212)
    path_512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 7), 'path')
    # Obtaining the member '__getitem__' of a type (line 212)
    getitem___513 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 7), path_512, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 212)
    subscript_call_result_514 = invoke(stypy.reporting.localization.Localization(__file__, 212, 7), getitem___513, slice_511)
    
    # Getting the type of 'os' (line 212)
    os_515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'os')
    # Obtaining the member 'sep' of a type (line 212)
    sep_516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), os_515, 'sep')
    # Applying the binary operator '==' (line 212)
    result_eq_517 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 7), '==', subscript_call_result_514, sep_516)
    
    # Testing the type of an if condition (line 212)
    if_condition_518 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 212, 4), result_eq_517)
    # Assigning a type to the variable 'if_condition_518' (line 212)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'if_condition_518', if_condition_518)
    # SSA begins for if statement (line 212)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a BinOp to a Name (line 213):
    
    # Assigning a BinOp to a Name (line 213):
    # Getting the type of 'drive' (line 213)
    drive_519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 15), 'drive')
    
    # Obtaining the type of the subscript
    int_520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 28), 'int')
    slice_521 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 213, 23), int_520, None, None)
    # Getting the type of 'path' (line 213)
    path_522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 23), 'path')
    # Obtaining the member '__getitem__' of a type (line 213)
    getitem___523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 23), path_522, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 213)
    subscript_call_result_524 = invoke(stypy.reporting.localization.Localization(__file__, 213, 23), getitem___523, slice_521)
    
    # Applying the binary operator '+' (line 213)
    result_add_525 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 15), '+', drive_519, subscript_call_result_524)
    
    # Assigning a type to the variable 'path' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'path', result_add_525)
    # SSA join for if statement (line 212)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'path' (line 214)
    path_526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 11), 'path')
    # Assigning a type to the variable 'stypy_return_type' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'stypy_return_type', path_526)
    
    # ################# End of 'ensure_relative(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'ensure_relative' in the type store
    # Getting the type of 'stypy_return_type' (line 206)
    stypy_return_type_527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_527)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'ensure_relative'
    return stypy_return_type_527

# Assigning a type to the variable 'ensure_relative' (line 206)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 206, 0), 'ensure_relative', ensure_relative)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
