
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.archive_util
2: 
3: Utility functions for creating archive files (tarballs, zip files,
4: that sort of thing).'''
5: 
6: __revision__ = "$Id$"
7: 
8: import os
9: from warnings import warn
10: import sys
11: 
12: from distutils.errors import DistutilsExecError
13: from distutils.spawn import spawn
14: from distutils.dir_util import mkpath
15: from distutils import log
16: 
17: try:
18:     from pwd import getpwnam
19: except ImportError:
20:     getpwnam = None
21: 
22: try:
23:     from grp import getgrnam
24: except ImportError:
25:     getgrnam = None
26: 
27: def _get_gid(name):
28:     '''Returns a gid, given a group name.'''
29:     if getgrnam is None or name is None:
30:         return None
31:     try:
32:         result = getgrnam(name)
33:     except KeyError:
34:         result = None
35:     if result is not None:
36:         return result[2]
37:     return None
38: 
39: def _get_uid(name):
40:     '''Returns an uid, given a user name.'''
41:     if getpwnam is None or name is None:
42:         return None
43:     try:
44:         result = getpwnam(name)
45:     except KeyError:
46:         result = None
47:     if result is not None:
48:         return result[2]
49:     return None
50: 
51: def make_tarball(base_name, base_dir, compress="gzip", verbose=0, dry_run=0,
52:                  owner=None, group=None):
53:     '''Create a (possibly compressed) tar file from all the files under
54:     'base_dir'.
55: 
56:     'compress' must be "gzip" (the default), "compress", "bzip2", or None.
57:     (compress will be deprecated in Python 3.2)
58: 
59:     'owner' and 'group' can be used to define an owner and a group for the
60:     archive that is being built. If not provided, the current owner and group
61:     will be used.
62: 
63:     The output tar file will be named 'base_dir' +  ".tar", possibly plus
64:     the appropriate compression extension (".gz", ".bz2" or ".Z").
65: 
66:     Returns the output filename.
67:     '''
68:     tar_compression = {'gzip': 'gz', 'bzip2': 'bz2', None: '', 'compress': ''}
69:     compress_ext = {'gzip': '.gz', 'bzip2': '.bz2', 'compress': '.Z'}
70: 
71:     # flags for compression program, each element of list will be an argument
72:     if compress is not None and compress not in compress_ext.keys():
73:         raise ValueError, \
74:               ("bad value for 'compress': must be None, 'gzip', 'bzip2' "
75:                "or 'compress'")
76: 
77:     archive_name = base_name + '.tar'
78:     if compress != 'compress':
79:         archive_name += compress_ext.get(compress, '')
80: 
81:     mkpath(os.path.dirname(archive_name), dry_run=dry_run)
82: 
83:     # creating the tarball
84:     import tarfile  # late import so Python build itself doesn't break
85: 
86:     log.info('Creating tar archive')
87: 
88:     uid = _get_uid(owner)
89:     gid = _get_gid(group)
90: 
91:     def _set_uid_gid(tarinfo):
92:         if gid is not None:
93:             tarinfo.gid = gid
94:             tarinfo.gname = group
95:         if uid is not None:
96:             tarinfo.uid = uid
97:             tarinfo.uname = owner
98:         return tarinfo
99: 
100:     if not dry_run:
101:         tar = tarfile.open(archive_name, 'w|%s' % tar_compression[compress])
102:         try:
103:             tar.add(base_dir, filter=_set_uid_gid)
104:         finally:
105:             tar.close()
106: 
107:     # compression using `compress`
108:     if compress == 'compress':
109:         warn("'compress' will be deprecated.", PendingDeprecationWarning)
110:         # the option varies depending on the platform
111:         compressed_name = archive_name + compress_ext[compress]
112:         if sys.platform == 'win32':
113:             cmd = [compress, archive_name, compressed_name]
114:         else:
115:             cmd = [compress, '-f', archive_name]
116:         spawn(cmd, dry_run=dry_run)
117:         return compressed_name
118: 
119:     return archive_name
120: 
121: def make_zipfile(base_name, base_dir, verbose=0, dry_run=0):
122:     '''Create a zip file from all the files under 'base_dir'.
123: 
124:     The output zip file will be named 'base_name' + ".zip".  Uses either the
125:     "zipfile" Python module (if available) or the InfoZIP "zip" utility
126:     (if installed and found on the default search path).  If neither tool is
127:     available, raises DistutilsExecError.  Returns the name of the output zip
128:     file.
129:     '''
130:     try:
131:         import zipfile
132:     except ImportError:
133:         zipfile = None
134: 
135:     zip_filename = base_name + ".zip"
136:     mkpath(os.path.dirname(zip_filename), dry_run=dry_run)
137: 
138:     # If zipfile module is not available, try spawning an external
139:     # 'zip' command.
140:     if zipfile is None:
141:         if verbose:
142:             zipoptions = "-r"
143:         else:
144:             zipoptions = "-rq"
145: 
146:         try:
147:             spawn(["zip", zipoptions, zip_filename, base_dir],
148:                   dry_run=dry_run)
149:         except DistutilsExecError:
150:             # XXX really should distinguish between "couldn't find
151:             # external 'zip' command" and "zip failed".
152:             raise DistutilsExecError, \
153:                   ("unable to create zip file '%s': "
154:                    "could neither import the 'zipfile' module nor "
155:                    "find a standalone zip utility") % zip_filename
156: 
157:     else:
158:         log.info("creating '%s' and adding '%s' to it",
159:                  zip_filename, base_dir)
160: 
161:         if not dry_run:
162:             zip = zipfile.ZipFile(zip_filename, "w",
163:                                   compression=zipfile.ZIP_DEFLATED)
164: 
165:             for dirpath, dirnames, filenames in os.walk(base_dir):
166:                 for name in filenames:
167:                     path = os.path.normpath(os.path.join(dirpath, name))
168:                     if os.path.isfile(path):
169:                         zip.write(path, path)
170:                         log.info("adding '%s'" % path)
171:             zip.close()
172: 
173:     return zip_filename
174: 
175: ARCHIVE_FORMATS = {
176:     'gztar': (make_tarball, [('compress', 'gzip')], "gzip'ed tar-file"),
177:     'bztar': (make_tarball, [('compress', 'bzip2')], "bzip2'ed tar-file"),
178:     'ztar':  (make_tarball, [('compress', 'compress')], "compressed tar file"),
179:     'tar':   (make_tarball, [('compress', None)], "uncompressed tar file"),
180:     'zip':   (make_zipfile, [],"ZIP file")
181:     }
182: 
183: def check_archive_formats(formats):
184:     '''Returns the first format from the 'format' list that is unknown.
185: 
186:     If all formats are known, returns None
187:     '''
188:     for format in formats:
189:         if format not in ARCHIVE_FORMATS:
190:             return format
191:     return None
192: 
193: def make_archive(base_name, format, root_dir=None, base_dir=None, verbose=0,
194:                  dry_run=0, owner=None, group=None):
195:     '''Create an archive file (eg. zip or tar).
196: 
197:     'base_name' is the name of the file to create, minus any format-specific
198:     extension; 'format' is the archive format: one of "zip", "tar", "ztar",
199:     or "gztar".
200: 
201:     'root_dir' is a directory that will be the root directory of the
202:     archive; ie. we typically chdir into 'root_dir' before creating the
203:     archive.  'base_dir' is the directory where we start archiving from;
204:     ie. 'base_dir' will be the common prefix of all files and
205:     directories in the archive.  'root_dir' and 'base_dir' both default
206:     to the current directory.  Returns the name of the archive file.
207: 
208:     'owner' and 'group' are used when creating a tar archive. By default,
209:     uses the current owner and group.
210:     '''
211:     save_cwd = os.getcwd()
212:     if root_dir is not None:
213:         log.debug("changing into '%s'", root_dir)
214:         base_name = os.path.abspath(base_name)
215:         if not dry_run:
216:             os.chdir(root_dir)
217: 
218:     if base_dir is None:
219:         base_dir = os.curdir
220: 
221:     kwargs = {'dry_run': dry_run}
222: 
223:     try:
224:         format_info = ARCHIVE_FORMATS[format]
225:     except KeyError:
226:         raise ValueError, "unknown archive format '%s'" % format
227: 
228:     func = format_info[0]
229:     for arg, val in format_info[1]:
230:         kwargs[arg] = val
231: 
232:     if format != 'zip':
233:         kwargs['owner'] = owner
234:         kwargs['group'] = group
235: 
236:     try:
237:         filename = func(base_name, base_dir, **kwargs)
238:     finally:
239:         if root_dir is not None:
240:             log.debug("changing back to '%s'", save_cwd)
241:             os.chdir(save_cwd)
242: 
243:     return filename
244: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_302364 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', 'distutils.archive_util\n\nUtility functions for creating archive files (tarballs, zip files,\nthat sort of thing).')

# Assigning a Str to a Name (line 6):
str_302365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_302365)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from warnings import warn' statement (line 9)
try:
    from warnings import warn

except:
    warn = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'warnings', None, module_type_store, ['warn'], [warn])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'import sys' statement (line 10)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.errors import DistutilsExecError' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302366 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors')

if (type(import_302366) is not StypyTypeError):

    if (import_302366 != 'pyd_module'):
        __import__(import_302366)
        sys_modules_302367 = sys.modules[import_302366]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', sys_modules_302367.module_type_store, module_type_store, ['DistutilsExecError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_302367, sys_modules_302367.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsExecError

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', None, module_type_store, ['DistutilsExecError'], [DistutilsExecError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', import_302366)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 0))

# 'from distutils.spawn import spawn' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302368 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.spawn')

if (type(import_302368) is not StypyTypeError):

    if (import_302368 != 'pyd_module'):
        __import__(import_302368)
        sys_modules_302369 = sys.modules[import_302368]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.spawn', sys_modules_302369.module_type_store, module_type_store, ['spawn'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 0), __file__, sys_modules_302369, sys_modules_302369.module_type_store, module_type_store)
    else:
        from distutils.spawn import spawn

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.spawn', None, module_type_store, ['spawn'], [spawn])

else:
    # Assigning a type to the variable 'distutils.spawn' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'distutils.spawn', import_302368)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 0))

# 'from distutils.dir_util import mkpath' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302370 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.dir_util')

if (type(import_302370) is not StypyTypeError):

    if (import_302370 != 'pyd_module'):
        __import__(import_302370)
        sys_modules_302371 = sys.modules[import_302370]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.dir_util', sys_modules_302371.module_type_store, module_type_store, ['mkpath'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 0), __file__, sys_modules_302371, sys_modules_302371.module_type_store, module_type_store)
    else:
        from distutils.dir_util import mkpath

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.dir_util', None, module_type_store, ['mkpath'], [mkpath])

else:
    # Assigning a type to the variable 'distutils.dir_util' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'distutils.dir_util', import_302370)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 0))

# 'from distutils import log' statement (line 15)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 15, 0), 'distutils', None, module_type_store, ['log'], [log])



# SSA begins for try-except statement (line 17)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))

# 'from pwd import getpwnam' statement (line 18)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302372 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'pwd')

if (type(import_302372) is not StypyTypeError):

    if (import_302372 != 'pyd_module'):
        __import__(import_302372)
        sys_modules_302373 = sys.modules[import_302372]
        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'pwd', sys_modules_302373.module_type_store, module_type_store, ['getpwnam'])
        nest_module(stypy.reporting.localization.Localization(__file__, 18, 4), __file__, sys_modules_302373, sys_modules_302373.module_type_store, module_type_store)
    else:
        from pwd import getpwnam

        import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'pwd', None, module_type_store, ['getpwnam'], [getpwnam])

else:
    # Assigning a type to the variable 'pwd' (line 18)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'pwd', import_302372)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

# SSA branch for the except part of a try statement (line 17)
# SSA branch for the except 'ImportError' branch of a try statement (line 17)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 20):
# Getting the type of 'None' (line 20)
None_302374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'None')
# Assigning a type to the variable 'getpwnam' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'getpwnam', None_302374)
# SSA join for try-except statement (line 17)
module_type_store = module_type_store.join_ssa_context()



# SSA begins for try-except statement (line 22)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 4))

# 'from grp import getgrnam' statement (line 23)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_302375 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'grp')

if (type(import_302375) is not StypyTypeError):

    if (import_302375 != 'pyd_module'):
        __import__(import_302375)
        sys_modules_302376 = sys.modules[import_302375]
        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'grp', sys_modules_302376.module_type_store, module_type_store, ['getgrnam'])
        nest_module(stypy.reporting.localization.Localization(__file__, 23, 4), __file__, sys_modules_302376, sys_modules_302376.module_type_store, module_type_store)
    else:
        from grp import getgrnam

        import_from_module(stypy.reporting.localization.Localization(__file__, 23, 4), 'grp', None, module_type_store, ['getgrnam'], [getgrnam])

else:
    # Assigning a type to the variable 'grp' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'grp', import_302375)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

# SSA branch for the except part of a try statement (line 22)
# SSA branch for the except 'ImportError' branch of a try statement (line 22)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 25):
# Getting the type of 'None' (line 25)
None_302377 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'None')
# Assigning a type to the variable 'getgrnam' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'getgrnam', None_302377)
# SSA join for try-except statement (line 22)
module_type_store = module_type_store.join_ssa_context()


@norecursion
def _get_gid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_gid'
    module_type_store = module_type_store.open_function_context('_get_gid', 27, 0, False)
    
    # Passed parameters checking function
    _get_gid.stypy_localization = localization
    _get_gid.stypy_type_of_self = None
    _get_gid.stypy_type_store = module_type_store
    _get_gid.stypy_function_name = '_get_gid'
    _get_gid.stypy_param_names_list = ['name']
    _get_gid.stypy_varargs_param_name = None
    _get_gid.stypy_kwargs_param_name = None
    _get_gid.stypy_call_defaults = defaults
    _get_gid.stypy_call_varargs = varargs
    _get_gid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_gid', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_gid', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_gid(...)' code ##################

    str_302378 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 4), 'str', 'Returns a gid, given a group name.')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'getgrnam' (line 29)
    getgrnam_302379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 7), 'getgrnam')
    # Getting the type of 'None' (line 29)
    None_302380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'None')
    # Applying the binary operator 'is' (line 29)
    result_is__302381 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), 'is', getgrnam_302379, None_302380)
    
    
    # Getting the type of 'name' (line 29)
    name_302382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 27), 'name')
    # Getting the type of 'None' (line 29)
    None_302383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'None')
    # Applying the binary operator 'is' (line 29)
    result_is__302384 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 27), 'is', name_302382, None_302383)
    
    # Applying the binary operator 'or' (line 29)
    result_or_keyword_302385 = python_operator(stypy.reporting.localization.Localization(__file__, 29, 7), 'or', result_is__302381, result_is__302384)
    
    # Testing the type of an if condition (line 29)
    if_condition_302386 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 4), result_or_keyword_302385)
    # Assigning a type to the variable 'if_condition_302386' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'if_condition_302386', if_condition_302386)
    # SSA begins for if statement (line 29)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 30)
    None_302387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 30)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', None_302387)
    # SSA join for if statement (line 29)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 32):
    
    # Call to getgrnam(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'name' (line 32)
    name_302389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 26), 'name', False)
    # Processing the call keyword arguments (line 32)
    kwargs_302390 = {}
    # Getting the type of 'getgrnam' (line 32)
    getgrnam_302388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'getgrnam', False)
    # Calling getgrnam(args, kwargs) (line 32)
    getgrnam_call_result_302391 = invoke(stypy.reporting.localization.Localization(__file__, 32, 17), getgrnam_302388, *[name_302389], **kwargs_302390)
    
    # Assigning a type to the variable 'result' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'result', getgrnam_call_result_302391)
    # SSA branch for the except part of a try statement (line 31)
    # SSA branch for the except 'KeyError' branch of a try statement (line 31)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 34):
    # Getting the type of 'None' (line 34)
    None_302392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 17), 'None')
    # Assigning a type to the variable 'result' (line 34)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'result', None_302392)
    # SSA join for try-except statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 35)
    # Getting the type of 'result' (line 35)
    result_302393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'result')
    # Getting the type of 'None' (line 35)
    None_302394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 21), 'None')
    
    (may_be_302395, more_types_in_union_302396) = may_not_be_none(result_302393, None_302394)

    if may_be_302395:

        if more_types_in_union_302396:
            # Runtime conditional SSA (line 35)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Obtaining the type of the subscript
        int_302397 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 22), 'int')
        # Getting the type of 'result' (line 36)
        result_302398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'result')
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___302399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 15), result_302398, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_302400 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), getitem___302399, int_302397)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', subscript_call_result_302400)

        if more_types_in_union_302396:
            # SSA join for if statement (line 35)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'None' (line 37)
    None_302401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 37)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type', None_302401)
    
    # ################# End of '_get_gid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_gid' in the type store
    # Getting the type of 'stypy_return_type' (line 27)
    stypy_return_type_302402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_302402)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_gid'
    return stypy_return_type_302402

# Assigning a type to the variable '_get_gid' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), '_get_gid', _get_gid)

@norecursion
def _get_uid(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_get_uid'
    module_type_store = module_type_store.open_function_context('_get_uid', 39, 0, False)
    
    # Passed parameters checking function
    _get_uid.stypy_localization = localization
    _get_uid.stypy_type_of_self = None
    _get_uid.stypy_type_store = module_type_store
    _get_uid.stypy_function_name = '_get_uid'
    _get_uid.stypy_param_names_list = ['name']
    _get_uid.stypy_varargs_param_name = None
    _get_uid.stypy_kwargs_param_name = None
    _get_uid.stypy_call_defaults = defaults
    _get_uid.stypy_call_varargs = varargs
    _get_uid.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_get_uid', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_get_uid', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_get_uid(...)' code ##################

    str_302403 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 4), 'str', 'Returns an uid, given a user name.')
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'getpwnam' (line 41)
    getpwnam_302404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 7), 'getpwnam')
    # Getting the type of 'None' (line 41)
    None_302405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 19), 'None')
    # Applying the binary operator 'is' (line 41)
    result_is__302406 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), 'is', getpwnam_302404, None_302405)
    
    
    # Getting the type of 'name' (line 41)
    name_302407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 27), 'name')
    # Getting the type of 'None' (line 41)
    None_302408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'None')
    # Applying the binary operator 'is' (line 41)
    result_is__302409 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 27), 'is', name_302407, None_302408)
    
    # Applying the binary operator 'or' (line 41)
    result_or_keyword_302410 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 7), 'or', result_is__302406, result_is__302409)
    
    # Testing the type of an if condition (line 41)
    if_condition_302411 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 41, 4), result_or_keyword_302410)
    # Assigning a type to the variable 'if_condition_302411' (line 41)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'if_condition_302411', if_condition_302411)
    # SSA begins for if statement (line 41)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'None' (line 42)
    None_302412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 42)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', None_302412)
    # SSA join for if statement (line 41)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 44):
    
    # Call to getpwnam(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'name' (line 44)
    name_302414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 26), 'name', False)
    # Processing the call keyword arguments (line 44)
    kwargs_302415 = {}
    # Getting the type of 'getpwnam' (line 44)
    getpwnam_302413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 17), 'getpwnam', False)
    # Calling getpwnam(args, kwargs) (line 44)
    getpwnam_call_result_302416 = invoke(stypy.reporting.localization.Localization(__file__, 44, 17), getpwnam_302413, *[name_302414], **kwargs_302415)
    
    # Assigning a type to the variable 'result' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'result', getpwnam_call_result_302416)
    # SSA branch for the except part of a try statement (line 43)
    # SSA branch for the except 'KeyError' branch of a try statement (line 43)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 46):
    # Getting the type of 'None' (line 46)
    None_302417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'None')
    # Assigning a type to the variable 'result' (line 46)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'result', None_302417)
    # SSA join for try-except statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Type idiom detected: calculating its left and rigth part (line 47)
    # Getting the type of 'result' (line 47)
    result_302418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'result')
    # Getting the type of 'None' (line 47)
    None_302419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 21), 'None')
    
    (may_be_302420, more_types_in_union_302421) = may_not_be_none(result_302418, None_302419)

    if may_be_302420:

        if more_types_in_union_302421:
            # Runtime conditional SSA (line 47)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Obtaining the type of the subscript
        int_302422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 22), 'int')
        # Getting the type of 'result' (line 48)
        result_302423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'result')
        # Obtaining the member '__getitem__' of a type (line 48)
        getitem___302424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), result_302423, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 48)
        subscript_call_result_302425 = invoke(stypy.reporting.localization.Localization(__file__, 48, 15), getitem___302424, int_302422)
        
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', subscript_call_result_302425)

        if more_types_in_union_302421:
            # SSA join for if statement (line 47)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'None' (line 49)
    None_302426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 49)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type', None_302426)
    
    # ################# End of '_get_uid(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_get_uid' in the type store
    # Getting the type of 'stypy_return_type' (line 39)
    stypy_return_type_302427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_302427)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_get_uid'
    return stypy_return_type_302427

# Assigning a type to the variable '_get_uid' (line 39)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), '_get_uid', _get_uid)

@norecursion
def make_tarball(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    str_302428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 47), 'str', 'gzip')
    int_302429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 63), 'int')
    int_302430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 74), 'int')
    # Getting the type of 'None' (line 52)
    None_302431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 23), 'None')
    # Getting the type of 'None' (line 52)
    None_302432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 35), 'None')
    defaults = [str_302428, int_302429, int_302430, None_302431, None_302432]
    # Create a new context for function 'make_tarball'
    module_type_store = module_type_store.open_function_context('make_tarball', 51, 0, False)
    
    # Passed parameters checking function
    make_tarball.stypy_localization = localization
    make_tarball.stypy_type_of_self = None
    make_tarball.stypy_type_store = module_type_store
    make_tarball.stypy_function_name = 'make_tarball'
    make_tarball.stypy_param_names_list = ['base_name', 'base_dir', 'compress', 'verbose', 'dry_run', 'owner', 'group']
    make_tarball.stypy_varargs_param_name = None
    make_tarball.stypy_kwargs_param_name = None
    make_tarball.stypy_call_defaults = defaults
    make_tarball.stypy_call_varargs = varargs
    make_tarball.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_tarball', ['base_name', 'base_dir', 'compress', 'verbose', 'dry_run', 'owner', 'group'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_tarball', localization, ['base_name', 'base_dir', 'compress', 'verbose', 'dry_run', 'owner', 'group'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_tarball(...)' code ##################

    str_302433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, (-1)), 'str', 'Create a (possibly compressed) tar file from all the files under\n    \'base_dir\'.\n\n    \'compress\' must be "gzip" (the default), "compress", "bzip2", or None.\n    (compress will be deprecated in Python 3.2)\n\n    \'owner\' and \'group\' can be used to define an owner and a group for the\n    archive that is being built. If not provided, the current owner and group\n    will be used.\n\n    The output tar file will be named \'base_dir\' +  ".tar", possibly plus\n    the appropriate compression extension (".gz", ".bz2" or ".Z").\n\n    Returns the output filename.\n    ')
    
    # Assigning a Dict to a Name (line 68):
    
    # Obtaining an instance of the builtin type 'dict' (line 68)
    dict_302434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 68)
    # Adding element type (key, value) (line 68)
    str_302435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'str', 'gzip')
    str_302436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 31), 'str', 'gz')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), dict_302434, (str_302435, str_302436))
    # Adding element type (key, value) (line 68)
    str_302437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 37), 'str', 'bzip2')
    str_302438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 46), 'str', 'bz2')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), dict_302434, (str_302437, str_302438))
    # Adding element type (key, value) (line 68)
    # Getting the type of 'None' (line 68)
    None_302439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 53), 'None')
    str_302440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 59), 'str', '')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), dict_302434, (None_302439, str_302440))
    # Adding element type (key, value) (line 68)
    str_302441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 63), 'str', 'compress')
    str_302442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 75), 'str', '')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), dict_302434, (str_302441, str_302442))
    
    # Assigning a type to the variable 'tar_compression' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 4), 'tar_compression', dict_302434)
    
    # Assigning a Dict to a Name (line 69):
    
    # Obtaining an instance of the builtin type 'dict' (line 69)
    dict_302443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 19), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 69)
    # Adding element type (key, value) (line 69)
    str_302444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 20), 'str', 'gzip')
    str_302445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 28), 'str', '.gz')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), dict_302443, (str_302444, str_302445))
    # Adding element type (key, value) (line 69)
    str_302446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 35), 'str', 'bzip2')
    str_302447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 44), 'str', '.bz2')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), dict_302443, (str_302446, str_302447))
    # Adding element type (key, value) (line 69)
    str_302448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 52), 'str', 'compress')
    str_302449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 64), 'str', '.Z')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 69, 19), dict_302443, (str_302448, str_302449))
    
    # Assigning a type to the variable 'compress_ext' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'compress_ext', dict_302443)
    
    
    # Evaluating a boolean operation
    
    # Getting the type of 'compress' (line 72)
    compress_302450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 7), 'compress')
    # Getting the type of 'None' (line 72)
    None_302451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 23), 'None')
    # Applying the binary operator 'isnot' (line 72)
    result_is_not_302452 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 7), 'isnot', compress_302450, None_302451)
    
    
    # Getting the type of 'compress' (line 72)
    compress_302453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 32), 'compress')
    
    # Call to keys(...): (line 72)
    # Processing the call keyword arguments (line 72)
    kwargs_302456 = {}
    # Getting the type of 'compress_ext' (line 72)
    compress_ext_302454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 48), 'compress_ext', False)
    # Obtaining the member 'keys' of a type (line 72)
    keys_302455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 48), compress_ext_302454, 'keys')
    # Calling keys(args, kwargs) (line 72)
    keys_call_result_302457 = invoke(stypy.reporting.localization.Localization(__file__, 72, 48), keys_302455, *[], **kwargs_302456)
    
    # Applying the binary operator 'notin' (line 72)
    result_contains_302458 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 32), 'notin', compress_302453, keys_call_result_302457)
    
    # Applying the binary operator 'and' (line 72)
    result_and_keyword_302459 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 7), 'and', result_is_not_302452, result_contains_302458)
    
    # Testing the type of an if condition (line 72)
    if_condition_302460 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 4), result_and_keyword_302459)
    # Assigning a type to the variable 'if_condition_302460' (line 72)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'if_condition_302460', if_condition_302460)
    # SSA begins for if statement (line 72)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'ValueError' (line 73)
    ValueError_302461 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 14), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 73, 8), ValueError_302461, 'raise parameter', BaseException)
    # SSA join for if statement (line 72)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 77):
    # Getting the type of 'base_name' (line 77)
    base_name_302462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 19), 'base_name')
    str_302463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 31), 'str', '.tar')
    # Applying the binary operator '+' (line 77)
    result_add_302464 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 19), '+', base_name_302462, str_302463)
    
    # Assigning a type to the variable 'archive_name' (line 77)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 4), 'archive_name', result_add_302464)
    
    
    # Getting the type of 'compress' (line 78)
    compress_302465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 7), 'compress')
    str_302466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 19), 'str', 'compress')
    # Applying the binary operator '!=' (line 78)
    result_ne_302467 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 7), '!=', compress_302465, str_302466)
    
    # Testing the type of an if condition (line 78)
    if_condition_302468 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 78, 4), result_ne_302467)
    # Assigning a type to the variable 'if_condition_302468' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'if_condition_302468', if_condition_302468)
    # SSA begins for if statement (line 78)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Getting the type of 'archive_name' (line 79)
    archive_name_302469 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'archive_name')
    
    # Call to get(...): (line 79)
    # Processing the call arguments (line 79)
    # Getting the type of 'compress' (line 79)
    compress_302472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 41), 'compress', False)
    str_302473 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 51), 'str', '')
    # Processing the call keyword arguments (line 79)
    kwargs_302474 = {}
    # Getting the type of 'compress_ext' (line 79)
    compress_ext_302470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 24), 'compress_ext', False)
    # Obtaining the member 'get' of a type (line 79)
    get_302471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 24), compress_ext_302470, 'get')
    # Calling get(args, kwargs) (line 79)
    get_call_result_302475 = invoke(stypy.reporting.localization.Localization(__file__, 79, 24), get_302471, *[compress_302472, str_302473], **kwargs_302474)
    
    # Applying the binary operator '+=' (line 79)
    result_iadd_302476 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 8), '+=', archive_name_302469, get_call_result_302475)
    # Assigning a type to the variable 'archive_name' (line 79)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'archive_name', result_iadd_302476)
    
    # SSA join for if statement (line 78)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to mkpath(...): (line 81)
    # Processing the call arguments (line 81)
    
    # Call to dirname(...): (line 81)
    # Processing the call arguments (line 81)
    # Getting the type of 'archive_name' (line 81)
    archive_name_302481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 27), 'archive_name', False)
    # Processing the call keyword arguments (line 81)
    kwargs_302482 = {}
    # Getting the type of 'os' (line 81)
    os_302478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 81)
    path_302479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), os_302478, 'path')
    # Obtaining the member 'dirname' of a type (line 81)
    dirname_302480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 11), path_302479, 'dirname')
    # Calling dirname(args, kwargs) (line 81)
    dirname_call_result_302483 = invoke(stypy.reporting.localization.Localization(__file__, 81, 11), dirname_302480, *[archive_name_302481], **kwargs_302482)
    
    # Processing the call keyword arguments (line 81)
    # Getting the type of 'dry_run' (line 81)
    dry_run_302484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 50), 'dry_run', False)
    keyword_302485 = dry_run_302484
    kwargs_302486 = {'dry_run': keyword_302485}
    # Getting the type of 'mkpath' (line 81)
    mkpath_302477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'mkpath', False)
    # Calling mkpath(args, kwargs) (line 81)
    mkpath_call_result_302487 = invoke(stypy.reporting.localization.Localization(__file__, 81, 4), mkpath_302477, *[dirname_call_result_302483], **kwargs_302486)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 84, 4))
    
    # 'import tarfile' statement (line 84)
    import tarfile

    import_module(stypy.reporting.localization.Localization(__file__, 84, 4), 'tarfile', tarfile, module_type_store)
    
    
    # Call to info(...): (line 86)
    # Processing the call arguments (line 86)
    str_302490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 13), 'str', 'Creating tar archive')
    # Processing the call keyword arguments (line 86)
    kwargs_302491 = {}
    # Getting the type of 'log' (line 86)
    log_302488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 4), 'log', False)
    # Obtaining the member 'info' of a type (line 86)
    info_302489 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 4), log_302488, 'info')
    # Calling info(args, kwargs) (line 86)
    info_call_result_302492 = invoke(stypy.reporting.localization.Localization(__file__, 86, 4), info_302489, *[str_302490], **kwargs_302491)
    
    
    # Assigning a Call to a Name (line 88):
    
    # Call to _get_uid(...): (line 88)
    # Processing the call arguments (line 88)
    # Getting the type of 'owner' (line 88)
    owner_302494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 19), 'owner', False)
    # Processing the call keyword arguments (line 88)
    kwargs_302495 = {}
    # Getting the type of '_get_uid' (line 88)
    _get_uid_302493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 10), '_get_uid', False)
    # Calling _get_uid(args, kwargs) (line 88)
    _get_uid_call_result_302496 = invoke(stypy.reporting.localization.Localization(__file__, 88, 10), _get_uid_302493, *[owner_302494], **kwargs_302495)
    
    # Assigning a type to the variable 'uid' (line 88)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 4), 'uid', _get_uid_call_result_302496)
    
    # Assigning a Call to a Name (line 89):
    
    # Call to _get_gid(...): (line 89)
    # Processing the call arguments (line 89)
    # Getting the type of 'group' (line 89)
    group_302498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 19), 'group', False)
    # Processing the call keyword arguments (line 89)
    kwargs_302499 = {}
    # Getting the type of '_get_gid' (line 89)
    _get_gid_302497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 10), '_get_gid', False)
    # Calling _get_gid(args, kwargs) (line 89)
    _get_gid_call_result_302500 = invoke(stypy.reporting.localization.Localization(__file__, 89, 10), _get_gid_302497, *[group_302498], **kwargs_302499)
    
    # Assigning a type to the variable 'gid' (line 89)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 89, 4), 'gid', _get_gid_call_result_302500)

    @norecursion
    def _set_uid_gid(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_set_uid_gid'
        module_type_store = module_type_store.open_function_context('_set_uid_gid', 91, 4, False)
        
        # Passed parameters checking function
        _set_uid_gid.stypy_localization = localization
        _set_uid_gid.stypy_type_of_self = None
        _set_uid_gid.stypy_type_store = module_type_store
        _set_uid_gid.stypy_function_name = '_set_uid_gid'
        _set_uid_gid.stypy_param_names_list = ['tarinfo']
        _set_uid_gid.stypy_varargs_param_name = None
        _set_uid_gid.stypy_kwargs_param_name = None
        _set_uid_gid.stypy_call_defaults = defaults
        _set_uid_gid.stypy_call_varargs = varargs
        _set_uid_gid.stypy_call_kwargs = kwargs
        arguments = process_argument_values(localization, None, module_type_store, '_set_uid_gid', ['tarinfo'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_set_uid_gid', localization, ['tarinfo'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_set_uid_gid(...)' code ##################

        
        # Type idiom detected: calculating its left and rigth part (line 92)
        # Getting the type of 'gid' (line 92)
        gid_302501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'gid')
        # Getting the type of 'None' (line 92)
        None_302502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 22), 'None')
        
        (may_be_302503, more_types_in_union_302504) = may_not_be_none(gid_302501, None_302502)

        if may_be_302503:

            if more_types_in_union_302504:
                # Runtime conditional SSA (line 92)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 93):
            # Getting the type of 'gid' (line 93)
            gid_302505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 26), 'gid')
            # Getting the type of 'tarinfo' (line 93)
            tarinfo_302506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 12), 'tarinfo')
            # Setting the type of the member 'gid' of a type (line 93)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 12), tarinfo_302506, 'gid', gid_302505)
            
            # Assigning a Name to a Attribute (line 94):
            # Getting the type of 'group' (line 94)
            group_302507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'group')
            # Getting the type of 'tarinfo' (line 94)
            tarinfo_302508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'tarinfo')
            # Setting the type of the member 'gname' of a type (line 94)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 94, 12), tarinfo_302508, 'gname', group_302507)

            if more_types_in_union_302504:
                # SSA join for if statement (line 92)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 95)
        # Getting the type of 'uid' (line 95)
        uid_302509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'uid')
        # Getting the type of 'None' (line 95)
        None_302510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 22), 'None')
        
        (may_be_302511, more_types_in_union_302512) = may_not_be_none(uid_302509, None_302510)

        if may_be_302511:

            if more_types_in_union_302512:
                # Runtime conditional SSA (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Name to a Attribute (line 96):
            # Getting the type of 'uid' (line 96)
            uid_302513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 26), 'uid')
            # Getting the type of 'tarinfo' (line 96)
            tarinfo_302514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'tarinfo')
            # Setting the type of the member 'uid' of a type (line 96)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 12), tarinfo_302514, 'uid', uid_302513)
            
            # Assigning a Name to a Attribute (line 97):
            # Getting the type of 'owner' (line 97)
            owner_302515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 28), 'owner')
            # Getting the type of 'tarinfo' (line 97)
            tarinfo_302516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'tarinfo')
            # Setting the type of the member 'uname' of a type (line 97)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 12), tarinfo_302516, 'uname', owner_302515)

            if more_types_in_union_302512:
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'tarinfo' (line 98)
        tarinfo_302517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 15), 'tarinfo')
        # Assigning a type to the variable 'stypy_return_type' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'stypy_return_type', tarinfo_302517)
        
        # ################# End of '_set_uid_gid(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_set_uid_gid' in the type store
        # Getting the type of 'stypy_return_type' (line 91)
        stypy_return_type_302518 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_302518)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_set_uid_gid'
        return stypy_return_type_302518

    # Assigning a type to the variable '_set_uid_gid' (line 91)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), '_set_uid_gid', _set_uid_gid)
    
    
    # Getting the type of 'dry_run' (line 100)
    dry_run_302519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 11), 'dry_run')
    # Applying the 'not' unary operator (line 100)
    result_not__302520 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 7), 'not', dry_run_302519)
    
    # Testing the type of an if condition (line 100)
    if_condition_302521 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 4), result_not__302520)
    # Assigning a type to the variable 'if_condition_302521' (line 100)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'if_condition_302521', if_condition_302521)
    # SSA begins for if statement (line 100)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 101):
    
    # Call to open(...): (line 101)
    # Processing the call arguments (line 101)
    # Getting the type of 'archive_name' (line 101)
    archive_name_302524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 27), 'archive_name', False)
    str_302525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 41), 'str', 'w|%s')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compress' (line 101)
    compress_302526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 66), 'compress', False)
    # Getting the type of 'tar_compression' (line 101)
    tar_compression_302527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 50), 'tar_compression', False)
    # Obtaining the member '__getitem__' of a type (line 101)
    getitem___302528 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 50), tar_compression_302527, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 101)
    subscript_call_result_302529 = invoke(stypy.reporting.localization.Localization(__file__, 101, 50), getitem___302528, compress_302526)
    
    # Applying the binary operator '%' (line 101)
    result_mod_302530 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 41), '%', str_302525, subscript_call_result_302529)
    
    # Processing the call keyword arguments (line 101)
    kwargs_302531 = {}
    # Getting the type of 'tarfile' (line 101)
    tarfile_302522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 14), 'tarfile', False)
    # Obtaining the member 'open' of a type (line 101)
    open_302523 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 14), tarfile_302522, 'open')
    # Calling open(args, kwargs) (line 101)
    open_call_result_302532 = invoke(stypy.reporting.localization.Localization(__file__, 101, 14), open_302523, *[archive_name_302524, result_mod_302530], **kwargs_302531)
    
    # Assigning a type to the variable 'tar' (line 101)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'tar', open_call_result_302532)
    
    # Try-finally block (line 102)
    
    # Call to add(...): (line 103)
    # Processing the call arguments (line 103)
    # Getting the type of 'base_dir' (line 103)
    base_dir_302535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 20), 'base_dir', False)
    # Processing the call keyword arguments (line 103)
    # Getting the type of '_set_uid_gid' (line 103)
    _set_uid_gid_302536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 37), '_set_uid_gid', False)
    keyword_302537 = _set_uid_gid_302536
    kwargs_302538 = {'filter': keyword_302537}
    # Getting the type of 'tar' (line 103)
    tar_302533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 12), 'tar', False)
    # Obtaining the member 'add' of a type (line 103)
    add_302534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 12), tar_302533, 'add')
    # Calling add(args, kwargs) (line 103)
    add_call_result_302539 = invoke(stypy.reporting.localization.Localization(__file__, 103, 12), add_302534, *[base_dir_302535], **kwargs_302538)
    
    
    # finally branch of the try-finally block (line 102)
    
    # Call to close(...): (line 105)
    # Processing the call keyword arguments (line 105)
    kwargs_302542 = {}
    # Getting the type of 'tar' (line 105)
    tar_302540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'tar', False)
    # Obtaining the member 'close' of a type (line 105)
    close_302541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 12), tar_302540, 'close')
    # Calling close(args, kwargs) (line 105)
    close_call_result_302543 = invoke(stypy.reporting.localization.Localization(__file__, 105, 12), close_302541, *[], **kwargs_302542)
    
    
    # SSA join for if statement (line 100)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'compress' (line 108)
    compress_302544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 7), 'compress')
    str_302545 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 19), 'str', 'compress')
    # Applying the binary operator '==' (line 108)
    result_eq_302546 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 7), '==', compress_302544, str_302545)
    
    # Testing the type of an if condition (line 108)
    if_condition_302547 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), result_eq_302546)
    # Assigning a type to the variable 'if_condition_302547' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_302547', if_condition_302547)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 109)
    # Processing the call arguments (line 109)
    str_302549 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 13), 'str', "'compress' will be deprecated.")
    # Getting the type of 'PendingDeprecationWarning' (line 109)
    PendingDeprecationWarning_302550 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 47), 'PendingDeprecationWarning', False)
    # Processing the call keyword arguments (line 109)
    kwargs_302551 = {}
    # Getting the type of 'warn' (line 109)
    warn_302548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'warn', False)
    # Calling warn(args, kwargs) (line 109)
    warn_call_result_302552 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), warn_302548, *[str_302549, PendingDeprecationWarning_302550], **kwargs_302551)
    
    
    # Assigning a BinOp to a Name (line 111):
    # Getting the type of 'archive_name' (line 111)
    archive_name_302553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 26), 'archive_name')
    
    # Obtaining the type of the subscript
    # Getting the type of 'compress' (line 111)
    compress_302554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 54), 'compress')
    # Getting the type of 'compress_ext' (line 111)
    compress_ext_302555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'compress_ext')
    # Obtaining the member '__getitem__' of a type (line 111)
    getitem___302556 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 41), compress_ext_302555, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 111)
    subscript_call_result_302557 = invoke(stypy.reporting.localization.Localization(__file__, 111, 41), getitem___302556, compress_302554)
    
    # Applying the binary operator '+' (line 111)
    result_add_302558 = python_operator(stypy.reporting.localization.Localization(__file__, 111, 26), '+', archive_name_302553, subscript_call_result_302557)
    
    # Assigning a type to the variable 'compressed_name' (line 111)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'compressed_name', result_add_302558)
    
    
    # Getting the type of 'sys' (line 112)
    sys_302559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'sys')
    # Obtaining the member 'platform' of a type (line 112)
    platform_302560 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 11), sys_302559, 'platform')
    str_302561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 27), 'str', 'win32')
    # Applying the binary operator '==' (line 112)
    result_eq_302562 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), '==', platform_302560, str_302561)
    
    # Testing the type of an if condition (line 112)
    if_condition_302563 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_eq_302562)
    # Assigning a type to the variable 'if_condition_302563' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_302563', if_condition_302563)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a List to a Name (line 113):
    
    # Obtaining an instance of the builtin type 'list' (line 113)
    list_302564 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 113)
    # Adding element type (line 113)
    # Getting the type of 'compress' (line 113)
    compress_302565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'compress')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 18), list_302564, compress_302565)
    # Adding element type (line 113)
    # Getting the type of 'archive_name' (line 113)
    archive_name_302566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 29), 'archive_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 18), list_302564, archive_name_302566)
    # Adding element type (line 113)
    # Getting the type of 'compressed_name' (line 113)
    compressed_name_302567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 43), 'compressed_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 113, 18), list_302564, compressed_name_302567)
    
    # Assigning a type to the variable 'cmd' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'cmd', list_302564)
    # SSA branch for the else part of an if statement (line 112)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a List to a Name (line 115):
    
    # Obtaining an instance of the builtin type 'list' (line 115)
    list_302568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 18), 'list')
    # Adding type elements to the builtin type 'list' instance (line 115)
    # Adding element type (line 115)
    # Getting the type of 'compress' (line 115)
    compress_302569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'compress')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 18), list_302568, compress_302569)
    # Adding element type (line 115)
    str_302570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 29), 'str', '-f')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 18), list_302568, str_302570)
    # Adding element type (line 115)
    # Getting the type of 'archive_name' (line 115)
    archive_name_302571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'archive_name')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 115, 18), list_302568, archive_name_302571)
    
    # Assigning a type to the variable 'cmd' (line 115)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'cmd', list_302568)
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to spawn(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'cmd' (line 116)
    cmd_302573 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'cmd', False)
    # Processing the call keyword arguments (line 116)
    # Getting the type of 'dry_run' (line 116)
    dry_run_302574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 27), 'dry_run', False)
    keyword_302575 = dry_run_302574
    kwargs_302576 = {'dry_run': keyword_302575}
    # Getting the type of 'spawn' (line 116)
    spawn_302572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'spawn', False)
    # Calling spawn(args, kwargs) (line 116)
    spawn_call_result_302577 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), spawn_302572, *[cmd_302573], **kwargs_302576)
    
    # Getting the type of 'compressed_name' (line 117)
    compressed_name_302578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 15), 'compressed_name')
    # Assigning a type to the variable 'stypy_return_type' (line 117)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'stypy_return_type', compressed_name_302578)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'archive_name' (line 119)
    archive_name_302579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'archive_name')
    # Assigning a type to the variable 'stypy_return_type' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type', archive_name_302579)
    
    # ################# End of 'make_tarball(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_tarball' in the type store
    # Getting the type of 'stypy_return_type' (line 51)
    stypy_return_type_302580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_302580)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_tarball'
    return stypy_return_type_302580

# Assigning a type to the variable 'make_tarball' (line 51)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'make_tarball', make_tarball)

@norecursion
def make_zipfile(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_302581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 46), 'int')
    int_302582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 57), 'int')
    defaults = [int_302581, int_302582]
    # Create a new context for function 'make_zipfile'
    module_type_store = module_type_store.open_function_context('make_zipfile', 121, 0, False)
    
    # Passed parameters checking function
    make_zipfile.stypy_localization = localization
    make_zipfile.stypy_type_of_self = None
    make_zipfile.stypy_type_store = module_type_store
    make_zipfile.stypy_function_name = 'make_zipfile'
    make_zipfile.stypy_param_names_list = ['base_name', 'base_dir', 'verbose', 'dry_run']
    make_zipfile.stypy_varargs_param_name = None
    make_zipfile.stypy_kwargs_param_name = None
    make_zipfile.stypy_call_defaults = defaults
    make_zipfile.stypy_call_varargs = varargs
    make_zipfile.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_zipfile', ['base_name', 'base_dir', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_zipfile', localization, ['base_name', 'base_dir', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_zipfile(...)' code ##################

    str_302583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, (-1)), 'str', 'Create a zip file from all the files under \'base_dir\'.\n\n    The output zip file will be named \'base_name\' + ".zip".  Uses either the\n    "zipfile" Python module (if available) or the InfoZIP "zip" utility\n    (if installed and found on the default search path).  If neither tool is\n    available, raises DistutilsExecError.  Returns the name of the output zip\n    file.\n    ')
    
    
    # SSA begins for try-except statement (line 130)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 131, 8))
    
    # 'import zipfile' statement (line 131)
    import zipfile

    import_module(stypy.reporting.localization.Localization(__file__, 131, 8), 'zipfile', zipfile, module_type_store)
    
    # SSA branch for the except part of a try statement (line 130)
    # SSA branch for the except 'ImportError' branch of a try statement (line 130)
    module_type_store.open_ssa_branch('except')
    
    # Assigning a Name to a Name (line 133):
    # Getting the type of 'None' (line 133)
    None_302584 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 18), 'None')
    # Assigning a type to the variable 'zipfile' (line 133)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'zipfile', None_302584)
    # SSA join for try-except statement (line 130)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a BinOp to a Name (line 135):
    # Getting the type of 'base_name' (line 135)
    base_name_302585 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 19), 'base_name')
    str_302586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 31), 'str', '.zip')
    # Applying the binary operator '+' (line 135)
    result_add_302587 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 19), '+', base_name_302585, str_302586)
    
    # Assigning a type to the variable 'zip_filename' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 4), 'zip_filename', result_add_302587)
    
    # Call to mkpath(...): (line 136)
    # Processing the call arguments (line 136)
    
    # Call to dirname(...): (line 136)
    # Processing the call arguments (line 136)
    # Getting the type of 'zip_filename' (line 136)
    zip_filename_302592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 27), 'zip_filename', False)
    # Processing the call keyword arguments (line 136)
    kwargs_302593 = {}
    # Getting the type of 'os' (line 136)
    os_302589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 136)
    path_302590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 11), os_302589, 'path')
    # Obtaining the member 'dirname' of a type (line 136)
    dirname_302591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 11), path_302590, 'dirname')
    # Calling dirname(args, kwargs) (line 136)
    dirname_call_result_302594 = invoke(stypy.reporting.localization.Localization(__file__, 136, 11), dirname_302591, *[zip_filename_302592], **kwargs_302593)
    
    # Processing the call keyword arguments (line 136)
    # Getting the type of 'dry_run' (line 136)
    dry_run_302595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 50), 'dry_run', False)
    keyword_302596 = dry_run_302595
    kwargs_302597 = {'dry_run': keyword_302596}
    # Getting the type of 'mkpath' (line 136)
    mkpath_302588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 4), 'mkpath', False)
    # Calling mkpath(args, kwargs) (line 136)
    mkpath_call_result_302598 = invoke(stypy.reporting.localization.Localization(__file__, 136, 4), mkpath_302588, *[dirname_call_result_302594], **kwargs_302597)
    
    
    # Type idiom detected: calculating its left and rigth part (line 140)
    # Getting the type of 'zipfile' (line 140)
    zipfile_302599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 7), 'zipfile')
    # Getting the type of 'None' (line 140)
    None_302600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 18), 'None')
    
    (may_be_302601, more_types_in_union_302602) = may_be_none(zipfile_302599, None_302600)

    if may_be_302601:

        if more_types_in_union_302602:
            # Runtime conditional SSA (line 140)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Getting the type of 'verbose' (line 141)
        verbose_302603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 11), 'verbose')
        # Testing the type of an if condition (line 141)
        if_condition_302604 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 141, 8), verbose_302603)
        # Assigning a type to the variable 'if_condition_302604' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'if_condition_302604', if_condition_302604)
        # SSA begins for if statement (line 141)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Str to a Name (line 142):
        str_302605 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 25), 'str', '-r')
        # Assigning a type to the variable 'zipoptions' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 12), 'zipoptions', str_302605)
        # SSA branch for the else part of an if statement (line 141)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Str to a Name (line 144):
        str_302606 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 25), 'str', '-rq')
        # Assigning a type to the variable 'zipoptions' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'zipoptions', str_302606)
        # SSA join for if statement (line 141)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # SSA begins for try-except statement (line 146)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to spawn(...): (line 147)
        # Processing the call arguments (line 147)
        
        # Obtaining an instance of the builtin type 'list' (line 147)
        list_302608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 147)
        # Adding element type (line 147)
        str_302609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 19), 'str', 'zip')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 18), list_302608, str_302609)
        # Adding element type (line 147)
        # Getting the type of 'zipoptions' (line 147)
        zipoptions_302610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 26), 'zipoptions', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 18), list_302608, zipoptions_302610)
        # Adding element type (line 147)
        # Getting the type of 'zip_filename' (line 147)
        zip_filename_302611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 38), 'zip_filename', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 18), list_302608, zip_filename_302611)
        # Adding element type (line 147)
        # Getting the type of 'base_dir' (line 147)
        base_dir_302612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 52), 'base_dir', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 18), list_302608, base_dir_302612)
        
        # Processing the call keyword arguments (line 147)
        # Getting the type of 'dry_run' (line 148)
        dry_run_302613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 148, 26), 'dry_run', False)
        keyword_302614 = dry_run_302613
        kwargs_302615 = {'dry_run': keyword_302614}
        # Getting the type of 'spawn' (line 147)
        spawn_302607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'spawn', False)
        # Calling spawn(args, kwargs) (line 147)
        spawn_call_result_302616 = invoke(stypy.reporting.localization.Localization(__file__, 147, 12), spawn_302607, *[list_302608], **kwargs_302615)
        
        # SSA branch for the except part of a try statement (line 146)
        # SSA branch for the except 'DistutilsExecError' branch of a try statement (line 146)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsExecError' (line 152)
        DistutilsExecError_302617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 18), 'DistutilsExecError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 152, 12), DistutilsExecError_302617, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 146)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_302602:
            # Runtime conditional SSA for else branch (line 140)
            module_type_store.open_ssa_branch('idiom else')



    if ((not may_be_302601) or more_types_in_union_302602):
        
        # Call to info(...): (line 158)
        # Processing the call arguments (line 158)
        str_302620 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 158, 17), 'str', "creating '%s' and adding '%s' to it")
        # Getting the type of 'zip_filename' (line 159)
        zip_filename_302621 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 17), 'zip_filename', False)
        # Getting the type of 'base_dir' (line 159)
        base_dir_302622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 31), 'base_dir', False)
        # Processing the call keyword arguments (line 158)
        kwargs_302623 = {}
        # Getting the type of 'log' (line 158)
        log_302618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 158)
        info_302619 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 8), log_302618, 'info')
        # Calling info(args, kwargs) (line 158)
        info_call_result_302624 = invoke(stypy.reporting.localization.Localization(__file__, 158, 8), info_302619, *[str_302620, zip_filename_302621, base_dir_302622], **kwargs_302623)
        
        
        
        # Getting the type of 'dry_run' (line 161)
        dry_run_302625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 161, 15), 'dry_run')
        # Applying the 'not' unary operator (line 161)
        result_not__302626 = python_operator(stypy.reporting.localization.Localization(__file__, 161, 11), 'not', dry_run_302625)
        
        # Testing the type of an if condition (line 161)
        if_condition_302627 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 161, 8), result_not__302626)
        # Assigning a type to the variable 'if_condition_302627' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'if_condition_302627', if_condition_302627)
        # SSA begins for if statement (line 161)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 162):
        
        # Call to ZipFile(...): (line 162)
        # Processing the call arguments (line 162)
        # Getting the type of 'zip_filename' (line 162)
        zip_filename_302630 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 34), 'zip_filename', False)
        str_302631 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 162, 48), 'str', 'w')
        # Processing the call keyword arguments (line 162)
        # Getting the type of 'zipfile' (line 163)
        zipfile_302632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 46), 'zipfile', False)
        # Obtaining the member 'ZIP_DEFLATED' of a type (line 163)
        ZIP_DEFLATED_302633 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 46), zipfile_302632, 'ZIP_DEFLATED')
        keyword_302634 = ZIP_DEFLATED_302633
        kwargs_302635 = {'compression': keyword_302634}
        # Getting the type of 'zipfile' (line 162)
        zipfile_302628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 18), 'zipfile', False)
        # Obtaining the member 'ZipFile' of a type (line 162)
        ZipFile_302629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 18), zipfile_302628, 'ZipFile')
        # Calling ZipFile(args, kwargs) (line 162)
        ZipFile_call_result_302636 = invoke(stypy.reporting.localization.Localization(__file__, 162, 18), ZipFile_302629, *[zip_filename_302630, str_302631], **kwargs_302635)
        
        # Assigning a type to the variable 'zip' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 12), 'zip', ZipFile_call_result_302636)
        
        
        # Call to walk(...): (line 165)
        # Processing the call arguments (line 165)
        # Getting the type of 'base_dir' (line 165)
        base_dir_302639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 56), 'base_dir', False)
        # Processing the call keyword arguments (line 165)
        kwargs_302640 = {}
        # Getting the type of 'os' (line 165)
        os_302637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 48), 'os', False)
        # Obtaining the member 'walk' of a type (line 165)
        walk_302638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 48), os_302637, 'walk')
        # Calling walk(args, kwargs) (line 165)
        walk_call_result_302641 = invoke(stypy.reporting.localization.Localization(__file__, 165, 48), walk_302638, *[base_dir_302639], **kwargs_302640)
        
        # Testing the type of a for loop iterable (line 165)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 165, 12), walk_call_result_302641)
        # Getting the type of the for loop variable (line 165)
        for_loop_var_302642 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 165, 12), walk_call_result_302641)
        # Assigning a type to the variable 'dirpath' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'dirpath', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 12), for_loop_var_302642))
        # Assigning a type to the variable 'dirnames' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'dirnames', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 12), for_loop_var_302642))
        # Assigning a type to the variable 'filenames' (line 165)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'filenames', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 165, 12), for_loop_var_302642))
        # SSA begins for a for statement (line 165)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Getting the type of 'filenames' (line 166)
        filenames_302643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 166, 28), 'filenames')
        # Testing the type of a for loop iterable (line 166)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 166, 16), filenames_302643)
        # Getting the type of the for loop variable (line 166)
        for_loop_var_302644 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 166, 16), filenames_302643)
        # Assigning a type to the variable 'name' (line 166)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 166, 16), 'name', for_loop_var_302644)
        # SSA begins for a for statement (line 166)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 167):
        
        # Call to normpath(...): (line 167)
        # Processing the call arguments (line 167)
        
        # Call to join(...): (line 167)
        # Processing the call arguments (line 167)
        # Getting the type of 'dirpath' (line 167)
        dirpath_302651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 57), 'dirpath', False)
        # Getting the type of 'name' (line 167)
        name_302652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 66), 'name', False)
        # Processing the call keyword arguments (line 167)
        kwargs_302653 = {}
        # Getting the type of 'os' (line 167)
        os_302648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 44), 'os', False)
        # Obtaining the member 'path' of a type (line 167)
        path_302649 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 44), os_302648, 'path')
        # Obtaining the member 'join' of a type (line 167)
        join_302650 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 44), path_302649, 'join')
        # Calling join(args, kwargs) (line 167)
        join_call_result_302654 = invoke(stypy.reporting.localization.Localization(__file__, 167, 44), join_302650, *[dirpath_302651, name_302652], **kwargs_302653)
        
        # Processing the call keyword arguments (line 167)
        kwargs_302655 = {}
        # Getting the type of 'os' (line 167)
        os_302645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 27), 'os', False)
        # Obtaining the member 'path' of a type (line 167)
        path_302646 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), os_302645, 'path')
        # Obtaining the member 'normpath' of a type (line 167)
        normpath_302647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 167, 27), path_302646, 'normpath')
        # Calling normpath(args, kwargs) (line 167)
        normpath_call_result_302656 = invoke(stypy.reporting.localization.Localization(__file__, 167, 27), normpath_302647, *[join_call_result_302654], **kwargs_302655)
        
        # Assigning a type to the variable 'path' (line 167)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 20), 'path', normpath_call_result_302656)
        
        
        # Call to isfile(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'path' (line 168)
        path_302660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 38), 'path', False)
        # Processing the call keyword arguments (line 168)
        kwargs_302661 = {}
        # Getting the type of 'os' (line 168)
        os_302657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 168)
        path_302658 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 23), os_302657, 'path')
        # Obtaining the member 'isfile' of a type (line 168)
        isfile_302659 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 23), path_302658, 'isfile')
        # Calling isfile(args, kwargs) (line 168)
        isfile_call_result_302662 = invoke(stypy.reporting.localization.Localization(__file__, 168, 23), isfile_302659, *[path_302660], **kwargs_302661)
        
        # Testing the type of an if condition (line 168)
        if_condition_302663 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 168, 20), isfile_call_result_302662)
        # Assigning a type to the variable 'if_condition_302663' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 20), 'if_condition_302663', if_condition_302663)
        # SSA begins for if statement (line 168)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to write(...): (line 169)
        # Processing the call arguments (line 169)
        # Getting the type of 'path' (line 169)
        path_302666 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 34), 'path', False)
        # Getting the type of 'path' (line 169)
        path_302667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 40), 'path', False)
        # Processing the call keyword arguments (line 169)
        kwargs_302668 = {}
        # Getting the type of 'zip' (line 169)
        zip_302664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 24), 'zip', False)
        # Obtaining the member 'write' of a type (line 169)
        write_302665 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 169, 24), zip_302664, 'write')
        # Calling write(args, kwargs) (line 169)
        write_call_result_302669 = invoke(stypy.reporting.localization.Localization(__file__, 169, 24), write_302665, *[path_302666, path_302667], **kwargs_302668)
        
        
        # Call to info(...): (line 170)
        # Processing the call arguments (line 170)
        str_302672 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 33), 'str', "adding '%s'")
        # Getting the type of 'path' (line 170)
        path_302673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 49), 'path', False)
        # Applying the binary operator '%' (line 170)
        result_mod_302674 = python_operator(stypy.reporting.localization.Localization(__file__, 170, 33), '%', str_302672, path_302673)
        
        # Processing the call keyword arguments (line 170)
        kwargs_302675 = {}
        # Getting the type of 'log' (line 170)
        log_302670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 24), 'log', False)
        # Obtaining the member 'info' of a type (line 170)
        info_302671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 24), log_302670, 'info')
        # Calling info(args, kwargs) (line 170)
        info_call_result_302676 = invoke(stypy.reporting.localization.Localization(__file__, 170, 24), info_302671, *[result_mod_302674], **kwargs_302675)
        
        # SSA join for if statement (line 168)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to close(...): (line 171)
        # Processing the call keyword arguments (line 171)
        kwargs_302679 = {}
        # Getting the type of 'zip' (line 171)
        zip_302677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'zip', False)
        # Obtaining the member 'close' of a type (line 171)
        close_302678 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 12), zip_302677, 'close')
        # Calling close(args, kwargs) (line 171)
        close_call_result_302680 = invoke(stypy.reporting.localization.Localization(__file__, 171, 12), close_302678, *[], **kwargs_302679)
        
        # SSA join for if statement (line 161)
        module_type_store = module_type_store.join_ssa_context()
        

        if (may_be_302601 and more_types_in_union_302602):
            # SSA join for if statement (line 140)
            module_type_store = module_type_store.join_ssa_context()


    
    # Getting the type of 'zip_filename' (line 173)
    zip_filename_302681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 11), 'zip_filename')
    # Assigning a type to the variable 'stypy_return_type' (line 173)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 173, 4), 'stypy_return_type', zip_filename_302681)
    
    # ################# End of 'make_zipfile(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_zipfile' in the type store
    # Getting the type of 'stypy_return_type' (line 121)
    stypy_return_type_302682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_302682)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_zipfile'
    return stypy_return_type_302682

# Assigning a type to the variable 'make_zipfile' (line 121)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 0), 'make_zipfile', make_zipfile)

# Assigning a Dict to a Name (line 175):

# Obtaining an instance of the builtin type 'dict' (line 175)
dict_302683 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 175, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 175)
# Adding element type (key, value) (line 175)
str_302684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 4), 'str', 'gztar')

# Obtaining an instance of the builtin type 'tuple' (line 176)
tuple_302685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 176)
# Adding element type (line 176)
# Getting the type of 'make_tarball' (line 176)
make_tarball_302686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 176, 14), 'make_tarball')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 14), tuple_302685, make_tarball_302686)
# Adding element type (line 176)

# Obtaining an instance of the builtin type 'list' (line 176)
list_302687 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 176)
# Adding element type (line 176)

# Obtaining an instance of the builtin type 'tuple' (line 176)
tuple_302688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 176)
# Adding element type (line 176)
str_302689 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 30), 'str', 'compress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 30), tuple_302688, str_302689)
# Adding element type (line 176)
str_302690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 42), 'str', 'gzip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 30), tuple_302688, str_302690)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 28), list_302687, tuple_302688)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 14), tuple_302685, list_302687)
# Adding element type (line 176)
str_302691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 176, 52), 'str', "gzip'ed tar-file")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 176, 14), tuple_302685, str_302691)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 18), dict_302683, (str_302684, tuple_302685))
# Adding element type (key, value) (line 175)
str_302692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 4), 'str', 'bztar')

# Obtaining an instance of the builtin type 'tuple' (line 177)
tuple_302693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 177)
# Adding element type (line 177)
# Getting the type of 'make_tarball' (line 177)
make_tarball_302694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 177, 14), 'make_tarball')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 14), tuple_302693, make_tarball_302694)
# Adding element type (line 177)

# Obtaining an instance of the builtin type 'list' (line 177)
list_302695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 177)
# Adding element type (line 177)

# Obtaining an instance of the builtin type 'tuple' (line 177)
tuple_302696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 177)
# Adding element type (line 177)
str_302697 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 30), 'str', 'compress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 30), tuple_302696, str_302697)
# Adding element type (line 177)
str_302698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 42), 'str', 'bzip2')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 30), tuple_302696, str_302698)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 28), list_302695, tuple_302696)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 14), tuple_302693, list_302695)
# Adding element type (line 177)
str_302699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 177, 53), 'str', "bzip2'ed tar-file")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 177, 14), tuple_302693, str_302699)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 18), dict_302683, (str_302692, tuple_302693))
# Adding element type (key, value) (line 175)
str_302700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 4), 'str', 'ztar')

# Obtaining an instance of the builtin type 'tuple' (line 178)
tuple_302701 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 178)
# Adding element type (line 178)
# Getting the type of 'make_tarball' (line 178)
make_tarball_302702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 14), 'make_tarball')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 14), tuple_302701, make_tarball_302702)
# Adding element type (line 178)

# Obtaining an instance of the builtin type 'list' (line 178)
list_302703 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 178)
# Adding element type (line 178)

# Obtaining an instance of the builtin type 'tuple' (line 178)
tuple_302704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 178)
# Adding element type (line 178)
str_302705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 30), 'str', 'compress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 30), tuple_302704, str_302705)
# Adding element type (line 178)
str_302706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 42), 'str', 'compress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 30), tuple_302704, str_302706)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 28), list_302703, tuple_302704)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 14), tuple_302701, list_302703)
# Adding element type (line 178)
str_302707 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 56), 'str', 'compressed tar file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 178, 14), tuple_302701, str_302707)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 18), dict_302683, (str_302700, tuple_302701))
# Adding element type (key, value) (line 175)
str_302708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 4), 'str', 'tar')

# Obtaining an instance of the builtin type 'tuple' (line 179)
tuple_302709 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 179)
# Adding element type (line 179)
# Getting the type of 'make_tarball' (line 179)
make_tarball_302710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 14), 'make_tarball')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 14), tuple_302709, make_tarball_302710)
# Adding element type (line 179)

# Obtaining an instance of the builtin type 'list' (line 179)
list_302711 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 179)
# Adding element type (line 179)

# Obtaining an instance of the builtin type 'tuple' (line 179)
tuple_302712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 30), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 179)
# Adding element type (line 179)
str_302713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 30), 'str', 'compress')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), tuple_302712, str_302713)
# Adding element type (line 179)
# Getting the type of 'None' (line 179)
None_302714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 42), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 30), tuple_302712, None_302714)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 28), list_302711, tuple_302712)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 14), tuple_302709, list_302711)
# Adding element type (line 179)
str_302715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 50), 'str', 'uncompressed tar file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 179, 14), tuple_302709, str_302715)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 18), dict_302683, (str_302708, tuple_302709))
# Adding element type (key, value) (line 175)
str_302716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 4), 'str', 'zip')

# Obtaining an instance of the builtin type 'tuple' (line 180)
tuple_302717 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 14), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 180)
# Adding element type (line 180)
# Getting the type of 'make_zipfile' (line 180)
make_zipfile_302718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 14), 'make_zipfile')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 14), tuple_302717, make_zipfile_302718)
# Adding element type (line 180)

# Obtaining an instance of the builtin type 'list' (line 180)
list_302719 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 28), 'list')
# Adding type elements to the builtin type 'list' instance (line 180)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 14), tuple_302717, list_302719)
# Adding element type (line 180)
str_302720 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 180, 31), 'str', 'ZIP file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 180, 14), tuple_302717, str_302720)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 175, 18), dict_302683, (str_302716, tuple_302717))

# Assigning a type to the variable 'ARCHIVE_FORMATS' (line 175)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 175, 0), 'ARCHIVE_FORMATS', dict_302683)

@norecursion
def check_archive_formats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'check_archive_formats'
    module_type_store = module_type_store.open_function_context('check_archive_formats', 183, 0, False)
    
    # Passed parameters checking function
    check_archive_formats.stypy_localization = localization
    check_archive_formats.stypy_type_of_self = None
    check_archive_formats.stypy_type_store = module_type_store
    check_archive_formats.stypy_function_name = 'check_archive_formats'
    check_archive_formats.stypy_param_names_list = ['formats']
    check_archive_formats.stypy_varargs_param_name = None
    check_archive_formats.stypy_kwargs_param_name = None
    check_archive_formats.stypy_call_defaults = defaults
    check_archive_formats.stypy_call_varargs = varargs
    check_archive_formats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'check_archive_formats', ['formats'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'check_archive_formats', localization, ['formats'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'check_archive_formats(...)' code ##################

    str_302721 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 187, (-1)), 'str', "Returns the first format from the 'format' list that is unknown.\n\n    If all formats are known, returns None\n    ")
    
    # Getting the type of 'formats' (line 188)
    formats_302722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 18), 'formats')
    # Testing the type of a for loop iterable (line 188)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 188, 4), formats_302722)
    # Getting the type of the for loop variable (line 188)
    for_loop_var_302723 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 188, 4), formats_302722)
    # Assigning a type to the variable 'format' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 4), 'format', for_loop_var_302723)
    # SSA begins for a for statement (line 188)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    
    # Getting the type of 'format' (line 189)
    format_302724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'format')
    # Getting the type of 'ARCHIVE_FORMATS' (line 189)
    ARCHIVE_FORMATS_302725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'ARCHIVE_FORMATS')
    # Applying the binary operator 'notin' (line 189)
    result_contains_302726 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 11), 'notin', format_302724, ARCHIVE_FORMATS_302725)
    
    # Testing the type of an if condition (line 189)
    if_condition_302727 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 8), result_contains_302726)
    # Assigning a type to the variable 'if_condition_302727' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 8), 'if_condition_302727', if_condition_302727)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'format' (line 190)
    format_302728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 19), 'format')
    # Assigning a type to the variable 'stypy_return_type' (line 190)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 190, 12), 'stypy_return_type', format_302728)
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'None' (line 191)
    None_302729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 11), 'None')
    # Assigning a type to the variable 'stypy_return_type' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 4), 'stypy_return_type', None_302729)
    
    # ################# End of 'check_archive_formats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'check_archive_formats' in the type store
    # Getting the type of 'stypy_return_type' (line 183)
    stypy_return_type_302730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_302730)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'check_archive_formats'
    return stypy_return_type_302730

# Assigning a type to the variable 'check_archive_formats' (line 183)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 0), 'check_archive_formats', check_archive_formats)

@norecursion
def make_archive(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    # Getting the type of 'None' (line 193)
    None_302731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 45), 'None')
    # Getting the type of 'None' (line 193)
    None_302732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 60), 'None')
    int_302733 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 193, 74), 'int')
    int_302734 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 194, 25), 'int')
    # Getting the type of 'None' (line 194)
    None_302735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 34), 'None')
    # Getting the type of 'None' (line 194)
    None_302736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 46), 'None')
    defaults = [None_302731, None_302732, int_302733, int_302734, None_302735, None_302736]
    # Create a new context for function 'make_archive'
    module_type_store = module_type_store.open_function_context('make_archive', 193, 0, False)
    
    # Passed parameters checking function
    make_archive.stypy_localization = localization
    make_archive.stypy_type_of_self = None
    make_archive.stypy_type_store = module_type_store
    make_archive.stypy_function_name = 'make_archive'
    make_archive.stypy_param_names_list = ['base_name', 'format', 'root_dir', 'base_dir', 'verbose', 'dry_run', 'owner', 'group']
    make_archive.stypy_varargs_param_name = None
    make_archive.stypy_kwargs_param_name = None
    make_archive.stypy_call_defaults = defaults
    make_archive.stypy_call_varargs = varargs
    make_archive.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'make_archive', ['base_name', 'format', 'root_dir', 'base_dir', 'verbose', 'dry_run', 'owner', 'group'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'make_archive', localization, ['base_name', 'format', 'root_dir', 'base_dir', 'verbose', 'dry_run', 'owner', 'group'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'make_archive(...)' code ##################

    str_302737 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 210, (-1)), 'str', 'Create an archive file (eg. zip or tar).\n\n    \'base_name\' is the name of the file to create, minus any format-specific\n    extension; \'format\' is the archive format: one of "zip", "tar", "ztar",\n    or "gztar".\n\n    \'root_dir\' is a directory that will be the root directory of the\n    archive; ie. we typically chdir into \'root_dir\' before creating the\n    archive.  \'base_dir\' is the directory where we start archiving from;\n    ie. \'base_dir\' will be the common prefix of all files and\n    directories in the archive.  \'root_dir\' and \'base_dir\' both default\n    to the current directory.  Returns the name of the archive file.\n\n    \'owner\' and \'group\' are used when creating a tar archive. By default,\n    uses the current owner and group.\n    ')
    
    # Assigning a Call to a Name (line 211):
    
    # Call to getcwd(...): (line 211)
    # Processing the call keyword arguments (line 211)
    kwargs_302740 = {}
    # Getting the type of 'os' (line 211)
    os_302738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 15), 'os', False)
    # Obtaining the member 'getcwd' of a type (line 211)
    getcwd_302739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 211, 15), os_302738, 'getcwd')
    # Calling getcwd(args, kwargs) (line 211)
    getcwd_call_result_302741 = invoke(stypy.reporting.localization.Localization(__file__, 211, 15), getcwd_302739, *[], **kwargs_302740)
    
    # Assigning a type to the variable 'save_cwd' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 4), 'save_cwd', getcwd_call_result_302741)
    
    # Type idiom detected: calculating its left and rigth part (line 212)
    # Getting the type of 'root_dir' (line 212)
    root_dir_302742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 4), 'root_dir')
    # Getting the type of 'None' (line 212)
    None_302743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 23), 'None')
    
    (may_be_302744, more_types_in_union_302745) = may_not_be_none(root_dir_302742, None_302743)

    if may_be_302744:

        if more_types_in_union_302745:
            # Runtime conditional SSA (line 212)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to debug(...): (line 213)
        # Processing the call arguments (line 213)
        str_302748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 18), 'str', "changing into '%s'")
        # Getting the type of 'root_dir' (line 213)
        root_dir_302749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 40), 'root_dir', False)
        # Processing the call keyword arguments (line 213)
        kwargs_302750 = {}
        # Getting the type of 'log' (line 213)
        log_302746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 8), 'log', False)
        # Obtaining the member 'debug' of a type (line 213)
        debug_302747 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 213, 8), log_302746, 'debug')
        # Calling debug(args, kwargs) (line 213)
        debug_call_result_302751 = invoke(stypy.reporting.localization.Localization(__file__, 213, 8), debug_302747, *[str_302748, root_dir_302749], **kwargs_302750)
        
        
        # Assigning a Call to a Name (line 214):
        
        # Call to abspath(...): (line 214)
        # Processing the call arguments (line 214)
        # Getting the type of 'base_name' (line 214)
        base_name_302755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 36), 'base_name', False)
        # Processing the call keyword arguments (line 214)
        kwargs_302756 = {}
        # Getting the type of 'os' (line 214)
        os_302752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 214)
        path_302753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), os_302752, 'path')
        # Obtaining the member 'abspath' of a type (line 214)
        abspath_302754 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), path_302753, 'abspath')
        # Calling abspath(args, kwargs) (line 214)
        abspath_call_result_302757 = invoke(stypy.reporting.localization.Localization(__file__, 214, 20), abspath_302754, *[base_name_302755], **kwargs_302756)
        
        # Assigning a type to the variable 'base_name' (line 214)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 8), 'base_name', abspath_call_result_302757)
        
        
        # Getting the type of 'dry_run' (line 215)
        dry_run_302758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 15), 'dry_run')
        # Applying the 'not' unary operator (line 215)
        result_not__302759 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 11), 'not', dry_run_302758)
        
        # Testing the type of an if condition (line 215)
        if_condition_302760 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 8), result_not__302759)
        # Assigning a type to the variable 'if_condition_302760' (line 215)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'if_condition_302760', if_condition_302760)
        # SSA begins for if statement (line 215)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to chdir(...): (line 216)
        # Processing the call arguments (line 216)
        # Getting the type of 'root_dir' (line 216)
        root_dir_302763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 21), 'root_dir', False)
        # Processing the call keyword arguments (line 216)
        kwargs_302764 = {}
        # Getting the type of 'os' (line 216)
        os_302761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 216)
        chdir_302762 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 12), os_302761, 'chdir')
        # Calling chdir(args, kwargs) (line 216)
        chdir_call_result_302765 = invoke(stypy.reporting.localization.Localization(__file__, 216, 12), chdir_302762, *[root_dir_302763], **kwargs_302764)
        
        # SSA join for if statement (line 215)
        module_type_store = module_type_store.join_ssa_context()
        

        if more_types_in_union_302745:
            # SSA join for if statement (line 212)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Type idiom detected: calculating its left and rigth part (line 218)
    # Getting the type of 'base_dir' (line 218)
    base_dir_302766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 7), 'base_dir')
    # Getting the type of 'None' (line 218)
    None_302767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 19), 'None')
    
    (may_be_302768, more_types_in_union_302769) = may_be_none(base_dir_302766, None_302767)

    if may_be_302768:

        if more_types_in_union_302769:
            # Runtime conditional SSA (line 218)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Assigning a Attribute to a Name (line 219):
        # Getting the type of 'os' (line 219)
        os_302770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 19), 'os')
        # Obtaining the member 'curdir' of a type (line 219)
        curdir_302771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 219, 19), os_302770, 'curdir')
        # Assigning a type to the variable 'base_dir' (line 219)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 8), 'base_dir', curdir_302771)

        if more_types_in_union_302769:
            # SSA join for if statement (line 218)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Dict to a Name (line 221):
    
    # Obtaining an instance of the builtin type 'dict' (line 221)
    dict_302772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 13), 'dict')
    # Adding type elements to the builtin type 'dict' instance (line 221)
    # Adding element type (key, value) (line 221)
    str_302773 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 14), 'str', 'dry_run')
    # Getting the type of 'dry_run' (line 221)
    dry_run_302774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 25), 'dry_run')
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 221, 13), dict_302772, (str_302773, dry_run_302774))
    
    # Assigning a type to the variable 'kwargs' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 4), 'kwargs', dict_302772)
    
    
    # SSA begins for try-except statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 224):
    
    # Obtaining the type of the subscript
    # Getting the type of 'format' (line 224)
    format_302775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 38), 'format')
    # Getting the type of 'ARCHIVE_FORMATS' (line 224)
    ARCHIVE_FORMATS_302776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 22), 'ARCHIVE_FORMATS')
    # Obtaining the member '__getitem__' of a type (line 224)
    getitem___302777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 22), ARCHIVE_FORMATS_302776, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 224)
    subscript_call_result_302778 = invoke(stypy.reporting.localization.Localization(__file__, 224, 22), getitem___302777, format_302775)
    
    # Assigning a type to the variable 'format_info' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 8), 'format_info', subscript_call_result_302778)
    # SSA branch for the except part of a try statement (line 223)
    # SSA branch for the except 'KeyError' branch of a try statement (line 223)
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'ValueError' (line 226)
    ValueError_302779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 14), 'ValueError')
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 226, 8), ValueError_302779, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Subscript to a Name (line 228):
    
    # Obtaining the type of the subscript
    int_302780 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 228, 23), 'int')
    # Getting the type of 'format_info' (line 228)
    format_info_302781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 11), 'format_info')
    # Obtaining the member '__getitem__' of a type (line 228)
    getitem___302782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 11), format_info_302781, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 228)
    subscript_call_result_302783 = invoke(stypy.reporting.localization.Localization(__file__, 228, 11), getitem___302782, int_302780)
    
    # Assigning a type to the variable 'func' (line 228)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 228, 4), 'func', subscript_call_result_302783)
    
    
    # Obtaining the type of the subscript
    int_302784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 229, 32), 'int')
    # Getting the type of 'format_info' (line 229)
    format_info_302785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'format_info')
    # Obtaining the member '__getitem__' of a type (line 229)
    getitem___302786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 229, 20), format_info_302785, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 229)
    subscript_call_result_302787 = invoke(stypy.reporting.localization.Localization(__file__, 229, 20), getitem___302786, int_302784)
    
    # Testing the type of a for loop iterable (line 229)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 229, 4), subscript_call_result_302787)
    # Getting the type of the for loop variable (line 229)
    for_loop_var_302788 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 229, 4), subscript_call_result_302787)
    # Assigning a type to the variable 'arg' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'arg', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 4), for_loop_var_302788))
    # Assigning a type to the variable 'val' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 4), 'val', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 229, 4), for_loop_var_302788))
    # SSA begins for a for statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Assigning a Name to a Subscript (line 230):
    # Getting the type of 'val' (line 230)
    val_302789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 22), 'val')
    # Getting the type of 'kwargs' (line 230)
    kwargs_302790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 8), 'kwargs')
    # Getting the type of 'arg' (line 230)
    arg_302791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 15), 'arg')
    # Storing an element on a container (line 230)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 230, 8), kwargs_302790, (arg_302791, val_302789))
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'format' (line 232)
    format_302792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 232, 7), 'format')
    str_302793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 232, 17), 'str', 'zip')
    # Applying the binary operator '!=' (line 232)
    result_ne_302794 = python_operator(stypy.reporting.localization.Localization(__file__, 232, 7), '!=', format_302792, str_302793)
    
    # Testing the type of an if condition (line 232)
    if_condition_302795 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 232, 4), result_ne_302794)
    # Assigning a type to the variable 'if_condition_302795' (line 232)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 232, 4), 'if_condition_302795', if_condition_302795)
    # SSA begins for if statement (line 232)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Subscript (line 233):
    # Getting the type of 'owner' (line 233)
    owner_302796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 26), 'owner')
    # Getting the type of 'kwargs' (line 233)
    kwargs_302797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 233, 8), 'kwargs')
    str_302798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, 15), 'str', 'owner')
    # Storing an element on a container (line 233)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 233, 8), kwargs_302797, (str_302798, owner_302796))
    
    # Assigning a Name to a Subscript (line 234):
    # Getting the type of 'group' (line 234)
    group_302799 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 26), 'group')
    # Getting the type of 'kwargs' (line 234)
    kwargs_302800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'kwargs')
    str_302801 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 15), 'str', 'group')
    # Storing an element on a container (line 234)
    set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 234, 8), kwargs_302800, (str_302801, group_302799))
    # SSA join for if statement (line 232)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Try-finally block (line 236)
    
    # Assigning a Call to a Name (line 237):
    
    # Call to func(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'base_name' (line 237)
    base_name_302803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 24), 'base_name', False)
    # Getting the type of 'base_dir' (line 237)
    base_dir_302804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 35), 'base_dir', False)
    # Processing the call keyword arguments (line 237)
    # Getting the type of 'kwargs' (line 237)
    kwargs_302805 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 47), 'kwargs', False)
    kwargs_302806 = {'kwargs_302805': kwargs_302805}
    # Getting the type of 'func' (line 237)
    func_302802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 19), 'func', False)
    # Calling func(args, kwargs) (line 237)
    func_call_result_302807 = invoke(stypy.reporting.localization.Localization(__file__, 237, 19), func_302802, *[base_name_302803, base_dir_302804], **kwargs_302806)
    
    # Assigning a type to the variable 'filename' (line 237)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 237, 8), 'filename', func_call_result_302807)
    
    # finally branch of the try-finally block (line 236)
    
    # Type idiom detected: calculating its left and rigth part (line 239)
    # Getting the type of 'root_dir' (line 239)
    root_dir_302808 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'root_dir')
    # Getting the type of 'None' (line 239)
    None_302809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 27), 'None')
    
    (may_be_302810, more_types_in_union_302811) = may_not_be_none(root_dir_302808, None_302809)

    if may_be_302810:

        if more_types_in_union_302811:
            # Runtime conditional SSA (line 239)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to debug(...): (line 240)
        # Processing the call arguments (line 240)
        str_302814 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 240, 22), 'str', "changing back to '%s'")
        # Getting the type of 'save_cwd' (line 240)
        save_cwd_302815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 47), 'save_cwd', False)
        # Processing the call keyword arguments (line 240)
        kwargs_302816 = {}
        # Getting the type of 'log' (line 240)
        log_302812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 240)
        debug_302813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), log_302812, 'debug')
        # Calling debug(args, kwargs) (line 240)
        debug_call_result_302817 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), debug_302813, *[str_302814, save_cwd_302815], **kwargs_302816)
        
        
        # Call to chdir(...): (line 241)
        # Processing the call arguments (line 241)
        # Getting the type of 'save_cwd' (line 241)
        save_cwd_302820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 21), 'save_cwd', False)
        # Processing the call keyword arguments (line 241)
        kwargs_302821 = {}
        # Getting the type of 'os' (line 241)
        os_302818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 241, 12), 'os', False)
        # Obtaining the member 'chdir' of a type (line 241)
        chdir_302819 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 241, 12), os_302818, 'chdir')
        # Calling chdir(args, kwargs) (line 241)
        chdir_call_result_302822 = invoke(stypy.reporting.localization.Localization(__file__, 241, 12), chdir_302819, *[save_cwd_302820], **kwargs_302821)
        

        if more_types_in_union_302811:
            # SSA join for if statement (line 239)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Getting the type of 'filename' (line 243)
    filename_302823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 243, 11), 'filename')
    # Assigning a type to the variable 'stypy_return_type' (line 243)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 243, 4), 'stypy_return_type', filename_302823)
    
    # ################# End of 'make_archive(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'make_archive' in the type store
    # Getting the type of 'stypy_return_type' (line 193)
    stypy_return_type_302824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_302824)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'make_archive'
    return stypy_return_type_302824

# Assigning a type to the variable 'make_archive' (line 193)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 0), 'make_archive', make_archive)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
