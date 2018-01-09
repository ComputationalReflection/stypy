
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.file_util
2: 
3: Utility functions for operating on single files.
4: '''
5: 
6: __revision__ = "$Id$"
7: 
8: import os
9: from distutils.errors import DistutilsFileError
10: from distutils import log
11: 
12: # for generating verbose output in 'copy_file()'
13: _copy_action = {None: 'copying',
14:                 'hard': 'hard linking',
15:                 'sym': 'symbolically linking'}
16: 
17: 
18: def _copy_file_contents(src, dst, buffer_size=16*1024):
19:     '''Copy the file 'src' to 'dst'.
20: 
21:     Both must be filenames. Any error opening either file, reading from
22:     'src', or writing to 'dst', raises DistutilsFileError.  Data is
23:     read/written in chunks of 'buffer_size' bytes (default 16k).  No attempt
24:     is made to handle anything apart from regular files.
25:     '''
26:     # Stolen from shutil module in the standard library, but with
27:     # custom error-handling added.
28:     fsrc = None
29:     fdst = None
30:     try:
31:         try:
32:             fsrc = open(src, 'rb')
33:         except os.error, (errno, errstr):
34:             raise DistutilsFileError("could not open '%s': %s" % (src, errstr))
35: 
36:         if os.path.exists(dst):
37:             try:
38:                 os.unlink(dst)
39:             except os.error, (errno, errstr):
40:                 raise DistutilsFileError(
41:                       "could not delete '%s': %s" % (dst, errstr))
42: 
43:         try:
44:             fdst = open(dst, 'wb')
45:         except os.error, (errno, errstr):
46:             raise DistutilsFileError(
47:                   "could not create '%s': %s" % (dst, errstr))
48: 
49:         while 1:
50:             try:
51:                 buf = fsrc.read(buffer_size)
52:             except os.error, (errno, errstr):
53:                 raise DistutilsFileError(
54:                       "could not read from '%s': %s" % (src, errstr))
55: 
56:             if not buf:
57:                 break
58: 
59:             try:
60:                 fdst.write(buf)
61:             except os.error, (errno, errstr):
62:                 raise DistutilsFileError(
63:                       "could not write to '%s': %s" % (dst, errstr))
64: 
65:     finally:
66:         if fdst:
67:             fdst.close()
68:         if fsrc:
69:             fsrc.close()
70: 
71: def copy_file(src, dst, preserve_mode=1, preserve_times=1, update=0,
72:               link=None, verbose=1, dry_run=0):
73:     '''Copy a file 'src' to 'dst'.
74: 
75:     If 'dst' is a directory, then 'src' is copied there with the same name;
76:     otherwise, it must be a filename.  (If the file exists, it will be
77:     ruthlessly clobbered.)  If 'preserve_mode' is true (the default),
78:     the file's mode (type and permission bits, or whatever is analogous on
79:     the current platform) is copied.  If 'preserve_times' is true (the
80:     default), the last-modified and last-access times are copied as well.
81:     If 'update' is true, 'src' will only be copied if 'dst' does not exist,
82:     or if 'dst' does exist but is older than 'src'.
83: 
84:     'link' allows you to make hard links (os.link) or symbolic links
85:     (os.symlink) instead of copying: set it to "hard" or "sym"; if it is
86:     None (the default), files are copied.  Don't set 'link' on systems that
87:     don't support it: 'copy_file()' doesn't check if hard or symbolic
88:     linking is available. If hardlink fails, falls back to
89:     _copy_file_contents().
90: 
91:     Under Mac OS, uses the native file copy function in macostools; on
92:     other systems, uses '_copy_file_contents()' to copy file contents.
93: 
94:     Return a tuple (dest_name, copied): 'dest_name' is the actual name of
95:     the output file, and 'copied' is true if the file was copied (or would
96:     have been copied, if 'dry_run' true).
97:     '''
98:     # XXX if the destination file already exists, we clobber it if
99:     # copying, but blow up if linking.  Hmmm.  And I don't know what
100:     # macostools.copyfile() does.  Should definitely be consistent, and
101:     # should probably blow up if destination exists and we would be
102:     # changing it (ie. it's not already a hard/soft link to src OR
103:     # (not update) and (src newer than dst).
104: 
105:     from distutils.dep_util import newer
106:     from stat import ST_ATIME, ST_MTIME, ST_MODE, S_IMODE
107: 
108:     if not os.path.isfile(src):
109:         raise DistutilsFileError(
110:               "can't copy '%s': doesn't exist or not a regular file" % src)
111: 
112:     if os.path.isdir(dst):
113:         dir = dst
114:         dst = os.path.join(dst, os.path.basename(src))
115:     else:
116:         dir = os.path.dirname(dst)
117: 
118:     if update and not newer(src, dst):
119:         if verbose >= 1:
120:             log.debug("not copying %s (output up-to-date)", src)
121:         return dst, 0
122: 
123:     try:
124:         action = _copy_action[link]
125:     except KeyError:
126:         raise ValueError("invalid value '%s' for 'link' argument" % link)
127: 
128:     if verbose >= 1:
129:         if os.path.basename(dst) == os.path.basename(src):
130:             log.info("%s %s -> %s", action, src, dir)
131:         else:
132:             log.info("%s %s -> %s", action, src, dst)
133: 
134:     if dry_run:
135:         return (dst, 1)
136: 
137:     # If linking (hard or symbolic), use the appropriate system call
138:     # (Unix only, of course, but that's the caller's responsibility)
139:     if link == 'hard':
140:         if not (os.path.exists(dst) and os.path.samefile(src, dst)):
141:             try:
142:                 os.link(src, dst)
143:                 return (dst, 1)
144:             except OSError:
145:                 # If hard linking fails, fall back on copying file
146:                 # (some special filesystems don't support hard linking
147:                 #  even under Unix, see issue #8876).
148:                 pass
149:     elif link == 'sym':
150:         if not (os.path.exists(dst) and os.path.samefile(src, dst)):
151:             os.symlink(src, dst)
152:             return (dst, 1)
153: 
154:     # Otherwise (non-Mac, not linking), copy the file contents and
155:     # (optionally) copy the times and mode.
156:     _copy_file_contents(src, dst)
157:     if preserve_mode or preserve_times:
158:         st = os.stat(src)
159: 
160:         # According to David Ascher <da@ski.org>, utime() should be done
161:         # before chmod() (at least under NT).
162:         if preserve_times:
163:             os.utime(dst, (st[ST_ATIME], st[ST_MTIME]))
164:         if preserve_mode:
165:             os.chmod(dst, S_IMODE(st[ST_MODE]))
166: 
167:     return (dst, 1)
168: 
169: # XXX I suspect this is Unix-specific -- need porting help!
170: def move_file (src, dst, verbose=1, dry_run=0):
171:     '''Move a file 'src' to 'dst'.
172: 
173:     If 'dst' is a directory, the file will be moved into it with the same
174:     name; otherwise, 'src' is just renamed to 'dst'.  Return the new
175:     full name of the file.
176: 
177:     Handles cross-device moves on Unix using 'copy_file()'.  What about
178:     other systems???
179:     '''
180:     from os.path import exists, isfile, isdir, basename, dirname
181:     import errno
182: 
183:     if verbose >= 1:
184:         log.info("moving %s -> %s", src, dst)
185: 
186:     if dry_run:
187:         return dst
188: 
189:     if not isfile(src):
190:         raise DistutilsFileError("can't move '%s': not a regular file" % src)
191: 
192:     if isdir(dst):
193:         dst = os.path.join(dst, basename(src))
194:     elif exists(dst):
195:         raise DistutilsFileError(
196:               "can't move '%s': destination '%s' already exists" %
197:               (src, dst))
198: 
199:     if not isdir(dirname(dst)):
200:         raise DistutilsFileError(
201:               "can't move '%s': destination '%s' not a valid path" % \
202:               (src, dst))
203: 
204:     copy_it = 0
205:     try:
206:         os.rename(src, dst)
207:     except os.error, (num, msg):
208:         if num == errno.EXDEV:
209:             copy_it = 1
210:         else:
211:             raise DistutilsFileError(
212:                   "couldn't move '%s' to '%s': %s" % (src, dst, msg))
213: 
214:     if copy_it:
215:         copy_file(src, dst, verbose=verbose)
216:         try:
217:             os.unlink(src)
218:         except os.error, (num, msg):
219:             try:
220:                 os.unlink(dst)
221:             except os.error:
222:                 pass
223:             raise DistutilsFileError(
224:                   ("couldn't move '%s' to '%s' by copy/delete: " +
225:                    "delete '%s' failed: %s") %
226:                   (src, dst, src, msg))
227:     return dst
228: 
229: 
230: def write_file (filename, contents):
231:     '''Create a file with the specified name and write 'contents' (a
232:     sequence of strings without line terminators) to it.
233:     '''
234:     f = open(filename, "w")
235:     try:
236:         for line in contents:
237:             f.write(line + "\n")
238:     finally:
239:         f.close()
240: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_2214 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', 'distutils.file_util\n\nUtility functions for operating on single files.\n')

# Assigning a Str to a Name (line 6):
str_2215 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_2215)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsFileError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/')
import_2216 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_2216) is not StypyTypeError):

    if (import_2216 != 'pyd_module'):
        __import__(import_2216)
        sys_modules_2217 = sys.modules[import_2216]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_2217.module_type_store, module_type_store, ['DistutilsFileError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_2217, sys_modules_2217.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsFileError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsFileError'], [DistutilsFileError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_2216)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils import log' statement (line 10)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils', None, module_type_store, ['log'], [log])


# Assigning a Dict to a Name (line 13):

# Obtaining an instance of the builtin type 'dict' (line 13)
dict_2218 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 15), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 13)
# Adding element type (key, value) (line 13)
# Getting the type of 'None' (line 13)
None_2219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 16), 'None')
str_2220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'str', 'copying')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), dict_2218, (None_2219, str_2220))
# Adding element type (key, value) (line 13)
str_2221 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 16), 'str', 'hard')
str_2222 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 24), 'str', 'hard linking')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), dict_2218, (str_2221, str_2222))
# Adding element type (key, value) (line 13)
str_2223 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 16), 'str', 'sym')
str_2224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 23), 'str', 'symbolically linking')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 15), dict_2218, (str_2223, str_2224))

# Assigning a type to the variable '_copy_action' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), '_copy_action', dict_2218)

@norecursion
def _copy_file_contents(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'int')
    int_2226 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 49), 'int')
    # Applying the binary operator '*' (line 18)
    result_mul_2227 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 46), '*', int_2225, int_2226)
    
    defaults = [result_mul_2227]
    # Create a new context for function '_copy_file_contents'
    module_type_store = module_type_store.open_function_context('_copy_file_contents', 18, 0, False)
    
    # Passed parameters checking function
    _copy_file_contents.stypy_localization = localization
    _copy_file_contents.stypy_type_of_self = None
    _copy_file_contents.stypy_type_store = module_type_store
    _copy_file_contents.stypy_function_name = '_copy_file_contents'
    _copy_file_contents.stypy_param_names_list = ['src', 'dst', 'buffer_size']
    _copy_file_contents.stypy_varargs_param_name = None
    _copy_file_contents.stypy_kwargs_param_name = None
    _copy_file_contents.stypy_call_defaults = defaults
    _copy_file_contents.stypy_call_varargs = varargs
    _copy_file_contents.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_copy_file_contents', ['src', 'dst', 'buffer_size'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, '_copy_file_contents', localization, ['src', 'dst', 'buffer_size'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of '_copy_file_contents(...)' code ##################

    str_2228 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, (-1)), 'str', "Copy the file 'src' to 'dst'.\n\n    Both must be filenames. Any error opening either file, reading from\n    'src', or writing to 'dst', raises DistutilsFileError.  Data is\n    read/written in chunks of 'buffer_size' bytes (default 16k).  No attempt\n    is made to handle anything apart from regular files.\n    ")
    
    # Assigning a Name to a Name (line 28):
    # Getting the type of 'None' (line 28)
    None_2229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 11), 'None')
    # Assigning a type to the variable 'fsrc' (line 28)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'fsrc', None_2229)
    
    # Assigning a Name to a Name (line 29):
    # Getting the type of 'None' (line 29)
    None_2230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 11), 'None')
    # Assigning a type to the variable 'fdst' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'fdst', None_2230)
    
    # Try-finally block (line 30)
    
    
    # SSA begins for try-except statement (line 31)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 32):
    
    # Call to open(...): (line 32)
    # Processing the call arguments (line 32)
    # Getting the type of 'src' (line 32)
    src_2232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 24), 'src', False)
    str_2233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 29), 'str', 'rb')
    # Processing the call keyword arguments (line 32)
    kwargs_2234 = {}
    # Getting the type of 'open' (line 32)
    open_2231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 19), 'open', False)
    # Calling open(args, kwargs) (line 32)
    open_call_result_2235 = invoke(stypy.reporting.localization.Localization(__file__, 32, 19), open_2231, *[src_2232, str_2233], **kwargs_2234)
    
    # Assigning a type to the variable 'fsrc' (line 32)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'fsrc', open_call_result_2235)
    # SSA branch for the except part of a try statement (line 31)
    # SSA branch for the except 'Attribute' branch of a try statement (line 31)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'os' (line 33)
    os_2236 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'os')
    # Obtaining the member 'error' of a type (line 33)
    error_2237 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 15), os_2236, 'error')
    # Assigning a type to the variable 'errno' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'errno', error_2237)
    # Assigning a type to the variable 'errstr' (line 33)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'errstr', error_2237)
    
    # Call to DistutilsFileError(...): (line 34)
    # Processing the call arguments (line 34)
    str_2239 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 37), 'str', "could not open '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 34)
    tuple_2240 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 66), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 34)
    # Adding element type (line 34)
    # Getting the type of 'src' (line 34)
    src_2241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 66), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 66), tuple_2240, src_2241)
    # Adding element type (line 34)
    # Getting the type of 'errstr' (line 34)
    errstr_2242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 71), 'errstr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 66), tuple_2240, errstr_2242)
    
    # Applying the binary operator '%' (line 34)
    result_mod_2243 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 37), '%', str_2239, tuple_2240)
    
    # Processing the call keyword arguments (line 34)
    kwargs_2244 = {}
    # Getting the type of 'DistutilsFileError' (line 34)
    DistutilsFileError_2238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 18), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 34)
    DistutilsFileError_call_result_2245 = invoke(stypy.reporting.localization.Localization(__file__, 34, 18), DistutilsFileError_2238, *[result_mod_2243], **kwargs_2244)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 34, 12), DistutilsFileError_call_result_2245, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 31)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to exists(...): (line 36)
    # Processing the call arguments (line 36)
    # Getting the type of 'dst' (line 36)
    dst_2249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 26), 'dst', False)
    # Processing the call keyword arguments (line 36)
    kwargs_2250 = {}
    # Getting the type of 'os' (line 36)
    os_2246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 36)
    path_2247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 11), os_2246, 'path')
    # Obtaining the member 'exists' of a type (line 36)
    exists_2248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 11), path_2247, 'exists')
    # Calling exists(args, kwargs) (line 36)
    exists_call_result_2251 = invoke(stypy.reporting.localization.Localization(__file__, 36, 11), exists_2248, *[dst_2249], **kwargs_2250)
    
    # Testing the type of an if condition (line 36)
    if_condition_2252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 8), exists_call_result_2251)
    # Assigning a type to the variable 'if_condition_2252' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'if_condition_2252', if_condition_2252)
    # SSA begins for if statement (line 36)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 37)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to unlink(...): (line 38)
    # Processing the call arguments (line 38)
    # Getting the type of 'dst' (line 38)
    dst_2255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 26), 'dst', False)
    # Processing the call keyword arguments (line 38)
    kwargs_2256 = {}
    # Getting the type of 'os' (line 38)
    os_2253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 16), 'os', False)
    # Obtaining the member 'unlink' of a type (line 38)
    unlink_2254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 16), os_2253, 'unlink')
    # Calling unlink(args, kwargs) (line 38)
    unlink_call_result_2257 = invoke(stypy.reporting.localization.Localization(__file__, 38, 16), unlink_2254, *[dst_2255], **kwargs_2256)
    
    # SSA branch for the except part of a try statement (line 37)
    # SSA branch for the except 'Attribute' branch of a try statement (line 37)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'os' (line 39)
    os_2258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 19), 'os')
    # Obtaining the member 'error' of a type (line 39)
    error_2259 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 19), os_2258, 'error')
    # Assigning a type to the variable 'errno' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'errno', error_2259)
    # Assigning a type to the variable 'errstr' (line 39)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'errstr', error_2259)
    
    # Call to DistutilsFileError(...): (line 40)
    # Processing the call arguments (line 40)
    str_2261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'str', "could not delete '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 41)
    tuple_2262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 53), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 41)
    # Adding element type (line 41)
    # Getting the type of 'dst' (line 41)
    dst_2263 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 53), 'dst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 53), tuple_2262, dst_2263)
    # Adding element type (line 41)
    # Getting the type of 'errstr' (line 41)
    errstr_2264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 58), 'errstr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 53), tuple_2262, errstr_2264)
    
    # Applying the binary operator '%' (line 41)
    result_mod_2265 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 22), '%', str_2261, tuple_2262)
    
    # Processing the call keyword arguments (line 40)
    kwargs_2266 = {}
    # Getting the type of 'DistutilsFileError' (line 40)
    DistutilsFileError_2260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 22), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 40)
    DistutilsFileError_call_result_2267 = invoke(stypy.reporting.localization.Localization(__file__, 40, 22), DistutilsFileError_2260, *[result_mod_2265], **kwargs_2266)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 40, 16), DistutilsFileError_call_result_2267, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 37)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 36)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 43)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 44):
    
    # Call to open(...): (line 44)
    # Processing the call arguments (line 44)
    # Getting the type of 'dst' (line 44)
    dst_2269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 24), 'dst', False)
    str_2270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 29), 'str', 'wb')
    # Processing the call keyword arguments (line 44)
    kwargs_2271 = {}
    # Getting the type of 'open' (line 44)
    open_2268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 19), 'open', False)
    # Calling open(args, kwargs) (line 44)
    open_call_result_2272 = invoke(stypy.reporting.localization.Localization(__file__, 44, 19), open_2268, *[dst_2269, str_2270], **kwargs_2271)
    
    # Assigning a type to the variable 'fdst' (line 44)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'fdst', open_call_result_2272)
    # SSA branch for the except part of a try statement (line 43)
    # SSA branch for the except 'Attribute' branch of a try statement (line 43)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'os' (line 45)
    os_2273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 15), 'os')
    # Obtaining the member 'error' of a type (line 45)
    error_2274 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 15), os_2273, 'error')
    # Assigning a type to the variable 'errno' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'errno', error_2274)
    # Assigning a type to the variable 'errstr' (line 45)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'errstr', error_2274)
    
    # Call to DistutilsFileError(...): (line 46)
    # Processing the call arguments (line 46)
    str_2276 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 18), 'str', "could not create '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 47)
    tuple_2277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 49), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 47)
    # Adding element type (line 47)
    # Getting the type of 'dst' (line 47)
    dst_2278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 49), 'dst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 49), tuple_2277, dst_2278)
    # Adding element type (line 47)
    # Getting the type of 'errstr' (line 47)
    errstr_2279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 54), 'errstr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 49), tuple_2277, errstr_2279)
    
    # Applying the binary operator '%' (line 47)
    result_mod_2280 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 18), '%', str_2276, tuple_2277)
    
    # Processing the call keyword arguments (line 46)
    kwargs_2281 = {}
    # Getting the type of 'DistutilsFileError' (line 46)
    DistutilsFileError_2275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 18), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 46)
    DistutilsFileError_call_result_2282 = invoke(stypy.reporting.localization.Localization(__file__, 46, 18), DistutilsFileError_2275, *[result_mod_2280], **kwargs_2281)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 46, 12), DistutilsFileError_call_result_2282, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 43)
    module_type_store = module_type_store.join_ssa_context()
    
    
    int_2283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 14), 'int')
    # Testing the type of an if condition (line 49)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 49, 8), int_2283)
    # SSA begins for while statement (line 49)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    
    # SSA begins for try-except statement (line 50)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Call to a Name (line 51):
    
    # Call to read(...): (line 51)
    # Processing the call arguments (line 51)
    # Getting the type of 'buffer_size' (line 51)
    buffer_size_2286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'buffer_size', False)
    # Processing the call keyword arguments (line 51)
    kwargs_2287 = {}
    # Getting the type of 'fsrc' (line 51)
    fsrc_2284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 22), 'fsrc', False)
    # Obtaining the member 'read' of a type (line 51)
    read_2285 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 22), fsrc_2284, 'read')
    # Calling read(args, kwargs) (line 51)
    read_call_result_2288 = invoke(stypy.reporting.localization.Localization(__file__, 51, 22), read_2285, *[buffer_size_2286], **kwargs_2287)
    
    # Assigning a type to the variable 'buf' (line 51)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'buf', read_call_result_2288)
    # SSA branch for the except part of a try statement (line 50)
    # SSA branch for the except 'Attribute' branch of a try statement (line 50)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'os' (line 52)
    os_2289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'os')
    # Obtaining the member 'error' of a type (line 52)
    error_2290 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 19), os_2289, 'error')
    # Assigning a type to the variable 'errno' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'errno', error_2290)
    # Assigning a type to the variable 'errstr' (line 52)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 12), 'errstr', error_2290)
    
    # Call to DistutilsFileError(...): (line 53)
    # Processing the call arguments (line 53)
    str_2292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 22), 'str', "could not read from '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 54)
    tuple_2293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 56), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 54)
    # Adding element type (line 54)
    # Getting the type of 'src' (line 54)
    src_2294 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 56), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 56), tuple_2293, src_2294)
    # Adding element type (line 54)
    # Getting the type of 'errstr' (line 54)
    errstr_2295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 61), 'errstr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 56), tuple_2293, errstr_2295)
    
    # Applying the binary operator '%' (line 54)
    result_mod_2296 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 22), '%', str_2292, tuple_2293)
    
    # Processing the call keyword arguments (line 53)
    kwargs_2297 = {}
    # Getting the type of 'DistutilsFileError' (line 53)
    DistutilsFileError_2291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 22), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 53)
    DistutilsFileError_call_result_2298 = invoke(stypy.reporting.localization.Localization(__file__, 53, 22), DistutilsFileError_2291, *[result_mod_2296], **kwargs_2297)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 53, 16), DistutilsFileError_call_result_2298, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 50)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'buf' (line 56)
    buf_2299 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 19), 'buf')
    # Applying the 'not' unary operator (line 56)
    result_not__2300 = python_operator(stypy.reporting.localization.Localization(__file__, 56, 15), 'not', buf_2299)
    
    # Testing the type of an if condition (line 56)
    if_condition_2301 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 56, 12), result_not__2300)
    # Assigning a type to the variable 'if_condition_2301' (line 56)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'if_condition_2301', if_condition_2301)
    # SSA begins for if statement (line 56)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # SSA join for if statement (line 56)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 59)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to write(...): (line 60)
    # Processing the call arguments (line 60)
    # Getting the type of 'buf' (line 60)
    buf_2304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 27), 'buf', False)
    # Processing the call keyword arguments (line 60)
    kwargs_2305 = {}
    # Getting the type of 'fdst' (line 60)
    fdst_2302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'fdst', False)
    # Obtaining the member 'write' of a type (line 60)
    write_2303 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), fdst_2302, 'write')
    # Calling write(args, kwargs) (line 60)
    write_call_result_2306 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), write_2303, *[buf_2304], **kwargs_2305)
    
    # SSA branch for the except part of a try statement (line 59)
    # SSA branch for the except 'Attribute' branch of a try statement (line 59)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'os' (line 61)
    os_2307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 19), 'os')
    # Obtaining the member 'error' of a type (line 61)
    error_2308 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 19), os_2307, 'error')
    # Assigning a type to the variable 'errno' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'errno', error_2308)
    # Assigning a type to the variable 'errstr' (line 61)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'errstr', error_2308)
    
    # Call to DistutilsFileError(...): (line 62)
    # Processing the call arguments (line 62)
    str_2310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'str', "could not write to '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 63)
    tuple_2311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 55), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 63)
    # Adding element type (line 63)
    # Getting the type of 'dst' (line 63)
    dst_2312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 55), 'dst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 55), tuple_2311, dst_2312)
    # Adding element type (line 63)
    # Getting the type of 'errstr' (line 63)
    errstr_2313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 60), 'errstr', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 55), tuple_2311, errstr_2313)
    
    # Applying the binary operator '%' (line 63)
    result_mod_2314 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 22), '%', str_2310, tuple_2311)
    
    # Processing the call keyword arguments (line 62)
    kwargs_2315 = {}
    # Getting the type of 'DistutilsFileError' (line 62)
    DistutilsFileError_2309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 22), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 62)
    DistutilsFileError_call_result_2316 = invoke(stypy.reporting.localization.Localization(__file__, 62, 22), DistutilsFileError_2309, *[result_mod_2314], **kwargs_2315)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 62, 16), DistutilsFileError_call_result_2316, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 59)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for while statement (line 49)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 30)
    
    # Getting the type of 'fdst' (line 66)
    fdst_2317 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 11), 'fdst')
    # Testing the type of an if condition (line 66)
    if_condition_2318 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 66, 8), fdst_2317)
    # Assigning a type to the variable 'if_condition_2318' (line 66)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'if_condition_2318', if_condition_2318)
    # SSA begins for if statement (line 66)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 67)
    # Processing the call keyword arguments (line 67)
    kwargs_2321 = {}
    # Getting the type of 'fdst' (line 67)
    fdst_2319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'fdst', False)
    # Obtaining the member 'close' of a type (line 67)
    close_2320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 12), fdst_2319, 'close')
    # Calling close(args, kwargs) (line 67)
    close_call_result_2322 = invoke(stypy.reporting.localization.Localization(__file__, 67, 12), close_2320, *[], **kwargs_2321)
    
    # SSA join for if statement (line 66)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'fsrc' (line 68)
    fsrc_2323 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'fsrc')
    # Testing the type of an if condition (line 68)
    if_condition_2324 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 8), fsrc_2323)
    # Assigning a type to the variable 'if_condition_2324' (line 68)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'if_condition_2324', if_condition_2324)
    # SSA begins for if statement (line 68)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to close(...): (line 69)
    # Processing the call keyword arguments (line 69)
    kwargs_2327 = {}
    # Getting the type of 'fsrc' (line 69)
    fsrc_2325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'fsrc', False)
    # Obtaining the member 'close' of a type (line 69)
    close_2326 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 12), fsrc_2325, 'close')
    # Calling close(args, kwargs) (line 69)
    close_call_result_2328 = invoke(stypy.reporting.localization.Localization(__file__, 69, 12), close_2326, *[], **kwargs_2327)
    
    # SSA join for if statement (line 68)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # ################# End of '_copy_file_contents(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function '_copy_file_contents' in the type store
    # Getting the type of 'stypy_return_type' (line 18)
    stypy_return_type_2329 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2329)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_copy_file_contents'
    return stypy_return_type_2329

# Assigning a type to the variable '_copy_file_contents' (line 18)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), '_copy_file_contents', _copy_file_contents)

@norecursion
def copy_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 38), 'int')
    int_2331 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 56), 'int')
    int_2332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 66), 'int')
    # Getting the type of 'None' (line 72)
    None_2333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'None')
    int_2334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 33), 'int')
    int_2335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 44), 'int')
    defaults = [int_2330, int_2331, int_2332, None_2333, int_2334, int_2335]
    # Create a new context for function 'copy_file'
    module_type_store = module_type_store.open_function_context('copy_file', 71, 0, False)
    
    # Passed parameters checking function
    copy_file.stypy_localization = localization
    copy_file.stypy_type_of_self = None
    copy_file.stypy_type_store = module_type_store
    copy_file.stypy_function_name = 'copy_file'
    copy_file.stypy_param_names_list = ['src', 'dst', 'preserve_mode', 'preserve_times', 'update', 'link', 'verbose', 'dry_run']
    copy_file.stypy_varargs_param_name = None
    copy_file.stypy_kwargs_param_name = None
    copy_file.stypy_call_defaults = defaults
    copy_file.stypy_call_varargs = varargs
    copy_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'copy_file', ['src', 'dst', 'preserve_mode', 'preserve_times', 'update', 'link', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'copy_file', localization, ['src', 'dst', 'preserve_mode', 'preserve_times', 'update', 'link', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'copy_file(...)' code ##################

    str_2336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 97, (-1)), 'str', 'Copy a file \'src\' to \'dst\'.\n\n    If \'dst\' is a directory, then \'src\' is copied there with the same name;\n    otherwise, it must be a filename.  (If the file exists, it will be\n    ruthlessly clobbered.)  If \'preserve_mode\' is true (the default),\n    the file\'s mode (type and permission bits, or whatever is analogous on\n    the current platform) is copied.  If \'preserve_times\' is true (the\n    default), the last-modified and last-access times are copied as well.\n    If \'update\' is true, \'src\' will only be copied if \'dst\' does not exist,\n    or if \'dst\' does exist but is older than \'src\'.\n\n    \'link\' allows you to make hard links (os.link) or symbolic links\n    (os.symlink) instead of copying: set it to "hard" or "sym"; if it is\n    None (the default), files are copied.  Don\'t set \'link\' on systems that\n    don\'t support it: \'copy_file()\' doesn\'t check if hard or symbolic\n    linking is available. If hardlink fails, falls back to\n    _copy_file_contents().\n\n    Under Mac OS, uses the native file copy function in macostools; on\n    other systems, uses \'_copy_file_contents()\' to copy file contents.\n\n    Return a tuple (dest_name, copied): \'dest_name\' is the actual name of\n    the output file, and \'copied\' is true if the file was copied (or would\n    have been copied, if \'dry_run\' true).\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 105, 4))
    
    # 'from distutils.dep_util import newer' statement (line 105)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_2337 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 105, 4), 'distutils.dep_util')

    if (type(import_2337) is not StypyTypeError):

        if (import_2337 != 'pyd_module'):
            __import__(import_2337)
            sys_modules_2338 = sys.modules[import_2337]
            import_from_module(stypy.reporting.localization.Localization(__file__, 105, 4), 'distutils.dep_util', sys_modules_2338.module_type_store, module_type_store, ['newer'])
            nest_module(stypy.reporting.localization.Localization(__file__, 105, 4), __file__, sys_modules_2338, sys_modules_2338.module_type_store, module_type_store)
        else:
            from distutils.dep_util import newer

            import_from_module(stypy.reporting.localization.Localization(__file__, 105, 4), 'distutils.dep_util', None, module_type_store, ['newer'], [newer])

    else:
        # Assigning a type to the variable 'distutils.dep_util' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 4), 'distutils.dep_util', import_2337)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 106, 4))
    
    # 'from stat import ST_ATIME, ST_MTIME, ST_MODE, S_IMODE' statement (line 106)
    try:
        from stat import ST_ATIME, ST_MTIME, ST_MODE, S_IMODE

    except:
        ST_ATIME = UndefinedType
        ST_MTIME = UndefinedType
        ST_MODE = UndefinedType
        S_IMODE = UndefinedType
    import_from_module(stypy.reporting.localization.Localization(__file__, 106, 4), 'stat', None, module_type_store, ['ST_ATIME', 'ST_MTIME', 'ST_MODE', 'S_IMODE'], [ST_ATIME, ST_MTIME, ST_MODE, S_IMODE])
    
    
    
    
    # Call to isfile(...): (line 108)
    # Processing the call arguments (line 108)
    # Getting the type of 'src' (line 108)
    src_2342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'src', False)
    # Processing the call keyword arguments (line 108)
    kwargs_2343 = {}
    # Getting the type of 'os' (line 108)
    os_2339 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 108)
    path_2340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), os_2339, 'path')
    # Obtaining the member 'isfile' of a type (line 108)
    isfile_2341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), path_2340, 'isfile')
    # Calling isfile(args, kwargs) (line 108)
    isfile_call_result_2344 = invoke(stypy.reporting.localization.Localization(__file__, 108, 11), isfile_2341, *[src_2342], **kwargs_2343)
    
    # Applying the 'not' unary operator (line 108)
    result_not__2345 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 7), 'not', isfile_call_result_2344)
    
    # Testing the type of an if condition (line 108)
    if_condition_2346 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 4), result_not__2345)
    # Assigning a type to the variable 'if_condition_2346' (line 108)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 4), 'if_condition_2346', if_condition_2346)
    # SSA begins for if statement (line 108)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to DistutilsFileError(...): (line 109)
    # Processing the call arguments (line 109)
    str_2348 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 14), 'str', "can't copy '%s': doesn't exist or not a regular file")
    # Getting the type of 'src' (line 110)
    src_2349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 71), 'src', False)
    # Applying the binary operator '%' (line 110)
    result_mod_2350 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 14), '%', str_2348, src_2349)
    
    # Processing the call keyword arguments (line 109)
    kwargs_2351 = {}
    # Getting the type of 'DistutilsFileError' (line 109)
    DistutilsFileError_2347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 14), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 109)
    DistutilsFileError_call_result_2352 = invoke(stypy.reporting.localization.Localization(__file__, 109, 14), DistutilsFileError_2347, *[result_mod_2350], **kwargs_2351)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 109, 8), DistutilsFileError_call_result_2352, 'raise parameter', BaseException)
    # SSA join for if statement (line 108)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isdir(...): (line 112)
    # Processing the call arguments (line 112)
    # Getting the type of 'dst' (line 112)
    dst_2356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 21), 'dst', False)
    # Processing the call keyword arguments (line 112)
    kwargs_2357 = {}
    # Getting the type of 'os' (line 112)
    os_2353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 7), 'os', False)
    # Obtaining the member 'path' of a type (line 112)
    path_2354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 7), os_2353, 'path')
    # Obtaining the member 'isdir' of a type (line 112)
    isdir_2355 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 7), path_2354, 'isdir')
    # Calling isdir(args, kwargs) (line 112)
    isdir_call_result_2358 = invoke(stypy.reporting.localization.Localization(__file__, 112, 7), isdir_2355, *[dst_2356], **kwargs_2357)
    
    # Testing the type of an if condition (line 112)
    if_condition_2359 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 4), isdir_call_result_2358)
    # Assigning a type to the variable 'if_condition_2359' (line 112)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'if_condition_2359', if_condition_2359)
    # SSA begins for if statement (line 112)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Name to a Name (line 113):
    # Getting the type of 'dst' (line 113)
    dst_2360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 14), 'dst')
    # Assigning a type to the variable 'dir' (line 113)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'dir', dst_2360)
    
    # Assigning a Call to a Name (line 114):
    
    # Call to join(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'dst' (line 114)
    dst_2364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 27), 'dst', False)
    
    # Call to basename(...): (line 114)
    # Processing the call arguments (line 114)
    # Getting the type of 'src' (line 114)
    src_2368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 49), 'src', False)
    # Processing the call keyword arguments (line 114)
    kwargs_2369 = {}
    # Getting the type of 'os' (line 114)
    os_2365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 32), 'os', False)
    # Obtaining the member 'path' of a type (line 114)
    path_2366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 32), os_2365, 'path')
    # Obtaining the member 'basename' of a type (line 114)
    basename_2367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 32), path_2366, 'basename')
    # Calling basename(args, kwargs) (line 114)
    basename_call_result_2370 = invoke(stypy.reporting.localization.Localization(__file__, 114, 32), basename_2367, *[src_2368], **kwargs_2369)
    
    # Processing the call keyword arguments (line 114)
    kwargs_2371 = {}
    # Getting the type of 'os' (line 114)
    os_2361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 114)
    path_2362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 14), os_2361, 'path')
    # Obtaining the member 'join' of a type (line 114)
    join_2363 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 14), path_2362, 'join')
    # Calling join(args, kwargs) (line 114)
    join_call_result_2372 = invoke(stypy.reporting.localization.Localization(__file__, 114, 14), join_2363, *[dst_2364, basename_call_result_2370], **kwargs_2371)
    
    # Assigning a type to the variable 'dst' (line 114)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'dst', join_call_result_2372)
    # SSA branch for the else part of an if statement (line 112)
    module_type_store.open_ssa_branch('else')
    
    # Assigning a Call to a Name (line 116):
    
    # Call to dirname(...): (line 116)
    # Processing the call arguments (line 116)
    # Getting the type of 'dst' (line 116)
    dst_2376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 30), 'dst', False)
    # Processing the call keyword arguments (line 116)
    kwargs_2377 = {}
    # Getting the type of 'os' (line 116)
    os_2373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 116)
    path_2374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 14), os_2373, 'path')
    # Obtaining the member 'dirname' of a type (line 116)
    dirname_2375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 14), path_2374, 'dirname')
    # Calling dirname(args, kwargs) (line 116)
    dirname_call_result_2378 = invoke(stypy.reporting.localization.Localization(__file__, 116, 14), dirname_2375, *[dst_2376], **kwargs_2377)
    
    # Assigning a type to the variable 'dir' (line 116)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'dir', dirname_call_result_2378)
    # SSA join for if statement (line 112)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'update' (line 118)
    update_2379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 7), 'update')
    
    
    # Call to newer(...): (line 118)
    # Processing the call arguments (line 118)
    # Getting the type of 'src' (line 118)
    src_2381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'src', False)
    # Getting the type of 'dst' (line 118)
    dst_2382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 33), 'dst', False)
    # Processing the call keyword arguments (line 118)
    kwargs_2383 = {}
    # Getting the type of 'newer' (line 118)
    newer_2380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 22), 'newer', False)
    # Calling newer(args, kwargs) (line 118)
    newer_call_result_2384 = invoke(stypy.reporting.localization.Localization(__file__, 118, 22), newer_2380, *[src_2381, dst_2382], **kwargs_2383)
    
    # Applying the 'not' unary operator (line 118)
    result_not__2385 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 18), 'not', newer_call_result_2384)
    
    # Applying the binary operator 'and' (line 118)
    result_and_keyword_2386 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 7), 'and', update_2379, result_not__2385)
    
    # Testing the type of an if condition (line 118)
    if_condition_2387 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 4), result_and_keyword_2386)
    # Assigning a type to the variable 'if_condition_2387' (line 118)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 4), 'if_condition_2387', if_condition_2387)
    # SSA begins for if statement (line 118)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # Getting the type of 'verbose' (line 119)
    verbose_2388 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 11), 'verbose')
    int_2389 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 22), 'int')
    # Applying the binary operator '>=' (line 119)
    result_ge_2390 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 11), '>=', verbose_2388, int_2389)
    
    # Testing the type of an if condition (line 119)
    if_condition_2391 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 119, 8), result_ge_2390)
    # Assigning a type to the variable 'if_condition_2391' (line 119)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'if_condition_2391', if_condition_2391)
    # SSA begins for if statement (line 119)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to debug(...): (line 120)
    # Processing the call arguments (line 120)
    str_2394 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 22), 'str', 'not copying %s (output up-to-date)')
    # Getting the type of 'src' (line 120)
    src_2395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 60), 'src', False)
    # Processing the call keyword arguments (line 120)
    kwargs_2396 = {}
    # Getting the type of 'log' (line 120)
    log_2392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 12), 'log', False)
    # Obtaining the member 'debug' of a type (line 120)
    debug_2393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 12), log_2392, 'debug')
    # Calling debug(args, kwargs) (line 120)
    debug_call_result_2397 = invoke(stypy.reporting.localization.Localization(__file__, 120, 12), debug_2393, *[str_2394, src_2395], **kwargs_2396)
    
    # SSA join for if statement (line 119)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 121)
    tuple_2398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 121)
    # Adding element type (line 121)
    # Getting the type of 'dst' (line 121)
    dst_2399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 15), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 15), tuple_2398, dst_2399)
    # Adding element type (line 121)
    int_2400 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 20), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 121, 15), tuple_2398, int_2400)
    
    # Assigning a type to the variable 'stypy_return_type' (line 121)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', tuple_2398)
    # SSA join for if statement (line 118)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # SSA begins for try-except statement (line 123)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Assigning a Subscript to a Name (line 124):
    
    # Obtaining the type of the subscript
    # Getting the type of 'link' (line 124)
    link_2401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 30), 'link')
    # Getting the type of '_copy_action' (line 124)
    _copy_action_2402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 17), '_copy_action')
    # Obtaining the member '__getitem__' of a type (line 124)
    getitem___2403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 17), _copy_action_2402, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 124)
    subscript_call_result_2404 = invoke(stypy.reporting.localization.Localization(__file__, 124, 17), getitem___2403, link_2401)
    
    # Assigning a type to the variable 'action' (line 124)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'action', subscript_call_result_2404)
    # SSA branch for the except part of a try statement (line 123)
    # SSA branch for the except 'KeyError' branch of a try statement (line 123)
    module_type_store.open_ssa_branch('except')
    
    # Call to ValueError(...): (line 126)
    # Processing the call arguments (line 126)
    str_2406 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 25), 'str', "invalid value '%s' for 'link' argument")
    # Getting the type of 'link' (line 126)
    link_2407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 68), 'link', False)
    # Applying the binary operator '%' (line 126)
    result_mod_2408 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 25), '%', str_2406, link_2407)
    
    # Processing the call keyword arguments (line 126)
    kwargs_2409 = {}
    # Getting the type of 'ValueError' (line 126)
    ValueError_2405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 14), 'ValueError', False)
    # Calling ValueError(args, kwargs) (line 126)
    ValueError_call_result_2410 = invoke(stypy.reporting.localization.Localization(__file__, 126, 14), ValueError_2405, *[result_mod_2408], **kwargs_2409)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 126, 8), ValueError_call_result_2410, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 123)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'verbose' (line 128)
    verbose_2411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 7), 'verbose')
    int_2412 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, 18), 'int')
    # Applying the binary operator '>=' (line 128)
    result_ge_2413 = python_operator(stypy.reporting.localization.Localization(__file__, 128, 7), '>=', verbose_2411, int_2412)
    
    # Testing the type of an if condition (line 128)
    if_condition_2414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 4), result_ge_2413)
    # Assigning a type to the variable 'if_condition_2414' (line 128)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 4), 'if_condition_2414', if_condition_2414)
    # SSA begins for if statement (line 128)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Call to basename(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'dst' (line 129)
    dst_2418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'dst', False)
    # Processing the call keyword arguments (line 129)
    kwargs_2419 = {}
    # Getting the type of 'os' (line 129)
    os_2415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 11), 'os', False)
    # Obtaining the member 'path' of a type (line 129)
    path_2416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 11), os_2415, 'path')
    # Obtaining the member 'basename' of a type (line 129)
    basename_2417 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 11), path_2416, 'basename')
    # Calling basename(args, kwargs) (line 129)
    basename_call_result_2420 = invoke(stypy.reporting.localization.Localization(__file__, 129, 11), basename_2417, *[dst_2418], **kwargs_2419)
    
    
    # Call to basename(...): (line 129)
    # Processing the call arguments (line 129)
    # Getting the type of 'src' (line 129)
    src_2424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 53), 'src', False)
    # Processing the call keyword arguments (line 129)
    kwargs_2425 = {}
    # Getting the type of 'os' (line 129)
    os_2421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 36), 'os', False)
    # Obtaining the member 'path' of a type (line 129)
    path_2422 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 36), os_2421, 'path')
    # Obtaining the member 'basename' of a type (line 129)
    basename_2423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 36), path_2422, 'basename')
    # Calling basename(args, kwargs) (line 129)
    basename_call_result_2426 = invoke(stypy.reporting.localization.Localization(__file__, 129, 36), basename_2423, *[src_2424], **kwargs_2425)
    
    # Applying the binary operator '==' (line 129)
    result_eq_2427 = python_operator(stypy.reporting.localization.Localization(__file__, 129, 11), '==', basename_call_result_2420, basename_call_result_2426)
    
    # Testing the type of an if condition (line 129)
    if_condition_2428 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 129, 8), result_eq_2427)
    # Assigning a type to the variable 'if_condition_2428' (line 129)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'if_condition_2428', if_condition_2428)
    # SSA begins for if statement (line 129)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to info(...): (line 130)
    # Processing the call arguments (line 130)
    str_2431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 21), 'str', '%s %s -> %s')
    # Getting the type of 'action' (line 130)
    action_2432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 36), 'action', False)
    # Getting the type of 'src' (line 130)
    src_2433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 44), 'src', False)
    # Getting the type of 'dir' (line 130)
    dir_2434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 49), 'dir', False)
    # Processing the call keyword arguments (line 130)
    kwargs_2435 = {}
    # Getting the type of 'log' (line 130)
    log_2429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'log', False)
    # Obtaining the member 'info' of a type (line 130)
    info_2430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 12), log_2429, 'info')
    # Calling info(args, kwargs) (line 130)
    info_call_result_2436 = invoke(stypy.reporting.localization.Localization(__file__, 130, 12), info_2430, *[str_2431, action_2432, src_2433, dir_2434], **kwargs_2435)
    
    # SSA branch for the else part of an if statement (line 129)
    module_type_store.open_ssa_branch('else')
    
    # Call to info(...): (line 132)
    # Processing the call arguments (line 132)
    str_2439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 132, 21), 'str', '%s %s -> %s')
    # Getting the type of 'action' (line 132)
    action_2440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 36), 'action', False)
    # Getting the type of 'src' (line 132)
    src_2441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 44), 'src', False)
    # Getting the type of 'dst' (line 132)
    dst_2442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 49), 'dst', False)
    # Processing the call keyword arguments (line 132)
    kwargs_2443 = {}
    # Getting the type of 'log' (line 132)
    log_2437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'log', False)
    # Obtaining the member 'info' of a type (line 132)
    info_2438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 12), log_2437, 'info')
    # Calling info(args, kwargs) (line 132)
    info_call_result_2444 = invoke(stypy.reporting.localization.Localization(__file__, 132, 12), info_2438, *[str_2439, action_2440, src_2441, dst_2442], **kwargs_2443)
    
    # SSA join for if statement (line 129)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 128)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'dry_run' (line 134)
    dry_run_2445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 7), 'dry_run')
    # Testing the type of an if condition (line 134)
    if_condition_2446 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 4), dry_run_2445)
    # Assigning a type to the variable 'if_condition_2446' (line 134)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'if_condition_2446', if_condition_2446)
    # SSA begins for if statement (line 134)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Obtaining an instance of the builtin type 'tuple' (line 135)
    tuple_2447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 16), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 135)
    # Adding element type (line 135)
    # Getting the type of 'dst' (line 135)
    dst_2448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 16), tuple_2447, dst_2448)
    # Adding element type (line 135)
    int_2449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 21), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 135, 16), tuple_2447, int_2449)
    
    # Assigning a type to the variable 'stypy_return_type' (line 135)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 135, 8), 'stypy_return_type', tuple_2447)
    # SSA join for if statement (line 134)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Getting the type of 'link' (line 139)
    link_2450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 7), 'link')
    str_2451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 139, 15), 'str', 'hard')
    # Applying the binary operator '==' (line 139)
    result_eq_2452 = python_operator(stypy.reporting.localization.Localization(__file__, 139, 7), '==', link_2450, str_2451)
    
    # Testing the type of an if condition (line 139)
    if_condition_2453 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 139, 4), result_eq_2452)
    # Assigning a type to the variable 'if_condition_2453' (line 139)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 4), 'if_condition_2453', if_condition_2453)
    # SSA begins for if statement (line 139)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Evaluating a boolean operation
    
    # Call to exists(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'dst' (line 140)
    dst_2457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 31), 'dst', False)
    # Processing the call keyword arguments (line 140)
    kwargs_2458 = {}
    # Getting the type of 'os' (line 140)
    os_2454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 140)
    path_2455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), os_2454, 'path')
    # Obtaining the member 'exists' of a type (line 140)
    exists_2456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), path_2455, 'exists')
    # Calling exists(args, kwargs) (line 140)
    exists_call_result_2459 = invoke(stypy.reporting.localization.Localization(__file__, 140, 16), exists_2456, *[dst_2457], **kwargs_2458)
    
    
    # Call to samefile(...): (line 140)
    # Processing the call arguments (line 140)
    # Getting the type of 'src' (line 140)
    src_2463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 57), 'src', False)
    # Getting the type of 'dst' (line 140)
    dst_2464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 62), 'dst', False)
    # Processing the call keyword arguments (line 140)
    kwargs_2465 = {}
    # Getting the type of 'os' (line 140)
    os_2460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 40), 'os', False)
    # Obtaining the member 'path' of a type (line 140)
    path_2461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 40), os_2460, 'path')
    # Obtaining the member 'samefile' of a type (line 140)
    samefile_2462 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 40), path_2461, 'samefile')
    # Calling samefile(args, kwargs) (line 140)
    samefile_call_result_2466 = invoke(stypy.reporting.localization.Localization(__file__, 140, 40), samefile_2462, *[src_2463, dst_2464], **kwargs_2465)
    
    # Applying the binary operator 'and' (line 140)
    result_and_keyword_2467 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 16), 'and', exists_call_result_2459, samefile_call_result_2466)
    
    # Applying the 'not' unary operator (line 140)
    result_not__2468 = python_operator(stypy.reporting.localization.Localization(__file__, 140, 11), 'not', result_and_keyword_2467)
    
    # Testing the type of an if condition (line 140)
    if_condition_2469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 140, 8), result_not__2468)
    # Assigning a type to the variable 'if_condition_2469' (line 140)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 8), 'if_condition_2469', if_condition_2469)
    # SSA begins for if statement (line 140)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    # SSA begins for try-except statement (line 141)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to link(...): (line 142)
    # Processing the call arguments (line 142)
    # Getting the type of 'src' (line 142)
    src_2472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 24), 'src', False)
    # Getting the type of 'dst' (line 142)
    dst_2473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'dst', False)
    # Processing the call keyword arguments (line 142)
    kwargs_2474 = {}
    # Getting the type of 'os' (line 142)
    os_2470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 16), 'os', False)
    # Obtaining the member 'link' of a type (line 142)
    link_2471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 16), os_2470, 'link')
    # Calling link(args, kwargs) (line 142)
    link_call_result_2475 = invoke(stypy.reporting.localization.Localization(__file__, 142, 16), link_2471, *[src_2472, dst_2473], **kwargs_2474)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 143)
    tuple_2476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 143)
    # Adding element type (line 143)
    # Getting the type of 'dst' (line 143)
    dst_2477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 143, 24), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 24), tuple_2476, dst_2477)
    # Adding element type (line 143)
    int_2478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, 29), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 143, 24), tuple_2476, int_2478)
    
    # Assigning a type to the variable 'stypy_return_type' (line 143)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 16), 'stypy_return_type', tuple_2476)
    # SSA branch for the except part of a try statement (line 141)
    # SSA branch for the except 'OSError' branch of a try statement (line 141)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 141)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 140)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 139)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'link' (line 149)
    link_2479 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 9), 'link')
    str_2480 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 149, 17), 'str', 'sym')
    # Applying the binary operator '==' (line 149)
    result_eq_2481 = python_operator(stypy.reporting.localization.Localization(__file__, 149, 9), '==', link_2479, str_2480)
    
    # Testing the type of an if condition (line 149)
    if_condition_2482 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 149, 9), result_eq_2481)
    # Assigning a type to the variable 'if_condition_2482' (line 149)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 9), 'if_condition_2482', if_condition_2482)
    # SSA begins for if statement (line 149)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Evaluating a boolean operation
    
    # Call to exists(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'dst' (line 150)
    dst_2486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 31), 'dst', False)
    # Processing the call keyword arguments (line 150)
    kwargs_2487 = {}
    # Getting the type of 'os' (line 150)
    os_2483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 16), 'os', False)
    # Obtaining the member 'path' of a type (line 150)
    path_2484 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), os_2483, 'path')
    # Obtaining the member 'exists' of a type (line 150)
    exists_2485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 16), path_2484, 'exists')
    # Calling exists(args, kwargs) (line 150)
    exists_call_result_2488 = invoke(stypy.reporting.localization.Localization(__file__, 150, 16), exists_2485, *[dst_2486], **kwargs_2487)
    
    
    # Call to samefile(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'src' (line 150)
    src_2492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 57), 'src', False)
    # Getting the type of 'dst' (line 150)
    dst_2493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 62), 'dst', False)
    # Processing the call keyword arguments (line 150)
    kwargs_2494 = {}
    # Getting the type of 'os' (line 150)
    os_2489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 40), 'os', False)
    # Obtaining the member 'path' of a type (line 150)
    path_2490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 40), os_2489, 'path')
    # Obtaining the member 'samefile' of a type (line 150)
    samefile_2491 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 150, 40), path_2490, 'samefile')
    # Calling samefile(args, kwargs) (line 150)
    samefile_call_result_2495 = invoke(stypy.reporting.localization.Localization(__file__, 150, 40), samefile_2491, *[src_2492, dst_2493], **kwargs_2494)
    
    # Applying the binary operator 'and' (line 150)
    result_and_keyword_2496 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 16), 'and', exists_call_result_2488, samefile_call_result_2495)
    
    # Applying the 'not' unary operator (line 150)
    result_not__2497 = python_operator(stypy.reporting.localization.Localization(__file__, 150, 11), 'not', result_and_keyword_2496)
    
    # Testing the type of an if condition (line 150)
    if_condition_2498 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 150, 8), result_not__2497)
    # Assigning a type to the variable 'if_condition_2498' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 8), 'if_condition_2498', if_condition_2498)
    # SSA begins for if statement (line 150)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to symlink(...): (line 151)
    # Processing the call arguments (line 151)
    # Getting the type of 'src' (line 151)
    src_2501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 23), 'src', False)
    # Getting the type of 'dst' (line 151)
    dst_2502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 28), 'dst', False)
    # Processing the call keyword arguments (line 151)
    kwargs_2503 = {}
    # Getting the type of 'os' (line 151)
    os_2499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 12), 'os', False)
    # Obtaining the member 'symlink' of a type (line 151)
    symlink_2500 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 151, 12), os_2499, 'symlink')
    # Calling symlink(args, kwargs) (line 151)
    symlink_call_result_2504 = invoke(stypy.reporting.localization.Localization(__file__, 151, 12), symlink_2500, *[src_2501, dst_2502], **kwargs_2503)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 152)
    tuple_2505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 20), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 152)
    # Adding element type (line 152)
    # Getting the type of 'dst' (line 152)
    dst_2506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 20), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 20), tuple_2505, dst_2506)
    # Adding element type (line 152)
    int_2507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 25), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 152, 20), tuple_2505, int_2507)
    
    # Assigning a type to the variable 'stypy_return_type' (line 152)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 12), 'stypy_return_type', tuple_2505)
    # SSA join for if statement (line 150)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 149)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 139)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to _copy_file_contents(...): (line 156)
    # Processing the call arguments (line 156)
    # Getting the type of 'src' (line 156)
    src_2509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 24), 'src', False)
    # Getting the type of 'dst' (line 156)
    dst_2510 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 29), 'dst', False)
    # Processing the call keyword arguments (line 156)
    kwargs_2511 = {}
    # Getting the type of '_copy_file_contents' (line 156)
    _copy_file_contents_2508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), '_copy_file_contents', False)
    # Calling _copy_file_contents(args, kwargs) (line 156)
    _copy_file_contents_call_result_2512 = invoke(stypy.reporting.localization.Localization(__file__, 156, 4), _copy_file_contents_2508, *[src_2509, dst_2510], **kwargs_2511)
    
    
    
    # Evaluating a boolean operation
    # Getting the type of 'preserve_mode' (line 157)
    preserve_mode_2513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 7), 'preserve_mode')
    # Getting the type of 'preserve_times' (line 157)
    preserve_times_2514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 24), 'preserve_times')
    # Applying the binary operator 'or' (line 157)
    result_or_keyword_2515 = python_operator(stypy.reporting.localization.Localization(__file__, 157, 7), 'or', preserve_mode_2513, preserve_times_2514)
    
    # Testing the type of an if condition (line 157)
    if_condition_2516 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 157, 4), result_or_keyword_2515)
    # Assigning a type to the variable 'if_condition_2516' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 4), 'if_condition_2516', if_condition_2516)
    # SSA begins for if statement (line 157)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 158):
    
    # Call to stat(...): (line 158)
    # Processing the call arguments (line 158)
    # Getting the type of 'src' (line 158)
    src_2519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 21), 'src', False)
    # Processing the call keyword arguments (line 158)
    kwargs_2520 = {}
    # Getting the type of 'os' (line 158)
    os_2517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 13), 'os', False)
    # Obtaining the member 'stat' of a type (line 158)
    stat_2518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 13), os_2517, 'stat')
    # Calling stat(args, kwargs) (line 158)
    stat_call_result_2521 = invoke(stypy.reporting.localization.Localization(__file__, 158, 13), stat_2518, *[src_2519], **kwargs_2520)
    
    # Assigning a type to the variable 'st' (line 158)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'st', stat_call_result_2521)
    
    # Getting the type of 'preserve_times' (line 162)
    preserve_times_2522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 11), 'preserve_times')
    # Testing the type of an if condition (line 162)
    if_condition_2523 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 162, 8), preserve_times_2522)
    # Assigning a type to the variable 'if_condition_2523' (line 162)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'if_condition_2523', if_condition_2523)
    # SSA begins for if statement (line 162)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to utime(...): (line 163)
    # Processing the call arguments (line 163)
    # Getting the type of 'dst' (line 163)
    dst_2526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 21), 'dst', False)
    
    # Obtaining an instance of the builtin type 'tuple' (line 163)
    tuple_2527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 27), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 163)
    # Adding element type (line 163)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ST_ATIME' (line 163)
    ST_ATIME_2528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'ST_ATIME', False)
    # Getting the type of 'st' (line 163)
    st_2529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 27), 'st', False)
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___2530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 27), st_2529, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_2531 = invoke(stypy.reporting.localization.Localization(__file__, 163, 27), getitem___2530, ST_ATIME_2528)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 27), tuple_2527, subscript_call_result_2531)
    # Adding element type (line 163)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ST_MTIME' (line 163)
    ST_MTIME_2532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 44), 'ST_MTIME', False)
    # Getting the type of 'st' (line 163)
    st_2533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 41), 'st', False)
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___2534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 41), st_2533, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_2535 = invoke(stypy.reporting.localization.Localization(__file__, 163, 41), getitem___2534, ST_MTIME_2532)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 163, 27), tuple_2527, subscript_call_result_2535)
    
    # Processing the call keyword arguments (line 163)
    kwargs_2536 = {}
    # Getting the type of 'os' (line 163)
    os_2524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'os', False)
    # Obtaining the member 'utime' of a type (line 163)
    utime_2525 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), os_2524, 'utime')
    # Calling utime(args, kwargs) (line 163)
    utime_call_result_2537 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), utime_2525, *[dst_2526, tuple_2527], **kwargs_2536)
    
    # SSA join for if statement (line 162)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'preserve_mode' (line 164)
    preserve_mode_2538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 11), 'preserve_mode')
    # Testing the type of an if condition (line 164)
    if_condition_2539 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 164, 8), preserve_mode_2538)
    # Assigning a type to the variable 'if_condition_2539' (line 164)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'if_condition_2539', if_condition_2539)
    # SSA begins for if statement (line 164)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to chmod(...): (line 165)
    # Processing the call arguments (line 165)
    # Getting the type of 'dst' (line 165)
    dst_2542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 21), 'dst', False)
    
    # Call to S_IMODE(...): (line 165)
    # Processing the call arguments (line 165)
    
    # Obtaining the type of the subscript
    # Getting the type of 'ST_MODE' (line 165)
    ST_MODE_2544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 37), 'ST_MODE', False)
    # Getting the type of 'st' (line 165)
    st_2545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 34), 'st', False)
    # Obtaining the member '__getitem__' of a type (line 165)
    getitem___2546 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 34), st_2545, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 165)
    subscript_call_result_2547 = invoke(stypy.reporting.localization.Localization(__file__, 165, 34), getitem___2546, ST_MODE_2544)
    
    # Processing the call keyword arguments (line 165)
    kwargs_2548 = {}
    # Getting the type of 'S_IMODE' (line 165)
    S_IMODE_2543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 26), 'S_IMODE', False)
    # Calling S_IMODE(args, kwargs) (line 165)
    S_IMODE_call_result_2549 = invoke(stypy.reporting.localization.Localization(__file__, 165, 26), S_IMODE_2543, *[subscript_call_result_2547], **kwargs_2548)
    
    # Processing the call keyword arguments (line 165)
    kwargs_2550 = {}
    # Getting the type of 'os' (line 165)
    os_2540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 165, 12), 'os', False)
    # Obtaining the member 'chmod' of a type (line 165)
    chmod_2541 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 165, 12), os_2540, 'chmod')
    # Calling chmod(args, kwargs) (line 165)
    chmod_call_result_2551 = invoke(stypy.reporting.localization.Localization(__file__, 165, 12), chmod_2541, *[dst_2542, S_IMODE_call_result_2549], **kwargs_2550)
    
    # SSA join for if statement (line 164)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 157)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 167)
    tuple_2552 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 12), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 167)
    # Adding element type (line 167)
    # Getting the type of 'dst' (line 167)
    dst_2553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 12), 'dst')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 12), tuple_2552, dst_2553)
    # Adding element type (line 167)
    int_2554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 167, 17), 'int')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 167, 12), tuple_2552, int_2554)
    
    # Assigning a type to the variable 'stypy_return_type' (line 167)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type', tuple_2552)
    
    # ################# End of 'copy_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'copy_file' in the type store
    # Getting the type of 'stypy_return_type' (line 71)
    stypy_return_type_2555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2555)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'copy_file'
    return stypy_return_type_2555

# Assigning a type to the variable 'copy_file' (line 71)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'copy_file', copy_file)

@norecursion
def move_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    int_2556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 33), 'int')
    int_2557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 170, 44), 'int')
    defaults = [int_2556, int_2557]
    # Create a new context for function 'move_file'
    module_type_store = module_type_store.open_function_context('move_file', 170, 0, False)
    
    # Passed parameters checking function
    move_file.stypy_localization = localization
    move_file.stypy_type_of_self = None
    move_file.stypy_type_store = module_type_store
    move_file.stypy_function_name = 'move_file'
    move_file.stypy_param_names_list = ['src', 'dst', 'verbose', 'dry_run']
    move_file.stypy_varargs_param_name = None
    move_file.stypy_kwargs_param_name = None
    move_file.stypy_call_defaults = defaults
    move_file.stypy_call_varargs = varargs
    move_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'move_file', ['src', 'dst', 'verbose', 'dry_run'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'move_file', localization, ['src', 'dst', 'verbose', 'dry_run'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'move_file(...)' code ##################

    str_2558 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, (-1)), 'str', "Move a file 'src' to 'dst'.\n\n    If 'dst' is a directory, the file will be moved into it with the same\n    name; otherwise, 'src' is just renamed to 'dst'.  Return the new\n    full name of the file.\n\n    Handles cross-device moves on Unix using 'copy_file()'.  What about\n    other systems???\n    ")
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 180, 4))
    
    # 'from os.path import exists, isfile, isdir, basename, dirname' statement (line 180)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_2559 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 180, 4), 'os.path')

    if (type(import_2559) is not StypyTypeError):

        if (import_2559 != 'pyd_module'):
            __import__(import_2559)
            sys_modules_2560 = sys.modules[import_2559]
            import_from_module(stypy.reporting.localization.Localization(__file__, 180, 4), 'os.path', sys_modules_2560.module_type_store, module_type_store, ['exists', 'isfile', 'isdir', 'basename', 'dirname'])
            nest_module(stypy.reporting.localization.Localization(__file__, 180, 4), __file__, sys_modules_2560, sys_modules_2560.module_type_store, module_type_store)
        else:
            from os.path import exists, isfile, isdir, basename, dirname

            import_from_module(stypy.reporting.localization.Localization(__file__, 180, 4), 'os.path', None, module_type_store, ['exists', 'isfile', 'isdir', 'basename', 'dirname'], [exists, isfile, isdir, basename, dirname])

    else:
        # Assigning a type to the variable 'os.path' (line 180)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 4), 'os.path', import_2559)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 181, 4))
    
    # 'import errno' statement (line 181)
    import errno

    import_module(stypy.reporting.localization.Localization(__file__, 181, 4), 'errno', errno, module_type_store)
    
    
    
    # Getting the type of 'verbose' (line 183)
    verbose_2561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 7), 'verbose')
    int_2562 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 18), 'int')
    # Applying the binary operator '>=' (line 183)
    result_ge_2563 = python_operator(stypy.reporting.localization.Localization(__file__, 183, 7), '>=', verbose_2561, int_2562)
    
    # Testing the type of an if condition (line 183)
    if_condition_2564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 183, 4), result_ge_2563)
    # Assigning a type to the variable 'if_condition_2564' (line 183)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 4), 'if_condition_2564', if_condition_2564)
    # SSA begins for if statement (line 183)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to info(...): (line 184)
    # Processing the call arguments (line 184)
    str_2567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 17), 'str', 'moving %s -> %s')
    # Getting the type of 'src' (line 184)
    src_2568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 36), 'src', False)
    # Getting the type of 'dst' (line 184)
    dst_2569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 41), 'dst', False)
    # Processing the call keyword arguments (line 184)
    kwargs_2570 = {}
    # Getting the type of 'log' (line 184)
    log_2565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 8), 'log', False)
    # Obtaining the member 'info' of a type (line 184)
    info_2566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 8), log_2565, 'info')
    # Calling info(args, kwargs) (line 184)
    info_call_result_2571 = invoke(stypy.reporting.localization.Localization(__file__, 184, 8), info_2566, *[str_2567, src_2568, dst_2569], **kwargs_2570)
    
    # SSA join for if statement (line 183)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'dry_run' (line 186)
    dry_run_2572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 186, 7), 'dry_run')
    # Testing the type of an if condition (line 186)
    if_condition_2573 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 186, 4), dry_run_2572)
    # Assigning a type to the variable 'if_condition_2573' (line 186)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 186, 4), 'if_condition_2573', if_condition_2573)
    # SSA begins for if statement (line 186)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Getting the type of 'dst' (line 187)
    dst_2574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 187, 15), 'dst')
    # Assigning a type to the variable 'stypy_return_type' (line 187)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 187, 8), 'stypy_return_type', dst_2574)
    # SSA join for if statement (line 186)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isfile(...): (line 189)
    # Processing the call arguments (line 189)
    # Getting the type of 'src' (line 189)
    src_2576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 18), 'src', False)
    # Processing the call keyword arguments (line 189)
    kwargs_2577 = {}
    # Getting the type of 'isfile' (line 189)
    isfile_2575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 11), 'isfile', False)
    # Calling isfile(args, kwargs) (line 189)
    isfile_call_result_2578 = invoke(stypy.reporting.localization.Localization(__file__, 189, 11), isfile_2575, *[src_2576], **kwargs_2577)
    
    # Applying the 'not' unary operator (line 189)
    result_not__2579 = python_operator(stypy.reporting.localization.Localization(__file__, 189, 7), 'not', isfile_call_result_2578)
    
    # Testing the type of an if condition (line 189)
    if_condition_2580 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 189, 4), result_not__2579)
    # Assigning a type to the variable 'if_condition_2580' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 4), 'if_condition_2580', if_condition_2580)
    # SSA begins for if statement (line 189)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to DistutilsFileError(...): (line 190)
    # Processing the call arguments (line 190)
    str_2582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 190, 33), 'str', "can't move '%s': not a regular file")
    # Getting the type of 'src' (line 190)
    src_2583 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 73), 'src', False)
    # Applying the binary operator '%' (line 190)
    result_mod_2584 = python_operator(stypy.reporting.localization.Localization(__file__, 190, 33), '%', str_2582, src_2583)
    
    # Processing the call keyword arguments (line 190)
    kwargs_2585 = {}
    # Getting the type of 'DistutilsFileError' (line 190)
    DistutilsFileError_2581 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 190, 14), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 190)
    DistutilsFileError_call_result_2586 = invoke(stypy.reporting.localization.Localization(__file__, 190, 14), DistutilsFileError_2581, *[result_mod_2584], **kwargs_2585)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 190, 8), DistutilsFileError_call_result_2586, 'raise parameter', BaseException)
    # SSA join for if statement (line 189)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    # Call to isdir(...): (line 192)
    # Processing the call arguments (line 192)
    # Getting the type of 'dst' (line 192)
    dst_2588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 13), 'dst', False)
    # Processing the call keyword arguments (line 192)
    kwargs_2589 = {}
    # Getting the type of 'isdir' (line 192)
    isdir_2587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 192, 7), 'isdir', False)
    # Calling isdir(args, kwargs) (line 192)
    isdir_call_result_2590 = invoke(stypy.reporting.localization.Localization(__file__, 192, 7), isdir_2587, *[dst_2588], **kwargs_2589)
    
    # Testing the type of an if condition (line 192)
    if_condition_2591 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 192, 4), isdir_call_result_2590)
    # Assigning a type to the variable 'if_condition_2591' (line 192)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 192, 4), 'if_condition_2591', if_condition_2591)
    # SSA begins for if statement (line 192)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 193):
    
    # Call to join(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'dst' (line 193)
    dst_2595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 27), 'dst', False)
    
    # Call to basename(...): (line 193)
    # Processing the call arguments (line 193)
    # Getting the type of 'src' (line 193)
    src_2597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 41), 'src', False)
    # Processing the call keyword arguments (line 193)
    kwargs_2598 = {}
    # Getting the type of 'basename' (line 193)
    basename_2596 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 32), 'basename', False)
    # Calling basename(args, kwargs) (line 193)
    basename_call_result_2599 = invoke(stypy.reporting.localization.Localization(__file__, 193, 32), basename_2596, *[src_2597], **kwargs_2598)
    
    # Processing the call keyword arguments (line 193)
    kwargs_2600 = {}
    # Getting the type of 'os' (line 193)
    os_2592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 193, 14), 'os', False)
    # Obtaining the member 'path' of a type (line 193)
    path_2593 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 14), os_2592, 'path')
    # Obtaining the member 'join' of a type (line 193)
    join_2594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 193, 14), path_2593, 'join')
    # Calling join(args, kwargs) (line 193)
    join_call_result_2601 = invoke(stypy.reporting.localization.Localization(__file__, 193, 14), join_2594, *[dst_2595, basename_call_result_2599], **kwargs_2600)
    
    # Assigning a type to the variable 'dst' (line 193)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 193, 8), 'dst', join_call_result_2601)
    # SSA branch for the else part of an if statement (line 192)
    module_type_store.open_ssa_branch('else')
    
    
    # Call to exists(...): (line 194)
    # Processing the call arguments (line 194)
    # Getting the type of 'dst' (line 194)
    dst_2603 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 16), 'dst', False)
    # Processing the call keyword arguments (line 194)
    kwargs_2604 = {}
    # Getting the type of 'exists' (line 194)
    exists_2602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 194, 9), 'exists', False)
    # Calling exists(args, kwargs) (line 194)
    exists_call_result_2605 = invoke(stypy.reporting.localization.Localization(__file__, 194, 9), exists_2602, *[dst_2603], **kwargs_2604)
    
    # Testing the type of an if condition (line 194)
    if_condition_2606 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 194, 9), exists_call_result_2605)
    # Assigning a type to the variable 'if_condition_2606' (line 194)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 194, 9), 'if_condition_2606', if_condition_2606)
    # SSA begins for if statement (line 194)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to DistutilsFileError(...): (line 195)
    # Processing the call arguments (line 195)
    str_2608 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 14), 'str', "can't move '%s': destination '%s' already exists")
    
    # Obtaining an instance of the builtin type 'tuple' (line 197)
    tuple_2609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 197)
    # Adding element type (line 197)
    # Getting the type of 'src' (line 197)
    src_2610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 15), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 15), tuple_2609, src_2610)
    # Adding element type (line 197)
    # Getting the type of 'dst' (line 197)
    dst_2611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'dst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 197, 15), tuple_2609, dst_2611)
    
    # Applying the binary operator '%' (line 196)
    result_mod_2612 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 14), '%', str_2608, tuple_2609)
    
    # Processing the call keyword arguments (line 195)
    kwargs_2613 = {}
    # Getting the type of 'DistutilsFileError' (line 195)
    DistutilsFileError_2607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 14), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 195)
    DistutilsFileError_call_result_2614 = invoke(stypy.reporting.localization.Localization(__file__, 195, 14), DistutilsFileError_2607, *[result_mod_2612], **kwargs_2613)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 195, 8), DistutilsFileError_call_result_2614, 'raise parameter', BaseException)
    # SSA join for if statement (line 194)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 192)
    module_type_store = module_type_store.join_ssa_context()
    
    
    
    
    # Call to isdir(...): (line 199)
    # Processing the call arguments (line 199)
    
    # Call to dirname(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'dst' (line 199)
    dst_2617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 25), 'dst', False)
    # Processing the call keyword arguments (line 199)
    kwargs_2618 = {}
    # Getting the type of 'dirname' (line 199)
    dirname_2616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 17), 'dirname', False)
    # Calling dirname(args, kwargs) (line 199)
    dirname_call_result_2619 = invoke(stypy.reporting.localization.Localization(__file__, 199, 17), dirname_2616, *[dst_2617], **kwargs_2618)
    
    # Processing the call keyword arguments (line 199)
    kwargs_2620 = {}
    # Getting the type of 'isdir' (line 199)
    isdir_2615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 11), 'isdir', False)
    # Calling isdir(args, kwargs) (line 199)
    isdir_call_result_2621 = invoke(stypy.reporting.localization.Localization(__file__, 199, 11), isdir_2615, *[dirname_call_result_2619], **kwargs_2620)
    
    # Applying the 'not' unary operator (line 199)
    result_not__2622 = python_operator(stypy.reporting.localization.Localization(__file__, 199, 7), 'not', isdir_call_result_2621)
    
    # Testing the type of an if condition (line 199)
    if_condition_2623 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 199, 4), result_not__2622)
    # Assigning a type to the variable 'if_condition_2623' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 4), 'if_condition_2623', if_condition_2623)
    # SSA begins for if statement (line 199)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to DistutilsFileError(...): (line 200)
    # Processing the call arguments (line 200)
    str_2625 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 14), 'str', "can't move '%s': destination '%s' not a valid path")
    
    # Obtaining an instance of the builtin type 'tuple' (line 202)
    tuple_2626 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 202, 15), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 202)
    # Adding element type (line 202)
    # Getting the type of 'src' (line 202)
    src_2627 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 15), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 15), tuple_2626, src_2627)
    # Adding element type (line 202)
    # Getting the type of 'dst' (line 202)
    dst_2628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 20), 'dst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 202, 15), tuple_2626, dst_2628)
    
    # Applying the binary operator '%' (line 201)
    result_mod_2629 = python_operator(stypy.reporting.localization.Localization(__file__, 201, 14), '%', str_2625, tuple_2626)
    
    # Processing the call keyword arguments (line 200)
    kwargs_2630 = {}
    # Getting the type of 'DistutilsFileError' (line 200)
    DistutilsFileError_2624 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 14), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 200)
    DistutilsFileError_call_result_2631 = invoke(stypy.reporting.localization.Localization(__file__, 200, 14), DistutilsFileError_2624, *[result_mod_2629], **kwargs_2630)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 200, 8), DistutilsFileError_call_result_2631, 'raise parameter', BaseException)
    # SSA join for if statement (line 199)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Num to a Name (line 204):
    int_2632 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 14), 'int')
    # Assigning a type to the variable 'copy_it' (line 204)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 204, 4), 'copy_it', int_2632)
    
    
    # SSA begins for try-except statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to rename(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'src' (line 206)
    src_2635 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 18), 'src', False)
    # Getting the type of 'dst' (line 206)
    dst_2636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 23), 'dst', False)
    # Processing the call keyword arguments (line 206)
    kwargs_2637 = {}
    # Getting the type of 'os' (line 206)
    os_2633 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 8), 'os', False)
    # Obtaining the member 'rename' of a type (line 206)
    rename_2634 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 8), os_2633, 'rename')
    # Calling rename(args, kwargs) (line 206)
    rename_call_result_2638 = invoke(stypy.reporting.localization.Localization(__file__, 206, 8), rename_2634, *[src_2635, dst_2636], **kwargs_2637)
    
    # SSA branch for the except part of a try statement (line 205)
    # SSA branch for the except 'Attribute' branch of a try statement (line 205)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'os' (line 207)
    os_2639 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 11), 'os')
    # Obtaining the member 'error' of a type (line 207)
    error_2640 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 11), os_2639, 'error')
    # Assigning a type to the variable 'num' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'num', error_2640)
    # Assigning a type to the variable 'msg' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 4), 'msg', error_2640)
    
    
    # Getting the type of 'num' (line 208)
    num_2641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 11), 'num')
    # Getting the type of 'errno' (line 208)
    errno_2642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 18), 'errno')
    # Obtaining the member 'EXDEV' of a type (line 208)
    EXDEV_2643 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 18), errno_2642, 'EXDEV')
    # Applying the binary operator '==' (line 208)
    result_eq_2644 = python_operator(stypy.reporting.localization.Localization(__file__, 208, 11), '==', num_2641, EXDEV_2643)
    
    # Testing the type of an if condition (line 208)
    if_condition_2645 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 208, 8), result_eq_2644)
    # Assigning a type to the variable 'if_condition_2645' (line 208)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 208, 8), 'if_condition_2645', if_condition_2645)
    # SSA begins for if statement (line 208)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Num to a Name (line 209):
    int_2646 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 22), 'int')
    # Assigning a type to the variable 'copy_it' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 12), 'copy_it', int_2646)
    # SSA branch for the else part of an if statement (line 208)
    module_type_store.open_ssa_branch('else')
    
    # Call to DistutilsFileError(...): (line 211)
    # Processing the call arguments (line 211)
    str_2648 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 18), 'str', "couldn't move '%s' to '%s': %s")
    
    # Obtaining an instance of the builtin type 'tuple' (line 212)
    tuple_2649 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 212, 54), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 212)
    # Adding element type (line 212)
    # Getting the type of 'src' (line 212)
    src_2650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 54), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 54), tuple_2649, src_2650)
    # Adding element type (line 212)
    # Getting the type of 'dst' (line 212)
    dst_2651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 59), 'dst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 54), tuple_2649, dst_2651)
    # Adding element type (line 212)
    # Getting the type of 'msg' (line 212)
    msg_2652 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 64), 'msg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 212, 54), tuple_2649, msg_2652)
    
    # Applying the binary operator '%' (line 212)
    result_mod_2653 = python_operator(stypy.reporting.localization.Localization(__file__, 212, 18), '%', str_2648, tuple_2649)
    
    # Processing the call keyword arguments (line 211)
    kwargs_2654 = {}
    # Getting the type of 'DistutilsFileError' (line 211)
    DistutilsFileError_2647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 18), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 211)
    DistutilsFileError_call_result_2655 = invoke(stypy.reporting.localization.Localization(__file__, 211, 18), DistutilsFileError_2647, *[result_mod_2653], **kwargs_2654)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 211, 12), DistutilsFileError_call_result_2655, 'raise parameter', BaseException)
    # SSA join for if statement (line 208)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for try-except statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Getting the type of 'copy_it' (line 214)
    copy_it_2656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 7), 'copy_it')
    # Testing the type of an if condition (line 214)
    if_condition_2657 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 214, 4), copy_it_2656)
    # Assigning a type to the variable 'if_condition_2657' (line 214)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 214, 4), 'if_condition_2657', if_condition_2657)
    # SSA begins for if statement (line 214)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to copy_file(...): (line 215)
    # Processing the call arguments (line 215)
    # Getting the type of 'src' (line 215)
    src_2659 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 18), 'src', False)
    # Getting the type of 'dst' (line 215)
    dst_2660 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 23), 'dst', False)
    # Processing the call keyword arguments (line 215)
    # Getting the type of 'verbose' (line 215)
    verbose_2661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 36), 'verbose', False)
    keyword_2662 = verbose_2661
    kwargs_2663 = {'verbose': keyword_2662}
    # Getting the type of 'copy_file' (line 215)
    copy_file_2658 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 8), 'copy_file', False)
    # Calling copy_file(args, kwargs) (line 215)
    copy_file_call_result_2664 = invoke(stypy.reporting.localization.Localization(__file__, 215, 8), copy_file_2658, *[src_2659, dst_2660], **kwargs_2663)
    
    
    
    # SSA begins for try-except statement (line 216)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to unlink(...): (line 217)
    # Processing the call arguments (line 217)
    # Getting the type of 'src' (line 217)
    src_2667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 22), 'src', False)
    # Processing the call keyword arguments (line 217)
    kwargs_2668 = {}
    # Getting the type of 'os' (line 217)
    os_2665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 12), 'os', False)
    # Obtaining the member 'unlink' of a type (line 217)
    unlink_2666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 217, 12), os_2665, 'unlink')
    # Calling unlink(args, kwargs) (line 217)
    unlink_call_result_2669 = invoke(stypy.reporting.localization.Localization(__file__, 217, 12), unlink_2666, *[src_2667], **kwargs_2668)
    
    # SSA branch for the except part of a try statement (line 216)
    # SSA branch for the except 'Attribute' branch of a try statement (line 216)
    # Storing handler type
    module_type_store.open_ssa_branch('except')
    # Getting the type of 'os' (line 218)
    os_2670 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 15), 'os')
    # Obtaining the member 'error' of a type (line 218)
    error_2671 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 15), os_2670, 'error')
    # Assigning a type to the variable 'num' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'num', error_2671)
    # Assigning a type to the variable 'msg' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 8), 'msg', error_2671)
    
    
    # SSA begins for try-except statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
    
    # Call to unlink(...): (line 220)
    # Processing the call arguments (line 220)
    # Getting the type of 'dst' (line 220)
    dst_2674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 26), 'dst', False)
    # Processing the call keyword arguments (line 220)
    kwargs_2675 = {}
    # Getting the type of 'os' (line 220)
    os_2672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 16), 'os', False)
    # Obtaining the member 'unlink' of a type (line 220)
    unlink_2673 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 16), os_2672, 'unlink')
    # Calling unlink(args, kwargs) (line 220)
    unlink_call_result_2676 = invoke(stypy.reporting.localization.Localization(__file__, 220, 16), unlink_2673, *[dst_2674], **kwargs_2675)
    
    # SSA branch for the except part of a try statement (line 219)
    # SSA branch for the except 'Attribute' branch of a try statement (line 219)
    module_type_store.open_ssa_branch('except')
    pass
    # SSA join for try-except statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to DistutilsFileError(...): (line 223)
    # Processing the call arguments (line 223)
    str_2678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 224, 19), 'str', "couldn't move '%s' to '%s' by copy/delete: ")
    str_2679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 19), 'str', "delete '%s' failed: %s")
    # Applying the binary operator '+' (line 224)
    result_add_2680 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 19), '+', str_2678, str_2679)
    
    
    # Obtaining an instance of the builtin type 'tuple' (line 226)
    tuple_2681 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 226, 19), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 226)
    # Adding element type (line 226)
    # Getting the type of 'src' (line 226)
    src_2682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 19), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 19), tuple_2681, src_2682)
    # Adding element type (line 226)
    # Getting the type of 'dst' (line 226)
    dst_2683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 24), 'dst', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 19), tuple_2681, dst_2683)
    # Adding element type (line 226)
    # Getting the type of 'src' (line 226)
    src_2684 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 29), 'src', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 19), tuple_2681, src_2684)
    # Adding element type (line 226)
    # Getting the type of 'msg' (line 226)
    msg_2685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 34), 'msg', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 226, 19), tuple_2681, msg_2685)
    
    # Applying the binary operator '%' (line 224)
    result_mod_2686 = python_operator(stypy.reporting.localization.Localization(__file__, 224, 18), '%', result_add_2680, tuple_2681)
    
    # Processing the call keyword arguments (line 223)
    kwargs_2687 = {}
    # Getting the type of 'DistutilsFileError' (line 223)
    DistutilsFileError_2677 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 18), 'DistutilsFileError', False)
    # Calling DistutilsFileError(args, kwargs) (line 223)
    DistutilsFileError_call_result_2688 = invoke(stypy.reporting.localization.Localization(__file__, 223, 18), DistutilsFileError_2677, *[result_mod_2686], **kwargs_2687)
    
    ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 223, 12), DistutilsFileError_call_result_2688, 'raise parameter', BaseException)
    # SSA join for try-except statement (line 216)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 214)
    module_type_store = module_type_store.join_ssa_context()
    
    # Getting the type of 'dst' (line 227)
    dst_2689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 11), 'dst')
    # Assigning a type to the variable 'stypy_return_type' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 4), 'stypy_return_type', dst_2689)
    
    # ################# End of 'move_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'move_file' in the type store
    # Getting the type of 'stypy_return_type' (line 170)
    stypy_return_type_2690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2690)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'move_file'
    return stypy_return_type_2690

# Assigning a type to the variable 'move_file' (line 170)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 0), 'move_file', move_file)

@norecursion
def write_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'write_file'
    module_type_store = module_type_store.open_function_context('write_file', 230, 0, False)
    
    # Passed parameters checking function
    write_file.stypy_localization = localization
    write_file.stypy_type_of_self = None
    write_file.stypy_type_store = module_type_store
    write_file.stypy_function_name = 'write_file'
    write_file.stypy_param_names_list = ['filename', 'contents']
    write_file.stypy_varargs_param_name = None
    write_file.stypy_kwargs_param_name = None
    write_file.stypy_call_defaults = defaults
    write_file.stypy_call_varargs = varargs
    write_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'write_file', ['filename', 'contents'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'write_file', localization, ['filename', 'contents'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'write_file(...)' code ##################

    str_2691 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 233, (-1)), 'str', "Create a file with the specified name and write 'contents' (a\n    sequence of strings without line terminators) to it.\n    ")
    
    # Assigning a Call to a Name (line 234):
    
    # Call to open(...): (line 234)
    # Processing the call arguments (line 234)
    # Getting the type of 'filename' (line 234)
    filename_2693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 13), 'filename', False)
    str_2694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 234, 23), 'str', 'w')
    # Processing the call keyword arguments (line 234)
    kwargs_2695 = {}
    # Getting the type of 'open' (line 234)
    open_2692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 234, 8), 'open', False)
    # Calling open(args, kwargs) (line 234)
    open_call_result_2696 = invoke(stypy.reporting.localization.Localization(__file__, 234, 8), open_2692, *[filename_2693, str_2694], **kwargs_2695)
    
    # Assigning a type to the variable 'f' (line 234)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 234, 4), 'f', open_call_result_2696)
    
    # Try-finally block (line 235)
    
    # Getting the type of 'contents' (line 236)
    contents_2697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'contents')
    # Testing the type of a for loop iterable (line 236)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 236, 8), contents_2697)
    # Getting the type of the for loop variable (line 236)
    for_loop_var_2698 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 236, 8), contents_2697)
    # Assigning a type to the variable 'line' (line 236)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 236, 8), 'line', for_loop_var_2698)
    # SSA begins for a for statement (line 236)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to write(...): (line 237)
    # Processing the call arguments (line 237)
    # Getting the type of 'line' (line 237)
    line_2701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 20), 'line', False)
    str_2702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 237, 27), 'str', '\n')
    # Applying the binary operator '+' (line 237)
    result_add_2703 = python_operator(stypy.reporting.localization.Localization(__file__, 237, 20), '+', line_2701, str_2702)
    
    # Processing the call keyword arguments (line 237)
    kwargs_2704 = {}
    # Getting the type of 'f' (line 237)
    f_2699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 237, 12), 'f', False)
    # Obtaining the member 'write' of a type (line 237)
    write_2700 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 237, 12), f_2699, 'write')
    # Calling write(args, kwargs) (line 237)
    write_call_result_2705 = invoke(stypy.reporting.localization.Localization(__file__, 237, 12), write_2700, *[result_add_2703], **kwargs_2704)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 235)
    
    # Call to close(...): (line 239)
    # Processing the call keyword arguments (line 239)
    kwargs_2708 = {}
    # Getting the type of 'f' (line 239)
    f_2706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 239, 8), 'f', False)
    # Obtaining the member 'close' of a type (line 239)
    close_2707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 239, 8), f_2706, 'close')
    # Calling close(args, kwargs) (line 239)
    close_call_result_2709 = invoke(stypy.reporting.localization.Localization(__file__, 239, 8), close_2707, *[], **kwargs_2708)
    
    
    
    # ################# End of 'write_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'write_file' in the type store
    # Getting the type of 'stypy_return_type' (line 230)
    stypy_return_type_2710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_2710)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'write_file'
    return stypy_return_type_2710

# Assigning a type to the variable 'write_file' (line 230)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 0), 'write_file', write_file)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
