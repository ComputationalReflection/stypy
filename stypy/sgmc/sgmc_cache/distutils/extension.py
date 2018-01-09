
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.extension
2: 
3: Provides the Extension class, used to describe C/C++ extension
4: modules in setup scripts.'''
5: 
6: __revision__ = "$Id$"
7: 
8: import os, string, sys
9: from types import *
10: 
11: try:
12:     import warnings
13: except ImportError:
14:     warnings = None
15: 
16: # This class is really only used by the "build_ext" command, so it might
17: # make sense to put it in distutils.command.build_ext.  However, that
18: # module is already big enough, and I want to make this class a bit more
19: # complex to simplify some common cases ("foo" module in "foo.c") and do
20: # better error-checking ("foo.c" actually exists).
21: #
22: # Also, putting this in build_ext.py means every setup script would have to
23: # import that large-ish module (indirectly, through distutils.core) in
24: # order to do anything.
25: 
26: class Extension:
27:     '''Just a collection of attributes that describes an extension
28:     module and everything needed to build it (hopefully in a portable
29:     way, but there are hooks that let you be as unportable as you need).
30: 
31:     Instance attributes:
32:       name : string
33:         the full name of the extension, including any packages -- ie.
34:         *not* a filename or pathname, but Python dotted name
35:       sources : [string]
36:         list of source filenames, relative to the distribution root
37:         (where the setup script lives), in Unix form (slash-separated)
38:         for portability.  Source files may be C, C++, SWIG (.i),
39:         platform-specific resource files, or whatever else is recognized
40:         by the "build_ext" command as source for a Python extension.
41:       include_dirs : [string]
42:         list of directories to search for C/C++ header files (in Unix
43:         form for portability)
44:       define_macros : [(name : string, value : string|None)]
45:         list of macros to define; each macro is defined using a 2-tuple,
46:         where 'value' is either the string to define it to or None to
47:         define it without a particular value (equivalent of "#define
48:         FOO" in source or -DFOO on Unix C compiler command line)
49:       undef_macros : [string]
50:         list of macros to undefine explicitly
51:       library_dirs : [string]
52:         list of directories to search for C/C++ libraries at link time
53:       libraries : [string]
54:         list of library names (not filenames or paths) to link against
55:       runtime_library_dirs : [string]
56:         list of directories to search for C/C++ libraries at run time
57:         (for shared extensions, this is when the extension is loaded)
58:       extra_objects : [string]
59:         list of extra files to link with (eg. object files not implied
60:         by 'sources', static library that must be explicitly specified,
61:         binary resource files, etc.)
62:       extra_compile_args : [string]
63:         any extra platform- and compiler-specific information to use
64:         when compiling the source files in 'sources'.  For platforms and
65:         compilers where "command line" makes sense, this is typically a
66:         list of command-line arguments, but for other platforms it could
67:         be anything.
68:       extra_link_args : [string]
69:         any extra platform- and compiler-specific information to use
70:         when linking object files together to create the extension (or
71:         to create a new static Python interpreter).  Similar
72:         interpretation as for 'extra_compile_args'.
73:       export_symbols : [string]
74:         list of symbols to be exported from a shared extension.  Not
75:         used on all platforms, and not generally necessary for Python
76:         extensions, which typically export exactly one symbol: "init" +
77:         extension_name.
78:       swig_opts : [string]
79:         any extra options to pass to SWIG if a source file has the .i
80:         extension.
81:       depends : [string]
82:         list of files that the extension depends on
83:       language : string
84:         extension language (i.e. "c", "c++", "objc"). Will be detected
85:         from the source extensions if not provided.
86:     '''
87: 
88:     # When adding arguments to this constructor, be sure to update
89:     # setup_keywords in core.py.
90:     def __init__ (self, name, sources,
91:                   include_dirs=None,
92:                   define_macros=None,
93:                   undef_macros=None,
94:                   library_dirs=None,
95:                   libraries=None,
96:                   runtime_library_dirs=None,
97:                   extra_objects=None,
98:                   extra_compile_args=None,
99:                   extra_link_args=None,
100:                   export_symbols=None,
101:                   swig_opts = None,
102:                   depends=None,
103:                   language=None,
104:                   **kw                      # To catch unknown keywords
105:                  ):
106:         assert type(name) is StringType, "'name' must be a string"
107:         assert (type(sources) is ListType and
108:                 map(type, sources) == [StringType]*len(sources)), \
109:                 "'sources' must be a list of strings"
110: 
111:         self.name = name
112:         self.sources = sources
113:         self.include_dirs = include_dirs or []
114:         self.define_macros = define_macros or []
115:         self.undef_macros = undef_macros or []
116:         self.library_dirs = library_dirs or []
117:         self.libraries = libraries or []
118:         self.runtime_library_dirs = runtime_library_dirs or []
119:         self.extra_objects = extra_objects or []
120:         self.extra_compile_args = extra_compile_args or []
121:         self.extra_link_args = extra_link_args or []
122:         self.export_symbols = export_symbols or []
123:         self.swig_opts = swig_opts or []
124:         self.depends = depends or []
125:         self.language = language
126: 
127:         # If there are unknown keyword options, warn about them
128:         if len(kw):
129:             L = kw.keys() ; L.sort()
130:             L = map(repr, L)
131:             msg = "Unknown Extension options: " + string.join(L, ', ')
132:             if warnings is not None:
133:                 warnings.warn(msg)
134:             else:
135:                 sys.stderr.write(msg + '\n')
136: # class Extension
137: 
138: 
139: def read_setup_file (filename):
140:     from distutils.sysconfig import \
141:          parse_makefile, expand_makefile_vars, _variable_rx
142:     from distutils.text_file import TextFile
143:     from distutils.util import split_quoted
144: 
145:     # First pass over the file to gather "VAR = VALUE" assignments.
146:     vars = parse_makefile(filename)
147: 
148:     # Second pass to gobble up the real content: lines of the form
149:     #   <module> ... [<sourcefile> ...] [<cpparg> ...] [<library> ...]
150:     file = TextFile(filename,
151:                     strip_comments=1, skip_blanks=1, join_lines=1,
152:                     lstrip_ws=1, rstrip_ws=1)
153:     try:
154:         extensions = []
155: 
156:         while 1:
157:             line = file.readline()
158:             if line is None:                # eof
159:                 break
160:             if _variable_rx.match(line):    # VAR=VALUE, handled in first pass
161:                 continue
162: 
163:                 if line[0] == line[-1] == "*":
164:                     file.warn("'%s' lines not handled yet" % line)
165:                     continue
166: 
167:             #print "original line: " + line
168:             line = expand_makefile_vars(line, vars)
169:             words = split_quoted(line)
170:             #print "expanded line: " + line
171: 
172:             # NB. this parses a slightly different syntax than the old
173:             # makesetup script: here, there must be exactly one extension per
174:             # line, and it must be the first word of the line.  I have no idea
175:             # why the old syntax supported multiple extensions per line, as
176:             # they all wind up being the same.
177: 
178:             module = words[0]
179:             ext = Extension(module, [])
180:             append_next_word = None
181: 
182:             for word in words[1:]:
183:                 if append_next_word is not None:
184:                     append_next_word.append(word)
185:                     append_next_word = None
186:                     continue
187: 
188:                 suffix = os.path.splitext(word)[1]
189:                 switch = word[0:2] ; value = word[2:]
190: 
191:                 if suffix in (".c", ".cc", ".cpp", ".cxx", ".c++", ".m", ".mm"):
192:                     # hmm, should we do something about C vs. C++ sources?
193:                     # or leave it up to the CCompiler implementation to
194:                     # worry about?
195:                     ext.sources.append(word)
196:                 elif switch == "-I":
197:                     ext.include_dirs.append(value)
198:                 elif switch == "-D":
199:                     equals = string.find(value, "=")
200:                     if equals == -1:        # bare "-DFOO" -- no value
201:                         ext.define_macros.append((value, None))
202:                     else:                   # "-DFOO=blah"
203:                         ext.define_macros.append((value[0:equals],
204:                                                   value[equals+2:]))
205:                 elif switch == "-U":
206:                     ext.undef_macros.append(value)
207:                 elif switch == "-C":        # only here 'cause makesetup has it!
208:                     ext.extra_compile_args.append(word)
209:                 elif switch == "-l":
210:                     ext.libraries.append(value)
211:                 elif switch == "-L":
212:                     ext.library_dirs.append(value)
213:                 elif switch == "-R":
214:                     ext.runtime_library_dirs.append(value)
215:                 elif word == "-rpath":
216:                     append_next_word = ext.runtime_library_dirs
217:                 elif word == "-Xlinker":
218:                     append_next_word = ext.extra_link_args
219:                 elif word == "-Xcompiler":
220:                     append_next_word = ext.extra_compile_args
221:                 elif switch == "-u":
222:                     ext.extra_link_args.append(word)
223:                     if not value:
224:                         append_next_word = ext.extra_link_args
225:                 elif word == "-Xcompiler":
226:                     append_next_word = ext.extra_compile_args
227:                 elif switch == "-u":
228:                     ext.extra_link_args.append(word)
229:                     if not value:
230:                         append_next_word = ext.extra_link_args
231:                 elif suffix in (".a", ".so", ".sl", ".o", ".dylib"):
232:                     # NB. a really faithful emulation of makesetup would
233:                     # append a .o file to extra_objects only if it
234:                     # had a slash in it; otherwise, it would s/.o/.c/
235:                     # and append it to sources.  Hmmmm.
236:                     ext.extra_objects.append(word)
237:                 else:
238:                     file.warn("unrecognized argument '%s'" % word)
239: 
240:             extensions.append(ext)
241:     finally:
242:         file.close()
243: 
244:         #print "module:", module
245:         #print "source files:", source_files
246:         #print "cpp args:", cpp_args
247:         #print "lib args:", library_args
248: 
249:         #extensions[module] = { 'sources': source_files,
250:         #                       'cpp_args': cpp_args,
251:         #                       'lib_args': library_args }
252: 
253:     return extensions
254: 
255: # read_setup_file ()
256: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_1 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', 'distutils.extension\n\nProvides the Extension class, used to describe C/C++ extension\nmodules in setup scripts.')

# Assigning a Str to a Name (line 6):
str_2 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_2)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# Multiple import statement. import os (1/3) (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)
# Multiple import statement. import string (2/3) (line 8)
import string

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'string', string, module_type_store)
# Multiple import statement. import sys (3/3) (line 8)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from types import ' statement (line 9)
try:
    from types import *

except:
    pass
import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'types', None, module_type_store, ['*'], None)



# SSA begins for try-except statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 4))

# 'import warnings' statement (line 12)
import warnings

import_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'warnings', warnings, module_type_store)

# SSA branch for the except part of a try statement (line 11)
# SSA branch for the except 'ImportError' branch of a try statement (line 11)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 14):
# Getting the type of 'None' (line 14)
None_3 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 15), 'None')
# Assigning a type to the variable 'warnings' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'warnings', None_3)
# SSA join for try-except statement (line 11)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'Extension' class

class Extension:
    str_4 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, (-1)), 'str', 'Just a collection of attributes that describes an extension\n    module and everything needed to build it (hopefully in a portable\n    way, but there are hooks that let you be as unportable as you need).\n\n    Instance attributes:\n      name : string\n        the full name of the extension, including any packages -- ie.\n        *not* a filename or pathname, but Python dotted name\n      sources : [string]\n        list of source filenames, relative to the distribution root\n        (where the setup script lives), in Unix form (slash-separated)\n        for portability.  Source files may be C, C++, SWIG (.i),\n        platform-specific resource files, or whatever else is recognized\n        by the "build_ext" command as source for a Python extension.\n      include_dirs : [string]\n        list of directories to search for C/C++ header files (in Unix\n        form for portability)\n      define_macros : [(name : string, value : string|None)]\n        list of macros to define; each macro is defined using a 2-tuple,\n        where \'value\' is either the string to define it to or None to\n        define it without a particular value (equivalent of "#define\n        FOO" in source or -DFOO on Unix C compiler command line)\n      undef_macros : [string]\n        list of macros to undefine explicitly\n      library_dirs : [string]\n        list of directories to search for C/C++ libraries at link time\n      libraries : [string]\n        list of library names (not filenames or paths) to link against\n      runtime_library_dirs : [string]\n        list of directories to search for C/C++ libraries at run time\n        (for shared extensions, this is when the extension is loaded)\n      extra_objects : [string]\n        list of extra files to link with (eg. object files not implied\n        by \'sources\', static library that must be explicitly specified,\n        binary resource files, etc.)\n      extra_compile_args : [string]\n        any extra platform- and compiler-specific information to use\n        when compiling the source files in \'sources\'.  For platforms and\n        compilers where "command line" makes sense, this is typically a\n        list of command-line arguments, but for other platforms it could\n        be anything.\n      extra_link_args : [string]\n        any extra platform- and compiler-specific information to use\n        when linking object files together to create the extension (or\n        to create a new static Python interpreter).  Similar\n        interpretation as for \'extra_compile_args\'.\n      export_symbols : [string]\n        list of symbols to be exported from a shared extension.  Not\n        used on all platforms, and not generally necessary for Python\n        extensions, which typically export exactly one symbol: "init" +\n        extension_name.\n      swig_opts : [string]\n        any extra options to pass to SWIG if a source file has the .i\n        extension.\n      depends : [string]\n        list of files that the extension depends on\n      language : string\n        extension language (i.e. "c", "c++", "objc"). Will be detected\n        from the source extensions if not provided.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 91)
        None_5 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 31), 'None')
        # Getting the type of 'None' (line 92)
        None_6 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 32), 'None')
        # Getting the type of 'None' (line 93)
        None_7 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 31), 'None')
        # Getting the type of 'None' (line 94)
        None_8 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 31), 'None')
        # Getting the type of 'None' (line 95)
        None_9 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 28), 'None')
        # Getting the type of 'None' (line 96)
        None_10 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 39), 'None')
        # Getting the type of 'None' (line 97)
        None_11 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 32), 'None')
        # Getting the type of 'None' (line 98)
        None_12 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 37), 'None')
        # Getting the type of 'None' (line 99)
        None_13 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 34), 'None')
        # Getting the type of 'None' (line 100)
        None_14 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 33), 'None')
        # Getting the type of 'None' (line 101)
        None_15 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 30), 'None')
        # Getting the type of 'None' (line 102)
        None_16 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 26), 'None')
        # Getting the type of 'None' (line 103)
        None_17 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 27), 'None')
        defaults = [None_5, None_6, None_7, None_8, None_9, None_10, None_11, None_12, None_13, None_14, None_15, None_16, None_17]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 90, 4, False)
        # Assigning a type to the variable 'self' (line 91)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'Extension.__init__', ['name', 'sources', 'include_dirs', 'define_macros', 'undef_macros', 'library_dirs', 'libraries', 'runtime_library_dirs', 'extra_objects', 'extra_compile_args', 'extra_link_args', 'export_symbols', 'swig_opts', 'depends', 'language'], None, 'kw', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['name', 'sources', 'include_dirs', 'define_macros', 'undef_macros', 'library_dirs', 'libraries', 'runtime_library_dirs', 'extra_objects', 'extra_compile_args', 'extra_link_args', 'export_symbols', 'swig_opts', 'depends', 'language'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        # Evaluating assert statement condition
        
        
        # Call to type(...): (line 106)
        # Processing the call arguments (line 106)
        # Getting the type of 'name' (line 106)
        name_19 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 20), 'name', False)
        # Processing the call keyword arguments (line 106)
        kwargs_20 = {}
        # Getting the type of 'type' (line 106)
        type_18 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 15), 'type', False)
        # Calling type(args, kwargs) (line 106)
        type_call_result_21 = invoke(stypy.reporting.localization.Localization(__file__, 106, 15), type_18, *[name_19], **kwargs_20)
        
        # Getting the type of 'StringType' (line 106)
        StringType_22 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 29), 'StringType')
        # Applying the binary operator 'is' (line 106)
        result_is__23 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 15), 'is', type_call_result_21, StringType_22)
        
        # Evaluating assert statement condition
        
        # Evaluating a boolean operation
        
        
        # Call to type(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'sources' (line 107)
        sources_25 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 21), 'sources', False)
        # Processing the call keyword arguments (line 107)
        kwargs_26 = {}
        # Getting the type of 'type' (line 107)
        type_24 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 16), 'type', False)
        # Calling type(args, kwargs) (line 107)
        type_call_result_27 = invoke(stypy.reporting.localization.Localization(__file__, 107, 16), type_24, *[sources_25], **kwargs_26)
        
        # Getting the type of 'ListType' (line 107)
        ListType_28 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 33), 'ListType')
        # Applying the binary operator 'is' (line 107)
        result_is__29 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 16), 'is', type_call_result_27, ListType_28)
        
        
        
        # Call to map(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'type' (line 108)
        type_31 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 20), 'type', False)
        # Getting the type of 'sources' (line 108)
        sources_32 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 26), 'sources', False)
        # Processing the call keyword arguments (line 108)
        kwargs_33 = {}
        # Getting the type of 'map' (line 108)
        map_30 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 16), 'map', False)
        # Calling map(args, kwargs) (line 108)
        map_call_result_34 = invoke(stypy.reporting.localization.Localization(__file__, 108, 16), map_30, *[type_31, sources_32], **kwargs_33)
        
        
        # Obtaining an instance of the builtin type 'list' (line 108)
        list_35 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 108)
        # Adding element type (line 108)
        # Getting the type of 'StringType' (line 108)
        StringType_36 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 39), 'StringType')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 108, 38), list_35, StringType_36)
        
        
        # Call to len(...): (line 108)
        # Processing the call arguments (line 108)
        # Getting the type of 'sources' (line 108)
        sources_38 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 55), 'sources', False)
        # Processing the call keyword arguments (line 108)
        kwargs_39 = {}
        # Getting the type of 'len' (line 108)
        len_37 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 51), 'len', False)
        # Calling len(args, kwargs) (line 108)
        len_call_result_40 = invoke(stypy.reporting.localization.Localization(__file__, 108, 51), len_37, *[sources_38], **kwargs_39)
        
        # Applying the binary operator '*' (line 108)
        result_mul_41 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 38), '*', list_35, len_call_result_40)
        
        # Applying the binary operator '==' (line 108)
        result_eq_42 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 16), '==', map_call_result_34, result_mul_41)
        
        # Applying the binary operator 'and' (line 107)
        result_and_keyword_43 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 16), 'and', result_is__29, result_eq_42)
        
        
        # Assigning a Name to a Attribute (line 111):
        # Getting the type of 'name' (line 111)
        name_44 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'name')
        # Getting the type of 'self' (line 111)
        self_45 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 8), 'self')
        # Setting the type of the member 'name' of a type (line 111)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 8), self_45, 'name', name_44)
        
        # Assigning a Name to a Attribute (line 112):
        # Getting the type of 'sources' (line 112)
        sources_46 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'sources')
        # Getting the type of 'self' (line 112)
        self_47 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'self')
        # Setting the type of the member 'sources' of a type (line 112)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 8), self_47, 'sources', sources_46)
        
        # Assigning a BoolOp to a Attribute (line 113):
        
        # Evaluating a boolean operation
        # Getting the type of 'include_dirs' (line 113)
        include_dirs_48 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 28), 'include_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 113)
        list_49 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 113)
        
        # Applying the binary operator 'or' (line 113)
        result_or_keyword_50 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 28), 'or', include_dirs_48, list_49)
        
        # Getting the type of 'self' (line 113)
        self_51 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 8), 'self')
        # Setting the type of the member 'include_dirs' of a type (line 113)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 8), self_51, 'include_dirs', result_or_keyword_50)
        
        # Assigning a BoolOp to a Attribute (line 114):
        
        # Evaluating a boolean operation
        # Getting the type of 'define_macros' (line 114)
        define_macros_52 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 29), 'define_macros')
        
        # Obtaining an instance of the builtin type 'list' (line 114)
        list_53 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 114)
        
        # Applying the binary operator 'or' (line 114)
        result_or_keyword_54 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 29), 'or', define_macros_52, list_53)
        
        # Getting the type of 'self' (line 114)
        self_55 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'self')
        # Setting the type of the member 'define_macros' of a type (line 114)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 8), self_55, 'define_macros', result_or_keyword_54)
        
        # Assigning a BoolOp to a Attribute (line 115):
        
        # Evaluating a boolean operation
        # Getting the type of 'undef_macros' (line 115)
        undef_macros_56 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'undef_macros')
        
        # Obtaining an instance of the builtin type 'list' (line 115)
        list_57 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 115)
        
        # Applying the binary operator 'or' (line 115)
        result_or_keyword_58 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 28), 'or', undef_macros_56, list_57)
        
        # Getting the type of 'self' (line 115)
        self_59 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'self')
        # Setting the type of the member 'undef_macros' of a type (line 115)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 115, 8), self_59, 'undef_macros', result_or_keyword_58)
        
        # Assigning a BoolOp to a Attribute (line 116):
        
        # Evaluating a boolean operation
        # Getting the type of 'library_dirs' (line 116)
        library_dirs_60 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 28), 'library_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 116)
        list_61 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 116)
        
        # Applying the binary operator 'or' (line 116)
        result_or_keyword_62 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 28), 'or', library_dirs_60, list_61)
        
        # Getting the type of 'self' (line 116)
        self_63 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self')
        # Setting the type of the member 'library_dirs' of a type (line 116)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_63, 'library_dirs', result_or_keyword_62)
        
        # Assigning a BoolOp to a Attribute (line 117):
        
        # Evaluating a boolean operation
        # Getting the type of 'libraries' (line 117)
        libraries_64 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 25), 'libraries')
        
        # Obtaining an instance of the builtin type 'list' (line 117)
        list_65 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 117)
        
        # Applying the binary operator 'or' (line 117)
        result_or_keyword_66 = python_operator(stypy.reporting.localization.Localization(__file__, 117, 25), 'or', libraries_64, list_65)
        
        # Getting the type of 'self' (line 117)
        self_67 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 117)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 8), self_67, 'libraries', result_or_keyword_66)
        
        # Assigning a BoolOp to a Attribute (line 118):
        
        # Evaluating a boolean operation
        # Getting the type of 'runtime_library_dirs' (line 118)
        runtime_library_dirs_68 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 36), 'runtime_library_dirs')
        
        # Obtaining an instance of the builtin type 'list' (line 118)
        list_69 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 60), 'list')
        # Adding type elements to the builtin type 'list' instance (line 118)
        
        # Applying the binary operator 'or' (line 118)
        result_or_keyword_70 = python_operator(stypy.reporting.localization.Localization(__file__, 118, 36), 'or', runtime_library_dirs_68, list_69)
        
        # Getting the type of 'self' (line 118)
        self_71 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 8), 'self')
        # Setting the type of the member 'runtime_library_dirs' of a type (line 118)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 8), self_71, 'runtime_library_dirs', result_or_keyword_70)
        
        # Assigning a BoolOp to a Attribute (line 119):
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_objects' (line 119)
        extra_objects_72 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 29), 'extra_objects')
        
        # Obtaining an instance of the builtin type 'list' (line 119)
        list_73 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 46), 'list')
        # Adding type elements to the builtin type 'list' instance (line 119)
        
        # Applying the binary operator 'or' (line 119)
        result_or_keyword_74 = python_operator(stypy.reporting.localization.Localization(__file__, 119, 29), 'or', extra_objects_72, list_73)
        
        # Getting the type of 'self' (line 119)
        self_75 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 8), 'self')
        # Setting the type of the member 'extra_objects' of a type (line 119)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 8), self_75, 'extra_objects', result_or_keyword_74)
        
        # Assigning a BoolOp to a Attribute (line 120):
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_compile_args' (line 120)
        extra_compile_args_76 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 34), 'extra_compile_args')
        
        # Obtaining an instance of the builtin type 'list' (line 120)
        list_77 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 56), 'list')
        # Adding type elements to the builtin type 'list' instance (line 120)
        
        # Applying the binary operator 'or' (line 120)
        result_or_keyword_78 = python_operator(stypy.reporting.localization.Localization(__file__, 120, 34), 'or', extra_compile_args_76, list_77)
        
        # Getting the type of 'self' (line 120)
        self_79 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'self')
        # Setting the type of the member 'extra_compile_args' of a type (line 120)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 8), self_79, 'extra_compile_args', result_or_keyword_78)
        
        # Assigning a BoolOp to a Attribute (line 121):
        
        # Evaluating a boolean operation
        # Getting the type of 'extra_link_args' (line 121)
        extra_link_args_80 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 31), 'extra_link_args')
        
        # Obtaining an instance of the builtin type 'list' (line 121)
        list_81 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 121, 50), 'list')
        # Adding type elements to the builtin type 'list' instance (line 121)
        
        # Applying the binary operator 'or' (line 121)
        result_or_keyword_82 = python_operator(stypy.reporting.localization.Localization(__file__, 121, 31), 'or', extra_link_args_80, list_81)
        
        # Getting the type of 'self' (line 121)
        self_83 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'self')
        # Setting the type of the member 'extra_link_args' of a type (line 121)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 121, 8), self_83, 'extra_link_args', result_or_keyword_82)
        
        # Assigning a BoolOp to a Attribute (line 122):
        
        # Evaluating a boolean operation
        # Getting the type of 'export_symbols' (line 122)
        export_symbols_84 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 30), 'export_symbols')
        
        # Obtaining an instance of the builtin type 'list' (line 122)
        list_85 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 48), 'list')
        # Adding type elements to the builtin type 'list' instance (line 122)
        
        # Applying the binary operator 'or' (line 122)
        result_or_keyword_86 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 30), 'or', export_symbols_84, list_85)
        
        # Getting the type of 'self' (line 122)
        self_87 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 8), 'self')
        # Setting the type of the member 'export_symbols' of a type (line 122)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 8), self_87, 'export_symbols', result_or_keyword_86)
        
        # Assigning a BoolOp to a Attribute (line 123):
        
        # Evaluating a boolean operation
        # Getting the type of 'swig_opts' (line 123)
        swig_opts_88 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 25), 'swig_opts')
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_89 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 38), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        
        # Applying the binary operator 'or' (line 123)
        result_or_keyword_90 = python_operator(stypy.reporting.localization.Localization(__file__, 123, 25), 'or', swig_opts_88, list_89)
        
        # Getting the type of 'self' (line 123)
        self_91 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'self')
        # Setting the type of the member 'swig_opts' of a type (line 123)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 8), self_91, 'swig_opts', result_or_keyword_90)
        
        # Assigning a BoolOp to a Attribute (line 124):
        
        # Evaluating a boolean operation
        # Getting the type of 'depends' (line 124)
        depends_92 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'depends')
        
        # Obtaining an instance of the builtin type 'list' (line 124)
        list_93 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 124, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 124)
        
        # Applying the binary operator 'or' (line 124)
        result_or_keyword_94 = python_operator(stypy.reporting.localization.Localization(__file__, 124, 23), 'or', depends_92, list_93)
        
        # Getting the type of 'self' (line 124)
        self_95 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'self')
        # Setting the type of the member 'depends' of a type (line 124)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 8), self_95, 'depends', result_or_keyword_94)
        
        # Assigning a Name to a Attribute (line 125):
        # Getting the type of 'language' (line 125)
        language_96 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 24), 'language')
        # Getting the type of 'self' (line 125)
        self_97 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'self')
        # Setting the type of the member 'language' of a type (line 125)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 125, 8), self_97, 'language', language_96)
        
        
        # Call to len(...): (line 128)
        # Processing the call arguments (line 128)
        # Getting the type of 'kw' (line 128)
        kw_99 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 15), 'kw', False)
        # Processing the call keyword arguments (line 128)
        kwargs_100 = {}
        # Getting the type of 'len' (line 128)
        len_98 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 11), 'len', False)
        # Calling len(args, kwargs) (line 128)
        len_call_result_101 = invoke(stypy.reporting.localization.Localization(__file__, 128, 11), len_98, *[kw_99], **kwargs_100)
        
        # Testing the type of an if condition (line 128)
        if_condition_102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 128, 8), len_call_result_101)
        # Assigning a type to the variable 'if_condition_102' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'if_condition_102', if_condition_102)
        # SSA begins for if statement (line 128)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 129):
        
        # Call to keys(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_105 = {}
        # Getting the type of 'kw' (line 129)
        kw_103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'kw', False)
        # Obtaining the member 'keys' of a type (line 129)
        keys_104 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), kw_103, 'keys')
        # Calling keys(args, kwargs) (line 129)
        keys_call_result_106 = invoke(stypy.reporting.localization.Localization(__file__, 129, 16), keys_104, *[], **kwargs_105)
        
        # Assigning a type to the variable 'L' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 12), 'L', keys_call_result_106)
        
        # Call to sort(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_109 = {}
        # Getting the type of 'L' (line 129)
        L_107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 28), 'L', False)
        # Obtaining the member 'sort' of a type (line 129)
        sort_108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 28), L_107, 'sort')
        # Calling sort(args, kwargs) (line 129)
        sort_call_result_110 = invoke(stypy.reporting.localization.Localization(__file__, 129, 28), sort_108, *[], **kwargs_109)
        
        
        # Assigning a Call to a Name (line 130):
        
        # Call to map(...): (line 130)
        # Processing the call arguments (line 130)
        # Getting the type of 'repr' (line 130)
        repr_112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 20), 'repr', False)
        # Getting the type of 'L' (line 130)
        L_113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 26), 'L', False)
        # Processing the call keyword arguments (line 130)
        kwargs_114 = {}
        # Getting the type of 'map' (line 130)
        map_111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 16), 'map', False)
        # Calling map(args, kwargs) (line 130)
        map_call_result_115 = invoke(stypy.reporting.localization.Localization(__file__, 130, 16), map_111, *[repr_112, L_113], **kwargs_114)
        
        # Assigning a type to the variable 'L' (line 130)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 130, 12), 'L', map_call_result_115)
        
        # Assigning a BinOp to a Name (line 131):
        str_116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 18), 'str', 'Unknown Extension options: ')
        
        # Call to join(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'L' (line 131)
        L_119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 62), 'L', False)
        str_120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 131, 65), 'str', ', ')
        # Processing the call keyword arguments (line 131)
        kwargs_121 = {}
        # Getting the type of 'string' (line 131)
        string_117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 50), 'string', False)
        # Obtaining the member 'join' of a type (line 131)
        join_118 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 50), string_117, 'join')
        # Calling join(args, kwargs) (line 131)
        join_call_result_122 = invoke(stypy.reporting.localization.Localization(__file__, 131, 50), join_118, *[L_119, str_120], **kwargs_121)
        
        # Applying the binary operator '+' (line 131)
        result_add_123 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 18), '+', str_116, join_call_result_122)
        
        # Assigning a type to the variable 'msg' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 12), 'msg', result_add_123)
        
        # Type idiom detected: calculating its left and rigth part (line 132)
        # Getting the type of 'warnings' (line 132)
        warnings_124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'warnings')
        # Getting the type of 'None' (line 132)
        None_125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 31), 'None')
        
        (may_be_126, more_types_in_union_127) = may_not_be_none(warnings_124, None_125)

        if may_be_126:

            if more_types_in_union_127:
                # Runtime conditional SSA (line 132)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to warn(...): (line 133)
            # Processing the call arguments (line 133)
            # Getting the type of 'msg' (line 133)
            msg_130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 30), 'msg', False)
            # Processing the call keyword arguments (line 133)
            kwargs_131 = {}
            # Getting the type of 'warnings' (line 133)
            warnings_128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 16), 'warnings', False)
            # Obtaining the member 'warn' of a type (line 133)
            warn_129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 16), warnings_128, 'warn')
            # Calling warn(args, kwargs) (line 133)
            warn_call_result_132 = invoke(stypy.reporting.localization.Localization(__file__, 133, 16), warn_129, *[msg_130], **kwargs_131)
            

            if more_types_in_union_127:
                # Runtime conditional SSA for else branch (line 132)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_126) or more_types_in_union_127):
            
            # Call to write(...): (line 135)
            # Processing the call arguments (line 135)
            # Getting the type of 'msg' (line 135)
            msg_136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'msg', False)
            str_137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 135, 39), 'str', '\n')
            # Applying the binary operator '+' (line 135)
            result_add_138 = python_operator(stypy.reporting.localization.Localization(__file__, 135, 33), '+', msg_136, str_137)
            
            # Processing the call keyword arguments (line 135)
            kwargs_139 = {}
            # Getting the type of 'sys' (line 135)
            sys_133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'sys', False)
            # Obtaining the member 'stderr' of a type (line 135)
            stderr_134 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), sys_133, 'stderr')
            # Obtaining the member 'write' of a type (line 135)
            write_135 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), stderr_134, 'write')
            # Calling write(args, kwargs) (line 135)
            write_call_result_140 = invoke(stypy.reporting.localization.Localization(__file__, 135, 16), write_135, *[result_add_138], **kwargs_139)
            

            if (may_be_126 and more_types_in_union_127):
                # SSA join for if statement (line 132)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for if statement (line 128)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'Extension' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'Extension', Extension)

@norecursion
def read_setup_file(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'read_setup_file'
    module_type_store = module_type_store.open_function_context('read_setup_file', 139, 0, False)
    
    # Passed parameters checking function
    read_setup_file.stypy_localization = localization
    read_setup_file.stypy_type_of_self = None
    read_setup_file.stypy_type_store = module_type_store
    read_setup_file.stypy_function_name = 'read_setup_file'
    read_setup_file.stypy_param_names_list = ['filename']
    read_setup_file.stypy_varargs_param_name = None
    read_setup_file.stypy_kwargs_param_name = None
    read_setup_file.stypy_call_defaults = defaults
    read_setup_file.stypy_call_varargs = varargs
    read_setup_file.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'read_setup_file', ['filename'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'read_setup_file', localization, ['filename'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'read_setup_file(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 140, 4))
    
    # 'from distutils.sysconfig import parse_makefile, expand_makefile_vars, _variable_rx' statement (line 140)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_141 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 140, 4), 'distutils.sysconfig')

    if (type(import_141) is not StypyTypeError):

        if (import_141 != 'pyd_module'):
            __import__(import_141)
            sys_modules_142 = sys.modules[import_141]
            import_from_module(stypy.reporting.localization.Localization(__file__, 140, 4), 'distutils.sysconfig', sys_modules_142.module_type_store, module_type_store, ['parse_makefile', 'expand_makefile_vars', '_variable_rx'])
            nest_module(stypy.reporting.localization.Localization(__file__, 140, 4), __file__, sys_modules_142, sys_modules_142.module_type_store, module_type_store)
        else:
            from distutils.sysconfig import parse_makefile, expand_makefile_vars, _variable_rx

            import_from_module(stypy.reporting.localization.Localization(__file__, 140, 4), 'distutils.sysconfig', None, module_type_store, ['parse_makefile', 'expand_makefile_vars', '_variable_rx'], [parse_makefile, expand_makefile_vars, _variable_rx])

    else:
        # Assigning a type to the variable 'distutils.sysconfig' (line 140)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 4), 'distutils.sysconfig', import_141)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 142, 4))
    
    # 'from distutils.text_file import TextFile' statement (line 142)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_143 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 142, 4), 'distutils.text_file')

    if (type(import_143) is not StypyTypeError):

        if (import_143 != 'pyd_module'):
            __import__(import_143)
            sys_modules_144 = sys.modules[import_143]
            import_from_module(stypy.reporting.localization.Localization(__file__, 142, 4), 'distutils.text_file', sys_modules_144.module_type_store, module_type_store, ['TextFile'])
            nest_module(stypy.reporting.localization.Localization(__file__, 142, 4), __file__, sys_modules_144, sys_modules_144.module_type_store, module_type_store)
        else:
            from distutils.text_file import TextFile

            import_from_module(stypy.reporting.localization.Localization(__file__, 142, 4), 'distutils.text_file', None, module_type_store, ['TextFile'], [TextFile])

    else:
        # Assigning a type to the variable 'distutils.text_file' (line 142)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 142, 4), 'distutils.text_file', import_143)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 143, 4))
    
    # 'from distutils.util import split_quoted' statement (line 143)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/')
    import_145 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 143, 4), 'distutils.util')

    if (type(import_145) is not StypyTypeError):

        if (import_145 != 'pyd_module'):
            __import__(import_145)
            sys_modules_146 = sys.modules[import_145]
            import_from_module(stypy.reporting.localization.Localization(__file__, 143, 4), 'distutils.util', sys_modules_146.module_type_store, module_type_store, ['split_quoted'])
            nest_module(stypy.reporting.localization.Localization(__file__, 143, 4), __file__, sys_modules_146, sys_modules_146.module_type_store, module_type_store)
        else:
            from distutils.util import split_quoted

            import_from_module(stypy.reporting.localization.Localization(__file__, 143, 4), 'distutils.util', None, module_type_store, ['split_quoted'], [split_quoted])

    else:
        # Assigning a type to the variable 'distutils.util' (line 143)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 143, 4), 'distutils.util', import_145)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/')
    
    
    # Assigning a Call to a Name (line 146):
    
    # Call to parse_makefile(...): (line 146)
    # Processing the call arguments (line 146)
    # Getting the type of 'filename' (line 146)
    filename_148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 26), 'filename', False)
    # Processing the call keyword arguments (line 146)
    kwargs_149 = {}
    # Getting the type of 'parse_makefile' (line 146)
    parse_makefile_147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 11), 'parse_makefile', False)
    # Calling parse_makefile(args, kwargs) (line 146)
    parse_makefile_call_result_150 = invoke(stypy.reporting.localization.Localization(__file__, 146, 11), parse_makefile_147, *[filename_148], **kwargs_149)
    
    # Assigning a type to the variable 'vars' (line 146)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 146, 4), 'vars', parse_makefile_call_result_150)
    
    # Assigning a Call to a Name (line 150):
    
    # Call to TextFile(...): (line 150)
    # Processing the call arguments (line 150)
    # Getting the type of 'filename' (line 150)
    filename_152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 20), 'filename', False)
    # Processing the call keyword arguments (line 150)
    int_153 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 35), 'int')
    keyword_154 = int_153
    int_155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 50), 'int')
    keyword_156 = int_155
    int_157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 151, 64), 'int')
    keyword_158 = int_157
    int_159 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 30), 'int')
    keyword_160 = int_159
    int_161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 152, 43), 'int')
    keyword_162 = int_161
    kwargs_163 = {'strip_comments': keyword_154, 'lstrip_ws': keyword_160, 'rstrip_ws': keyword_162, 'join_lines': keyword_158, 'skip_blanks': keyword_156}
    # Getting the type of 'TextFile' (line 150)
    TextFile_151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 11), 'TextFile', False)
    # Calling TextFile(args, kwargs) (line 150)
    TextFile_call_result_164 = invoke(stypy.reporting.localization.Localization(__file__, 150, 11), TextFile_151, *[filename_152], **kwargs_163)
    
    # Assigning a type to the variable 'file' (line 150)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 4), 'file', TextFile_call_result_164)
    
    # Try-finally block (line 153)
    
    # Assigning a List to a Name (line 154):
    
    # Obtaining an instance of the builtin type 'list' (line 154)
    list_165 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 154, 21), 'list')
    # Adding type elements to the builtin type 'list' instance (line 154)
    
    # Assigning a type to the variable 'extensions' (line 154)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 154, 8), 'extensions', list_165)
    
    int_166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 156, 14), 'int')
    # Testing the type of an if condition (line 156)
    is_suitable_condition(stypy.reporting.localization.Localization(__file__, 156, 8), int_166)
    # SSA begins for while statement (line 156)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'while loop')
    
    # Assigning a Call to a Name (line 157):
    
    # Call to readline(...): (line 157)
    # Processing the call keyword arguments (line 157)
    kwargs_169 = {}
    # Getting the type of 'file' (line 157)
    file_167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 157, 19), 'file', False)
    # Obtaining the member 'readline' of a type (line 157)
    readline_168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 157, 19), file_167, 'readline')
    # Calling readline(args, kwargs) (line 157)
    readline_call_result_170 = invoke(stypy.reporting.localization.Localization(__file__, 157, 19), readline_168, *[], **kwargs_169)
    
    # Assigning a type to the variable 'line' (line 157)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 157, 12), 'line', readline_call_result_170)
    
    # Type idiom detected: calculating its left and rigth part (line 158)
    # Getting the type of 'line' (line 158)
    line_171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'line')
    # Getting the type of 'None' (line 158)
    None_172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 23), 'None')
    
    (may_be_173, more_types_in_union_174) = may_be_none(line_171, None_172)

    if may_be_173:

        if more_types_in_union_174:
            # Runtime conditional SSA (line 158)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store


        if more_types_in_union_174:
            # SSA join for if statement (line 158)
            module_type_store = module_type_store.join_ssa_context()


    
    
    
    # Call to match(...): (line 160)
    # Processing the call arguments (line 160)
    # Getting the type of 'line' (line 160)
    line_177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 34), 'line', False)
    # Processing the call keyword arguments (line 160)
    kwargs_178 = {}
    # Getting the type of '_variable_rx' (line 160)
    _variable_rx_175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 160, 15), '_variable_rx', False)
    # Obtaining the member 'match' of a type (line 160)
    match_176 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 160, 15), _variable_rx_175, 'match')
    # Calling match(args, kwargs) (line 160)
    match_call_result_179 = invoke(stypy.reporting.localization.Localization(__file__, 160, 15), match_176, *[line_177], **kwargs_178)
    
    # Testing the type of an if condition (line 160)
    if_condition_180 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 160, 12), match_call_result_179)
    # Assigning a type to the variable 'if_condition_180' (line 160)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 160, 12), 'if_condition_180', if_condition_180)
    # SSA begins for if statement (line 160)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    
    
    # Obtaining the type of the subscript
    int_181 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 24), 'int')
    # Getting the type of 'line' (line 163)
    line_182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 19), 'line')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 19), line_182, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_184 = invoke(stypy.reporting.localization.Localization(__file__, 163, 19), getitem___183, int_181)
    
    
    # Obtaining the type of the subscript
    int_185 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 35), 'int')
    # Getting the type of 'line' (line 163)
    line_186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 30), 'line')
    # Obtaining the member '__getitem__' of a type (line 163)
    getitem___187 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 30), line_186, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 163)
    subscript_call_result_188 = invoke(stypy.reporting.localization.Localization(__file__, 163, 30), getitem___187, int_185)
    
    # Applying the binary operator '==' (line 163)
    result_eq_189 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 19), '==', subscript_call_result_184, subscript_call_result_188)
    str_190 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 163, 42), 'str', '*')
    # Applying the binary operator '==' (line 163)
    result_eq_191 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 19), '==', subscript_call_result_188, str_190)
    # Applying the binary operator '&' (line 163)
    result_and__192 = python_operator(stypy.reporting.localization.Localization(__file__, 163, 19), '&', result_eq_189, result_eq_191)
    
    # Testing the type of an if condition (line 163)
    if_condition_193 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 163, 16), result_and__192)
    # Assigning a type to the variable 'if_condition_193' (line 163)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 163, 16), 'if_condition_193', if_condition_193)
    # SSA begins for if statement (line 163)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to warn(...): (line 164)
    # Processing the call arguments (line 164)
    str_196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 164, 30), 'str', "'%s' lines not handled yet")
    # Getting the type of 'line' (line 164)
    line_197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 61), 'line', False)
    # Applying the binary operator '%' (line 164)
    result_mod_198 = python_operator(stypy.reporting.localization.Localization(__file__, 164, 30), '%', str_196, line_197)
    
    # Processing the call keyword arguments (line 164)
    kwargs_199 = {}
    # Getting the type of 'file' (line 164)
    file_194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 20), 'file', False)
    # Obtaining the member 'warn' of a type (line 164)
    warn_195 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 164, 20), file_194, 'warn')
    # Calling warn(args, kwargs) (line 164)
    warn_call_result_200 = invoke(stypy.reporting.localization.Localization(__file__, 164, 20), warn_195, *[result_mod_198], **kwargs_199)
    
    # SSA join for if statement (line 163)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 160)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 168):
    
    # Call to expand_makefile_vars(...): (line 168)
    # Processing the call arguments (line 168)
    # Getting the type of 'line' (line 168)
    line_202 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 40), 'line', False)
    # Getting the type of 'vars' (line 168)
    vars_203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 46), 'vars', False)
    # Processing the call keyword arguments (line 168)
    kwargs_204 = {}
    # Getting the type of 'expand_makefile_vars' (line 168)
    expand_makefile_vars_201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 19), 'expand_makefile_vars', False)
    # Calling expand_makefile_vars(args, kwargs) (line 168)
    expand_makefile_vars_call_result_205 = invoke(stypy.reporting.localization.Localization(__file__, 168, 19), expand_makefile_vars_201, *[line_202, vars_203], **kwargs_204)
    
    # Assigning a type to the variable 'line' (line 168)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 12), 'line', expand_makefile_vars_call_result_205)
    
    # Assigning a Call to a Name (line 169):
    
    # Call to split_quoted(...): (line 169)
    # Processing the call arguments (line 169)
    # Getting the type of 'line' (line 169)
    line_207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 33), 'line', False)
    # Processing the call keyword arguments (line 169)
    kwargs_208 = {}
    # Getting the type of 'split_quoted' (line 169)
    split_quoted_206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 169, 20), 'split_quoted', False)
    # Calling split_quoted(args, kwargs) (line 169)
    split_quoted_call_result_209 = invoke(stypy.reporting.localization.Localization(__file__, 169, 20), split_quoted_206, *[line_207], **kwargs_208)
    
    # Assigning a type to the variable 'words' (line 169)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 12), 'words', split_quoted_call_result_209)
    
    # Assigning a Subscript to a Name (line 178):
    
    # Obtaining the type of the subscript
    int_210 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 178, 27), 'int')
    # Getting the type of 'words' (line 178)
    words_211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 21), 'words')
    # Obtaining the member '__getitem__' of a type (line 178)
    getitem___212 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 21), words_211, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 178)
    subscript_call_result_213 = invoke(stypy.reporting.localization.Localization(__file__, 178, 21), getitem___212, int_210)
    
    # Assigning a type to the variable 'module' (line 178)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'module', subscript_call_result_213)
    
    # Assigning a Call to a Name (line 179):
    
    # Call to Extension(...): (line 179)
    # Processing the call arguments (line 179)
    # Getting the type of 'module' (line 179)
    module_215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 28), 'module', False)
    
    # Obtaining an instance of the builtin type 'list' (line 179)
    list_216 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 179, 36), 'list')
    # Adding type elements to the builtin type 'list' instance (line 179)
    
    # Processing the call keyword arguments (line 179)
    kwargs_217 = {}
    # Getting the type of 'Extension' (line 179)
    Extension_214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 18), 'Extension', False)
    # Calling Extension(args, kwargs) (line 179)
    Extension_call_result_218 = invoke(stypy.reporting.localization.Localization(__file__, 179, 18), Extension_214, *[module_215, list_216], **kwargs_217)
    
    # Assigning a type to the variable 'ext' (line 179)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 12), 'ext', Extension_call_result_218)
    
    # Assigning a Name to a Name (line 180):
    # Getting the type of 'None' (line 180)
    None_219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 180, 31), 'None')
    # Assigning a type to the variable 'append_next_word' (line 180)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 180, 12), 'append_next_word', None_219)
    
    
    # Obtaining the type of the subscript
    int_220 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 182, 30), 'int')
    slice_221 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 182, 24), int_220, None, None)
    # Getting the type of 'words' (line 182)
    words_222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 24), 'words')
    # Obtaining the member '__getitem__' of a type (line 182)
    getitem___223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 182, 24), words_222, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 182)
    subscript_call_result_224 = invoke(stypy.reporting.localization.Localization(__file__, 182, 24), getitem___223, slice_221)
    
    # Testing the type of a for loop iterable (line 182)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 12), subscript_call_result_224)
    # Getting the type of the for loop variable (line 182)
    for_loop_var_225 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 12), subscript_call_result_224)
    # Assigning a type to the variable 'word' (line 182)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 12), 'word', for_loop_var_225)
    # SSA begins for a for statement (line 182)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Type idiom detected: calculating its left and rigth part (line 183)
    # Getting the type of 'append_next_word' (line 183)
    append_next_word_226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 16), 'append_next_word')
    # Getting the type of 'None' (line 183)
    None_227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 43), 'None')
    
    (may_be_228, more_types_in_union_229) = may_not_be_none(append_next_word_226, None_227)

    if may_be_228:

        if more_types_in_union_229:
            # Runtime conditional SSA (line 183)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
        else:
            module_type_store = module_type_store

        
        # Call to append(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'word' (line 184)
        word_232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 44), 'word', False)
        # Processing the call keyword arguments (line 184)
        kwargs_233 = {}
        # Getting the type of 'append_next_word' (line 184)
        append_next_word_230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 20), 'append_next_word', False)
        # Obtaining the member 'append' of a type (line 184)
        append_231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 184, 20), append_next_word_230, 'append')
        # Calling append(args, kwargs) (line 184)
        append_call_result_234 = invoke(stypy.reporting.localization.Localization(__file__, 184, 20), append_231, *[word_232], **kwargs_233)
        
        
        # Assigning a Name to a Name (line 185):
        # Getting the type of 'None' (line 185)
        None_235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 39), 'None')
        # Assigning a type to the variable 'append_next_word' (line 185)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 185, 20), 'append_next_word', None_235)

        if more_types_in_union_229:
            # SSA join for if statement (line 183)
            module_type_store = module_type_store.join_ssa_context()


    
    
    # Assigning a Subscript to a Name (line 188):
    
    # Obtaining the type of the subscript
    int_236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 188, 48), 'int')
    
    # Call to splitext(...): (line 188)
    # Processing the call arguments (line 188)
    # Getting the type of 'word' (line 188)
    word_240 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 42), 'word', False)
    # Processing the call keyword arguments (line 188)
    kwargs_241 = {}
    # Getting the type of 'os' (line 188)
    os_237 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 188, 25), 'os', False)
    # Obtaining the member 'path' of a type (line 188)
    path_238 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 25), os_237, 'path')
    # Obtaining the member 'splitext' of a type (line 188)
    splitext_239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 25), path_238, 'splitext')
    # Calling splitext(args, kwargs) (line 188)
    splitext_call_result_242 = invoke(stypy.reporting.localization.Localization(__file__, 188, 25), splitext_239, *[word_240], **kwargs_241)
    
    # Obtaining the member '__getitem__' of a type (line 188)
    getitem___243 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 188, 25), splitext_call_result_242, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 188)
    subscript_call_result_244 = invoke(stypy.reporting.localization.Localization(__file__, 188, 25), getitem___243, int_236)
    
    # Assigning a type to the variable 'suffix' (line 188)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 188, 16), 'suffix', subscript_call_result_244)
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 30), 'int')
    int_246 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 32), 'int')
    slice_247 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 189, 25), int_245, int_246, None)
    # Getting the type of 'word' (line 189)
    word_248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 25), 'word')
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___249 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 25), word_248, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_250 = invoke(stypy.reporting.localization.Localization(__file__, 189, 25), getitem___249, slice_247)
    
    # Assigning a type to the variable 'switch' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 16), 'switch', subscript_call_result_250)
    
    # Assigning a Subscript to a Name (line 189):
    
    # Obtaining the type of the subscript
    int_251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 189, 50), 'int')
    slice_252 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 189, 45), int_251, None, None)
    # Getting the type of 'word' (line 189)
    word_253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 45), 'word')
    # Obtaining the member '__getitem__' of a type (line 189)
    getitem___254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 189, 45), word_253, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 189)
    subscript_call_result_255 = invoke(stypy.reporting.localization.Localization(__file__, 189, 45), getitem___254, slice_252)
    
    # Assigning a type to the variable 'value' (line 189)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 37), 'value', subscript_call_result_255)
    
    
    # Getting the type of 'suffix' (line 191)
    suffix_256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 19), 'suffix')
    
    # Obtaining an instance of the builtin type 'tuple' (line 191)
    tuple_257 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 30), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 191)
    # Adding element type (line 191)
    str_258 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 30), 'str', '.c')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), tuple_257, str_258)
    # Adding element type (line 191)
    str_259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 36), 'str', '.cc')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), tuple_257, str_259)
    # Adding element type (line 191)
    str_260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 43), 'str', '.cpp')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), tuple_257, str_260)
    # Adding element type (line 191)
    str_261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 51), 'str', '.cxx')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), tuple_257, str_261)
    # Adding element type (line 191)
    str_262 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 59), 'str', '.c++')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), tuple_257, str_262)
    # Adding element type (line 191)
    str_263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 67), 'str', '.m')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), tuple_257, str_263)
    # Adding element type (line 191)
    str_264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 73), 'str', '.mm')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 191, 30), tuple_257, str_264)
    
    # Applying the binary operator 'in' (line 191)
    result_contains_265 = python_operator(stypy.reporting.localization.Localization(__file__, 191, 19), 'in', suffix_256, tuple_257)
    
    # Testing the type of an if condition (line 191)
    if_condition_266 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 191, 16), result_contains_265)
    # Assigning a type to the variable 'if_condition_266' (line 191)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 191, 16), 'if_condition_266', if_condition_266)
    # SSA begins for if statement (line 191)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 195)
    # Processing the call arguments (line 195)
    # Getting the type of 'word' (line 195)
    word_270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 39), 'word', False)
    # Processing the call keyword arguments (line 195)
    kwargs_271 = {}
    # Getting the type of 'ext' (line 195)
    ext_267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 195, 20), 'ext', False)
    # Obtaining the member 'sources' of a type (line 195)
    sources_268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 20), ext_267, 'sources')
    # Obtaining the member 'append' of a type (line 195)
    append_269 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 195, 20), sources_268, 'append')
    # Calling append(args, kwargs) (line 195)
    append_call_result_272 = invoke(stypy.reporting.localization.Localization(__file__, 195, 20), append_269, *[word_270], **kwargs_271)
    
    # SSA branch for the else part of an if statement (line 191)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 196)
    switch_273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'switch')
    str_274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 31), 'str', '-I')
    # Applying the binary operator '==' (line 196)
    result_eq_275 = python_operator(stypy.reporting.localization.Localization(__file__, 196, 21), '==', switch_273, str_274)
    
    # Testing the type of an if condition (line 196)
    if_condition_276 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 196, 21), result_eq_275)
    # Assigning a type to the variable 'if_condition_276' (line 196)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'if_condition_276', if_condition_276)
    # SSA begins for if statement (line 196)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 197)
    # Processing the call arguments (line 197)
    # Getting the type of 'value' (line 197)
    value_280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 44), 'value', False)
    # Processing the call keyword arguments (line 197)
    kwargs_281 = {}
    # Getting the type of 'ext' (line 197)
    ext_277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 20), 'ext', False)
    # Obtaining the member 'include_dirs' of a type (line 197)
    include_dirs_278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 20), ext_277, 'include_dirs')
    # Obtaining the member 'append' of a type (line 197)
    append_279 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 20), include_dirs_278, 'append')
    # Calling append(args, kwargs) (line 197)
    append_call_result_282 = invoke(stypy.reporting.localization.Localization(__file__, 197, 20), append_279, *[value_280], **kwargs_281)
    
    # SSA branch for the else part of an if statement (line 196)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 198)
    switch_283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'switch')
    str_284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 198, 31), 'str', '-D')
    # Applying the binary operator '==' (line 198)
    result_eq_285 = python_operator(stypy.reporting.localization.Localization(__file__, 198, 21), '==', switch_283, str_284)
    
    # Testing the type of an if condition (line 198)
    if_condition_286 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 198, 21), result_eq_285)
    # Assigning a type to the variable 'if_condition_286' (line 198)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 21), 'if_condition_286', if_condition_286)
    # SSA begins for if statement (line 198)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Call to a Name (line 199):
    
    # Call to find(...): (line 199)
    # Processing the call arguments (line 199)
    # Getting the type of 'value' (line 199)
    value_289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 41), 'value', False)
    str_290 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 199, 48), 'str', '=')
    # Processing the call keyword arguments (line 199)
    kwargs_291 = {}
    # Getting the type of 'string' (line 199)
    string_287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 29), 'string', False)
    # Obtaining the member 'find' of a type (line 199)
    find_288 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 29), string_287, 'find')
    # Calling find(args, kwargs) (line 199)
    find_call_result_292 = invoke(stypy.reporting.localization.Localization(__file__, 199, 29), find_288, *[value_289, str_290], **kwargs_291)
    
    # Assigning a type to the variable 'equals' (line 199)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 199, 20), 'equals', find_call_result_292)
    
    
    # Getting the type of 'equals' (line 200)
    equals_293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 23), 'equals')
    int_294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 200, 33), 'int')
    # Applying the binary operator '==' (line 200)
    result_eq_295 = python_operator(stypy.reporting.localization.Localization(__file__, 200, 23), '==', equals_293, int_294)
    
    # Testing the type of an if condition (line 200)
    if_condition_296 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 200, 20), result_eq_295)
    # Assigning a type to the variable 'if_condition_296' (line 200)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 200, 20), 'if_condition_296', if_condition_296)
    # SSA begins for if statement (line 200)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 201)
    # Processing the call arguments (line 201)
    
    # Obtaining an instance of the builtin type 'tuple' (line 201)
    tuple_300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 201, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 201)
    # Adding element type (line 201)
    # Getting the type of 'value' (line 201)
    value_301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 50), 'value', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 50), tuple_300, value_301)
    # Adding element type (line 201)
    # Getting the type of 'None' (line 201)
    None_302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 201, 50), tuple_300, None_302)
    
    # Processing the call keyword arguments (line 201)
    kwargs_303 = {}
    # Getting the type of 'ext' (line 201)
    ext_297 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 24), 'ext', False)
    # Obtaining the member 'define_macros' of a type (line 201)
    define_macros_298 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 24), ext_297, 'define_macros')
    # Obtaining the member 'append' of a type (line 201)
    append_299 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 201, 24), define_macros_298, 'append')
    # Calling append(args, kwargs) (line 201)
    append_call_result_304 = invoke(stypy.reporting.localization.Localization(__file__, 201, 24), append_299, *[tuple_300], **kwargs_303)
    
    # SSA branch for the else part of an if statement (line 200)
    module_type_store.open_ssa_branch('else')
    
    # Call to append(...): (line 203)
    # Processing the call arguments (line 203)
    
    # Obtaining an instance of the builtin type 'tuple' (line 203)
    tuple_308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 50), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 203)
    # Adding element type (line 203)
    
    # Obtaining the type of the subscript
    int_309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 203, 56), 'int')
    # Getting the type of 'equals' (line 203)
    equals_310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 58), 'equals', False)
    slice_311 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 203, 50), int_309, equals_310, None)
    # Getting the type of 'value' (line 203)
    value_312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 50), 'value', False)
    # Obtaining the member '__getitem__' of a type (line 203)
    getitem___313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 50), value_312, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 203)
    subscript_call_result_314 = invoke(stypy.reporting.localization.Localization(__file__, 203, 50), getitem___313, slice_311)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 50), tuple_308, subscript_call_result_314)
    # Adding element type (line 203)
    
    # Obtaining the type of the subscript
    # Getting the type of 'equals' (line 204)
    equals_315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 56), 'equals', False)
    int_316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 204, 63), 'int')
    # Applying the binary operator '+' (line 204)
    result_add_317 = python_operator(stypy.reporting.localization.Localization(__file__, 204, 56), '+', equals_315, int_316)
    
    slice_318 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 204, 50), result_add_317, None, None)
    # Getting the type of 'value' (line 204)
    value_319 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 204, 50), 'value', False)
    # Obtaining the member '__getitem__' of a type (line 204)
    getitem___320 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 204, 50), value_319, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 204)
    subscript_call_result_321 = invoke(stypy.reporting.localization.Localization(__file__, 204, 50), getitem___320, slice_318)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 203, 50), tuple_308, subscript_call_result_321)
    
    # Processing the call keyword arguments (line 203)
    kwargs_322 = {}
    # Getting the type of 'ext' (line 203)
    ext_305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 203, 24), 'ext', False)
    # Obtaining the member 'define_macros' of a type (line 203)
    define_macros_306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 24), ext_305, 'define_macros')
    # Obtaining the member 'append' of a type (line 203)
    append_307 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 203, 24), define_macros_306, 'append')
    # Calling append(args, kwargs) (line 203)
    append_call_result_323 = invoke(stypy.reporting.localization.Localization(__file__, 203, 24), append_307, *[tuple_308], **kwargs_322)
    
    # SSA join for if statement (line 200)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 198)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 205)
    switch_324 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'switch')
    str_325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 205, 31), 'str', '-U')
    # Applying the binary operator '==' (line 205)
    result_eq_326 = python_operator(stypy.reporting.localization.Localization(__file__, 205, 21), '==', switch_324, str_325)
    
    # Testing the type of an if condition (line 205)
    if_condition_327 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 205, 21), result_eq_326)
    # Assigning a type to the variable 'if_condition_327' (line 205)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 205, 21), 'if_condition_327', if_condition_327)
    # SSA begins for if statement (line 205)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 206)
    # Processing the call arguments (line 206)
    # Getting the type of 'value' (line 206)
    value_331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 44), 'value', False)
    # Processing the call keyword arguments (line 206)
    kwargs_332 = {}
    # Getting the type of 'ext' (line 206)
    ext_328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 206, 20), 'ext', False)
    # Obtaining the member 'undef_macros' of a type (line 206)
    undef_macros_329 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 20), ext_328, 'undef_macros')
    # Obtaining the member 'append' of a type (line 206)
    append_330 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 206, 20), undef_macros_329, 'append')
    # Calling append(args, kwargs) (line 206)
    append_call_result_333 = invoke(stypy.reporting.localization.Localization(__file__, 206, 20), append_330, *[value_331], **kwargs_332)
    
    # SSA branch for the else part of an if statement (line 205)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 207)
    switch_334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), 'switch')
    str_335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 207, 31), 'str', '-C')
    # Applying the binary operator '==' (line 207)
    result_eq_336 = python_operator(stypy.reporting.localization.Localization(__file__, 207, 21), '==', switch_334, str_335)
    
    # Testing the type of an if condition (line 207)
    if_condition_337 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 207, 21), result_eq_336)
    # Assigning a type to the variable 'if_condition_337' (line 207)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 207, 21), 'if_condition_337', if_condition_337)
    # SSA begins for if statement (line 207)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 208)
    # Processing the call arguments (line 208)
    # Getting the type of 'word' (line 208)
    word_341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 50), 'word', False)
    # Processing the call keyword arguments (line 208)
    kwargs_342 = {}
    # Getting the type of 'ext' (line 208)
    ext_338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 20), 'ext', False)
    # Obtaining the member 'extra_compile_args' of a type (line 208)
    extra_compile_args_339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), ext_338, 'extra_compile_args')
    # Obtaining the member 'append' of a type (line 208)
    append_340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 20), extra_compile_args_339, 'append')
    # Calling append(args, kwargs) (line 208)
    append_call_result_343 = invoke(stypy.reporting.localization.Localization(__file__, 208, 20), append_340, *[word_341], **kwargs_342)
    
    # SSA branch for the else part of an if statement (line 207)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 209)
    switch_344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'switch')
    str_345 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 209, 31), 'str', '-l')
    # Applying the binary operator '==' (line 209)
    result_eq_346 = python_operator(stypy.reporting.localization.Localization(__file__, 209, 21), '==', switch_344, str_345)
    
    # Testing the type of an if condition (line 209)
    if_condition_347 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 209, 21), result_eq_346)
    # Assigning a type to the variable 'if_condition_347' (line 209)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 209, 21), 'if_condition_347', if_condition_347)
    # SSA begins for if statement (line 209)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 210)
    # Processing the call arguments (line 210)
    # Getting the type of 'value' (line 210)
    value_351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 41), 'value', False)
    # Processing the call keyword arguments (line 210)
    kwargs_352 = {}
    # Getting the type of 'ext' (line 210)
    ext_348 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 210, 20), 'ext', False)
    # Obtaining the member 'libraries' of a type (line 210)
    libraries_349 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 20), ext_348, 'libraries')
    # Obtaining the member 'append' of a type (line 210)
    append_350 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 210, 20), libraries_349, 'append')
    # Calling append(args, kwargs) (line 210)
    append_call_result_353 = invoke(stypy.reporting.localization.Localization(__file__, 210, 20), append_350, *[value_351], **kwargs_352)
    
    # SSA branch for the else part of an if statement (line 209)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 211)
    switch_354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'switch')
    str_355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 211, 31), 'str', '-L')
    # Applying the binary operator '==' (line 211)
    result_eq_356 = python_operator(stypy.reporting.localization.Localization(__file__, 211, 21), '==', switch_354, str_355)
    
    # Testing the type of an if condition (line 211)
    if_condition_357 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 211, 21), result_eq_356)
    # Assigning a type to the variable 'if_condition_357' (line 211)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 211, 21), 'if_condition_357', if_condition_357)
    # SSA begins for if statement (line 211)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 212)
    # Processing the call arguments (line 212)
    # Getting the type of 'value' (line 212)
    value_361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 44), 'value', False)
    # Processing the call keyword arguments (line 212)
    kwargs_362 = {}
    # Getting the type of 'ext' (line 212)
    ext_358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 212, 20), 'ext', False)
    # Obtaining the member 'library_dirs' of a type (line 212)
    library_dirs_359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), ext_358, 'library_dirs')
    # Obtaining the member 'append' of a type (line 212)
    append_360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 212, 20), library_dirs_359, 'append')
    # Calling append(args, kwargs) (line 212)
    append_call_result_363 = invoke(stypy.reporting.localization.Localization(__file__, 212, 20), append_360, *[value_361], **kwargs_362)
    
    # SSA branch for the else part of an if statement (line 211)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 213)
    switch_364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 213, 21), 'switch')
    str_365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 213, 31), 'str', '-R')
    # Applying the binary operator '==' (line 213)
    result_eq_366 = python_operator(stypy.reporting.localization.Localization(__file__, 213, 21), '==', switch_364, str_365)
    
    # Testing the type of an if condition (line 213)
    if_condition_367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 213, 21), result_eq_366)
    # Assigning a type to the variable 'if_condition_367' (line 213)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 213, 21), 'if_condition_367', if_condition_367)
    # SSA begins for if statement (line 213)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 214)
    # Processing the call arguments (line 214)
    # Getting the type of 'value' (line 214)
    value_371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 52), 'value', False)
    # Processing the call keyword arguments (line 214)
    kwargs_372 = {}
    # Getting the type of 'ext' (line 214)
    ext_368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 214, 20), 'ext', False)
    # Obtaining the member 'runtime_library_dirs' of a type (line 214)
    runtime_library_dirs_369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), ext_368, 'runtime_library_dirs')
    # Obtaining the member 'append' of a type (line 214)
    append_370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 214, 20), runtime_library_dirs_369, 'append')
    # Calling append(args, kwargs) (line 214)
    append_call_result_373 = invoke(stypy.reporting.localization.Localization(__file__, 214, 20), append_370, *[value_371], **kwargs_372)
    
    # SSA branch for the else part of an if statement (line 213)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'word' (line 215)
    word_374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'word')
    str_375 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 215, 29), 'str', '-rpath')
    # Applying the binary operator '==' (line 215)
    result_eq_376 = python_operator(stypy.reporting.localization.Localization(__file__, 215, 21), '==', word_374, str_375)
    
    # Testing the type of an if condition (line 215)
    if_condition_377 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 215, 21), result_eq_376)
    # Assigning a type to the variable 'if_condition_377' (line 215)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 215, 21), 'if_condition_377', if_condition_377)
    # SSA begins for if statement (line 215)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 216):
    # Getting the type of 'ext' (line 216)
    ext_378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 216, 39), 'ext')
    # Obtaining the member 'runtime_library_dirs' of a type (line 216)
    runtime_library_dirs_379 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 216, 39), ext_378, 'runtime_library_dirs')
    # Assigning a type to the variable 'append_next_word' (line 216)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 216, 20), 'append_next_word', runtime_library_dirs_379)
    # SSA branch for the else part of an if statement (line 215)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'word' (line 217)
    word_380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'word')
    str_381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 217, 29), 'str', '-Xlinker')
    # Applying the binary operator '==' (line 217)
    result_eq_382 = python_operator(stypy.reporting.localization.Localization(__file__, 217, 21), '==', word_380, str_381)
    
    # Testing the type of an if condition (line 217)
    if_condition_383 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 217, 21), result_eq_382)
    # Assigning a type to the variable 'if_condition_383' (line 217)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 217, 21), 'if_condition_383', if_condition_383)
    # SSA begins for if statement (line 217)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 218):
    # Getting the type of 'ext' (line 218)
    ext_384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 218, 39), 'ext')
    # Obtaining the member 'extra_link_args' of a type (line 218)
    extra_link_args_385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 218, 39), ext_384, 'extra_link_args')
    # Assigning a type to the variable 'append_next_word' (line 218)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 218, 20), 'append_next_word', extra_link_args_385)
    # SSA branch for the else part of an if statement (line 217)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'word' (line 219)
    word_386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'word')
    str_387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 219, 29), 'str', '-Xcompiler')
    # Applying the binary operator '==' (line 219)
    result_eq_388 = python_operator(stypy.reporting.localization.Localization(__file__, 219, 21), '==', word_386, str_387)
    
    # Testing the type of an if condition (line 219)
    if_condition_389 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 219, 21), result_eq_388)
    # Assigning a type to the variable 'if_condition_389' (line 219)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 219, 21), 'if_condition_389', if_condition_389)
    # SSA begins for if statement (line 219)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 220):
    # Getting the type of 'ext' (line 220)
    ext_390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 220, 39), 'ext')
    # Obtaining the member 'extra_compile_args' of a type (line 220)
    extra_compile_args_391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 220, 39), ext_390, 'extra_compile_args')
    # Assigning a type to the variable 'append_next_word' (line 220)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 220, 20), 'append_next_word', extra_compile_args_391)
    # SSA branch for the else part of an if statement (line 219)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 221)
    switch_392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'switch')
    str_393 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 221, 31), 'str', '-u')
    # Applying the binary operator '==' (line 221)
    result_eq_394 = python_operator(stypy.reporting.localization.Localization(__file__, 221, 21), '==', switch_392, str_393)
    
    # Testing the type of an if condition (line 221)
    if_condition_395 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 221, 21), result_eq_394)
    # Assigning a type to the variable 'if_condition_395' (line 221)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 221, 21), 'if_condition_395', if_condition_395)
    # SSA begins for if statement (line 221)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 222)
    # Processing the call arguments (line 222)
    # Getting the type of 'word' (line 222)
    word_399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 47), 'word', False)
    # Processing the call keyword arguments (line 222)
    kwargs_400 = {}
    # Getting the type of 'ext' (line 222)
    ext_396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 222, 20), 'ext', False)
    # Obtaining the member 'extra_link_args' of a type (line 222)
    extra_link_args_397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 20), ext_396, 'extra_link_args')
    # Obtaining the member 'append' of a type (line 222)
    append_398 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 222, 20), extra_link_args_397, 'append')
    # Calling append(args, kwargs) (line 222)
    append_call_result_401 = invoke(stypy.reporting.localization.Localization(__file__, 222, 20), append_398, *[word_399], **kwargs_400)
    
    
    
    # Getting the type of 'value' (line 223)
    value_402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 223, 27), 'value')
    # Applying the 'not' unary operator (line 223)
    result_not__403 = python_operator(stypy.reporting.localization.Localization(__file__, 223, 23), 'not', value_402)
    
    # Testing the type of an if condition (line 223)
    if_condition_404 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 223, 20), result_not__403)
    # Assigning a type to the variable 'if_condition_404' (line 223)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 223, 20), 'if_condition_404', if_condition_404)
    # SSA begins for if statement (line 223)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 224):
    # Getting the type of 'ext' (line 224)
    ext_405 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 224, 43), 'ext')
    # Obtaining the member 'extra_link_args' of a type (line 224)
    extra_link_args_406 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 224, 43), ext_405, 'extra_link_args')
    # Assigning a type to the variable 'append_next_word' (line 224)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 224, 24), 'append_next_word', extra_link_args_406)
    # SSA join for if statement (line 223)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 221)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'word' (line 225)
    word_407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 225, 21), 'word')
    str_408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 225, 29), 'str', '-Xcompiler')
    # Applying the binary operator '==' (line 225)
    result_eq_409 = python_operator(stypy.reporting.localization.Localization(__file__, 225, 21), '==', word_407, str_408)
    
    # Testing the type of an if condition (line 225)
    if_condition_410 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 225, 21), result_eq_409)
    # Assigning a type to the variable 'if_condition_410' (line 225)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 225, 21), 'if_condition_410', if_condition_410)
    # SSA begins for if statement (line 225)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 226):
    # Getting the type of 'ext' (line 226)
    ext_411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 226, 39), 'ext')
    # Obtaining the member 'extra_compile_args' of a type (line 226)
    extra_compile_args_412 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 226, 39), ext_411, 'extra_compile_args')
    # Assigning a type to the variable 'append_next_word' (line 226)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 226, 20), 'append_next_word', extra_compile_args_412)
    # SSA branch for the else part of an if statement (line 225)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'switch' (line 227)
    switch_413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'switch')
    str_414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 227, 31), 'str', '-u')
    # Applying the binary operator '==' (line 227)
    result_eq_415 = python_operator(stypy.reporting.localization.Localization(__file__, 227, 21), '==', switch_413, str_414)
    
    # Testing the type of an if condition (line 227)
    if_condition_416 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 227, 21), result_eq_415)
    # Assigning a type to the variable 'if_condition_416' (line 227)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 227, 21), 'if_condition_416', if_condition_416)
    # SSA begins for if statement (line 227)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 228)
    # Processing the call arguments (line 228)
    # Getting the type of 'word' (line 228)
    word_420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 47), 'word', False)
    # Processing the call keyword arguments (line 228)
    kwargs_421 = {}
    # Getting the type of 'ext' (line 228)
    ext_417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 228, 20), 'ext', False)
    # Obtaining the member 'extra_link_args' of a type (line 228)
    extra_link_args_418 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 20), ext_417, 'extra_link_args')
    # Obtaining the member 'append' of a type (line 228)
    append_419 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 228, 20), extra_link_args_418, 'append')
    # Calling append(args, kwargs) (line 228)
    append_call_result_422 = invoke(stypy.reporting.localization.Localization(__file__, 228, 20), append_419, *[word_420], **kwargs_421)
    
    
    
    # Getting the type of 'value' (line 229)
    value_423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 229, 27), 'value')
    # Applying the 'not' unary operator (line 229)
    result_not__424 = python_operator(stypy.reporting.localization.Localization(__file__, 229, 23), 'not', value_423)
    
    # Testing the type of an if condition (line 229)
    if_condition_425 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 229, 20), result_not__424)
    # Assigning a type to the variable 'if_condition_425' (line 229)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 229, 20), 'if_condition_425', if_condition_425)
    # SSA begins for if statement (line 229)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Assigning a Attribute to a Name (line 230):
    # Getting the type of 'ext' (line 230)
    ext_426 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 230, 43), 'ext')
    # Obtaining the member 'extra_link_args' of a type (line 230)
    extra_link_args_427 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 230, 43), ext_426, 'extra_link_args')
    # Assigning a type to the variable 'append_next_word' (line 230)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 230, 24), 'append_next_word', extra_link_args_427)
    # SSA join for if statement (line 229)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA branch for the else part of an if statement (line 227)
    module_type_store.open_ssa_branch('else')
    
    
    # Getting the type of 'suffix' (line 231)
    suffix_428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 'suffix')
    
    # Obtaining an instance of the builtin type 'tuple' (line 231)
    tuple_429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 32), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 231)
    # Adding element type (line 231)
    str_430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 32), 'str', '.a')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 32), tuple_429, str_430)
    # Adding element type (line 231)
    str_431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 38), 'str', '.so')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 32), tuple_429, str_431)
    # Adding element type (line 231)
    str_432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 45), 'str', '.sl')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 32), tuple_429, str_432)
    # Adding element type (line 231)
    str_433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 52), 'str', '.o')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 32), tuple_429, str_433)
    # Adding element type (line 231)
    str_434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 231, 58), 'str', '.dylib')
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 231, 32), tuple_429, str_434)
    
    # Applying the binary operator 'in' (line 231)
    result_contains_435 = python_operator(stypy.reporting.localization.Localization(__file__, 231, 21), 'in', suffix_428, tuple_429)
    
    # Testing the type of an if condition (line 231)
    if_condition_436 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 231, 21), result_contains_435)
    # Assigning a type to the variable 'if_condition_436' (line 231)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 231, 21), 'if_condition_436', if_condition_436)
    # SSA begins for if statement (line 231)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    
    # Call to append(...): (line 236)
    # Processing the call arguments (line 236)
    # Getting the type of 'word' (line 236)
    word_440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 45), 'word', False)
    # Processing the call keyword arguments (line 236)
    kwargs_441 = {}
    # Getting the type of 'ext' (line 236)
    ext_437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 236, 20), 'ext', False)
    # Obtaining the member 'extra_objects' of a type (line 236)
    extra_objects_438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 20), ext_437, 'extra_objects')
    # Obtaining the member 'append' of a type (line 236)
    append_439 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 236, 20), extra_objects_438, 'append')
    # Calling append(args, kwargs) (line 236)
    append_call_result_442 = invoke(stypy.reporting.localization.Localization(__file__, 236, 20), append_439, *[word_440], **kwargs_441)
    
    # SSA branch for the else part of an if statement (line 231)
    module_type_store.open_ssa_branch('else')
    
    # Call to warn(...): (line 238)
    # Processing the call arguments (line 238)
    str_445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 238, 30), 'str', "unrecognized argument '%s'")
    # Getting the type of 'word' (line 238)
    word_446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 61), 'word', False)
    # Applying the binary operator '%' (line 238)
    result_mod_447 = python_operator(stypy.reporting.localization.Localization(__file__, 238, 30), '%', str_445, word_446)
    
    # Processing the call keyword arguments (line 238)
    kwargs_448 = {}
    # Getting the type of 'file' (line 238)
    file_443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 238, 20), 'file', False)
    # Obtaining the member 'warn' of a type (line 238)
    warn_444 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 238, 20), file_443, 'warn')
    # Calling warn(args, kwargs) (line 238)
    warn_call_result_449 = invoke(stypy.reporting.localization.Localization(__file__, 238, 20), warn_444, *[result_mod_447], **kwargs_448)
    
    # SSA join for if statement (line 231)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 227)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 225)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 221)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 219)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 217)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 215)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 213)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 211)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 209)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 207)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 205)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 198)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 196)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for if statement (line 191)
    module_type_store = module_type_store.join_ssa_context()
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 240)
    # Processing the call arguments (line 240)
    # Getting the type of 'ext' (line 240)
    ext_452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 30), 'ext', False)
    # Processing the call keyword arguments (line 240)
    kwargs_453 = {}
    # Getting the type of 'extensions' (line 240)
    extensions_450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 240, 12), 'extensions', False)
    # Obtaining the member 'append' of a type (line 240)
    append_451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 240, 12), extensions_450, 'append')
    # Calling append(args, kwargs) (line 240)
    append_call_result_454 = invoke(stypy.reporting.localization.Localization(__file__, 240, 12), append_451, *[ext_452], **kwargs_453)
    
    # SSA join for while statement (line 156)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # finally branch of the try-finally block (line 153)
    
    # Call to close(...): (line 242)
    # Processing the call keyword arguments (line 242)
    kwargs_457 = {}
    # Getting the type of 'file' (line 242)
    file_455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 242, 8), 'file', False)
    # Obtaining the member 'close' of a type (line 242)
    close_456 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 242, 8), file_455, 'close')
    # Calling close(args, kwargs) (line 242)
    close_call_result_458 = invoke(stypy.reporting.localization.Localization(__file__, 242, 8), close_456, *[], **kwargs_457)
    
    
    # Getting the type of 'extensions' (line 253)
    extensions_459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 253, 11), 'extensions')
    # Assigning a type to the variable 'stypy_return_type' (line 253)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 253, 4), 'stypy_return_type', extensions_459)
    
    # ################# End of 'read_setup_file(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'read_setup_file' in the type store
    # Getting the type of 'stypy_return_type' (line 139)
    stypy_return_type_460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_460)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'read_setup_file'
    return stypy_return_type_460

# Assigning a type to the variable 'read_setup_file' (line 139)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 139, 0), 'read_setup_file', read_setup_file)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
