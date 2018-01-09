
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.build_clib
2: 
3: Implements the Distutils 'build_clib' command, to build a C/C++ library
4: that is included in the module distribution and needed by an extension
5: module.'''
6: 
7: __revision__ = "$Id$"
8: 
9: 
10: # XXX this module has *lots* of code ripped-off quite transparently from
11: # build_ext.py -- not surprisingly really, as the work required to build
12: # a static library from a collection of C source files is not really all
13: # that different from what's required to build a shared object file from
14: # a collection of C source files.  Nevertheless, I haven't done the
15: # necessary refactoring to account for the overlap in code between the
16: # two modules, mainly because a number of subtle details changed in the
17: # cut 'n paste.  Sigh.
18: 
19: import os
20: from distutils.core import Command
21: from distutils.errors import DistutilsSetupError
22: from distutils.sysconfig import customize_compiler
23: from distutils import log
24: 
25: def show_compilers():
26:     from distutils.ccompiler import show_compilers
27:     show_compilers()
28: 
29: 
30: class build_clib(Command):
31: 
32:     description = "build C/C++ libraries used by Python extensions"
33: 
34:     user_options = [
35:         ('build-clib=', 'b',
36:          "directory to build C/C++ libraries to"),
37:         ('build-temp=', 't',
38:          "directory to put temporary build by-products"),
39:         ('debug', 'g',
40:          "compile with debugging information"),
41:         ('force', 'f',
42:          "forcibly build everything (ignore file timestamps)"),
43:         ('compiler=', 'c',
44:          "specify the compiler type"),
45:         ]
46: 
47:     boolean_options = ['debug', 'force']
48: 
49:     help_options = [
50:         ('help-compiler', None,
51:          "list available compilers", show_compilers),
52:         ]
53: 
54:     def initialize_options(self):
55:         self.build_clib = None
56:         self.build_temp = None
57: 
58:         # List of libraries to build
59:         self.libraries = None
60: 
61:         # Compilation options for all libraries
62:         self.include_dirs = None
63:         self.define = None
64:         self.undef = None
65:         self.debug = None
66:         self.force = 0
67:         self.compiler = None
68: 
69: 
70:     def finalize_options(self):
71:         # This might be confusing: both build-clib and build-temp default
72:         # to build-temp as defined by the "build" command.  This is because
73:         # I think that C libraries are really just temporary build
74:         # by-products, at least from the point of view of building Python
75:         # extensions -- but I want to keep my options open.
76:         self.set_undefined_options('build',
77:                                    ('build_temp', 'build_clib'),
78:                                    ('build_temp', 'build_temp'),
79:                                    ('compiler', 'compiler'),
80:                                    ('debug', 'debug'),
81:                                    ('force', 'force'))
82: 
83:         self.libraries = self.distribution.libraries
84:         if self.libraries:
85:             self.check_library_list(self.libraries)
86: 
87:         if self.include_dirs is None:
88:             self.include_dirs = self.distribution.include_dirs or []
89:         if isinstance(self.include_dirs, str):
90:             self.include_dirs = self.include_dirs.split(os.pathsep)
91: 
92:         # XXX same as for build_ext -- what about 'self.define' and
93:         # 'self.undef' ?
94: 
95:     def run(self):
96:         if not self.libraries:
97:             return
98: 
99:         # Yech -- this is cut 'n pasted from build_ext.py!
100:         from distutils.ccompiler import new_compiler
101:         self.compiler = new_compiler(compiler=self.compiler,
102:                                      dry_run=self.dry_run,
103:                                      force=self.force)
104:         customize_compiler(self.compiler)
105: 
106:         if self.include_dirs is not None:
107:             self.compiler.set_include_dirs(self.include_dirs)
108:         if self.define is not None:
109:             # 'define' option is a list of (name,value) tuples
110:             for (name,value) in self.define:
111:                 self.compiler.define_macro(name, value)
112:         if self.undef is not None:
113:             for macro in self.undef:
114:                 self.compiler.undefine_macro(macro)
115: 
116:         self.build_libraries(self.libraries)
117: 
118: 
119:     def check_library_list(self, libraries):
120:         '''Ensure that the list of libraries is valid.
121: 
122:         `library` is presumably provided as a command option 'libraries'.
123:         This method checks that it is a list of 2-tuples, where the tuples
124:         are (library_name, build_info_dict).
125: 
126:         Raise DistutilsSetupError if the structure is invalid anywhere;
127:         just returns otherwise.
128:         '''
129:         if not isinstance(libraries, list):
130:             raise DistutilsSetupError, \
131:                   "'libraries' option must be a list of tuples"
132: 
133:         for lib in libraries:
134:             if not isinstance(lib, tuple) and len(lib) != 2:
135:                 raise DistutilsSetupError, \
136:                       "each element of 'libraries' must a 2-tuple"
137: 
138:             name, build_info = lib
139: 
140:             if not isinstance(name, str):
141:                 raise DistutilsSetupError, \
142:                       "first element of each tuple in 'libraries' " + \
143:                       "must be a string (the library name)"
144:             if '/' in name or (os.sep != '/' and os.sep in name):
145:                 raise DistutilsSetupError, \
146:                       ("bad library name '%s': " +
147:                        "may not contain directory separators") % \
148:                       lib[0]
149: 
150:             if not isinstance(build_info, dict):
151:                 raise DistutilsSetupError, \
152:                       "second element of each tuple in 'libraries' " + \
153:                       "must be a dictionary (build info)"
154: 
155:     def get_library_names(self):
156:         # Assume the library list is valid -- 'check_library_list()' is
157:         # called from 'finalize_options()', so it should be!
158:         if not self.libraries:
159:             return None
160: 
161:         lib_names = []
162:         for (lib_name, build_info) in self.libraries:
163:             lib_names.append(lib_name)
164:         return lib_names
165: 
166: 
167:     def get_source_files(self):
168:         self.check_library_list(self.libraries)
169:         filenames = []
170:         for (lib_name, build_info) in self.libraries:
171:             sources = build_info.get('sources')
172:             if sources is None or not isinstance(sources, (list, tuple)):
173:                 raise DistutilsSetupError, \
174:                       ("in 'libraries' option (library '%s'), "
175:                        "'sources' must be present and must be "
176:                        "a list of source filenames") % lib_name
177: 
178:             filenames.extend(sources)
179:         return filenames
180: 
181:     def build_libraries(self, libraries):
182:         for (lib_name, build_info) in libraries:
183:             sources = build_info.get('sources')
184:             if sources is None or not isinstance(sources, (list, tuple)):
185:                 raise DistutilsSetupError, \
186:                       ("in 'libraries' option (library '%s'), " +
187:                        "'sources' must be present and must be " +
188:                        "a list of source filenames") % lib_name
189:             sources = list(sources)
190: 
191:             log.info("building '%s' library", lib_name)
192: 
193:             # First, compile the source code to object files in the library
194:             # directory.  (This should probably change to putting object
195:             # files in a temporary build directory.)
196:             macros = build_info.get('macros')
197:             include_dirs = build_info.get('include_dirs')
198:             objects = self.compiler.compile(sources,
199:                                             output_dir=self.build_temp,
200:                                             macros=macros,
201:                                             include_dirs=include_dirs,
202:                                             debug=self.debug)
203: 
204:             # Now "link" the object files together into a static library.
205:             # (On Unix at least, this isn't really linking -- it just
206:             # builds an archive.  Whatever.)
207:             self.compiler.create_static_lib(objects, lib_name,
208:                                             output_dir=self.build_clib,
209:                                             debug=self.debug)
210: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_17475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, (-1)), 'str', "distutils.command.build_clib\n\nImplements the Distutils 'build_clib' command, to build a C/C++ library\nthat is included in the module distribution and needed by an extension\nmodule.")

# Assigning a Str to a Name (line 7):

# Assigning a Str to a Name (line 7):
str_17476 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_17476)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 19, 0))

# 'import os' statement (line 19)
import os

import_module(stypy.reporting.localization.Localization(__file__, 19, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 20, 0))

# 'from distutils.core import Command' statement (line 20)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_17477 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.core')

if (type(import_17477) is not StypyTypeError):

    if (import_17477 != 'pyd_module'):
        __import__(import_17477)
        sys_modules_17478 = sys.modules[import_17477]
        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.core', sys_modules_17478.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 20, 0), __file__, sys_modules_17478, sys_modules_17478.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'distutils.core', import_17477)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 21, 0))

# 'from distutils.errors import DistutilsSetupError' statement (line 21)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_17479 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.errors')

if (type(import_17479) is not StypyTypeError):

    if (import_17479 != 'pyd_module'):
        __import__(import_17479)
        sys_modules_17480 = sys.modules[import_17479]
        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.errors', sys_modules_17480.module_type_store, module_type_store, ['DistutilsSetupError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 21, 0), __file__, sys_modules_17480, sys_modules_17480.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsSetupError

        import_from_module(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.errors', None, module_type_store, ['DistutilsSetupError'], [DistutilsSetupError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 21)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'distutils.errors', import_17479)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 22, 0))

# 'from distutils.sysconfig import customize_compiler' statement (line 22)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_17481 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.sysconfig')

if (type(import_17481) is not StypyTypeError):

    if (import_17481 != 'pyd_module'):
        __import__(import_17481)
        sys_modules_17482 = sys.modules[import_17481]
        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.sysconfig', sys_modules_17482.module_type_store, module_type_store, ['customize_compiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 22, 0), __file__, sys_modules_17482, sys_modules_17482.module_type_store, module_type_store)
    else:
        from distutils.sysconfig import customize_compiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.sysconfig', None, module_type_store, ['customize_compiler'], [customize_compiler])

else:
    # Assigning a type to the variable 'distutils.sysconfig' (line 22)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 0), 'distutils.sysconfig', import_17481)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 23, 0))

# 'from distutils import log' statement (line 23)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 23, 0), 'distutils', None, module_type_store, ['log'], [log])


@norecursion
def show_compilers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'show_compilers'
    module_type_store = module_type_store.open_function_context('show_compilers', 25, 0, False)
    
    # Passed parameters checking function
    show_compilers.stypy_localization = localization
    show_compilers.stypy_type_of_self = None
    show_compilers.stypy_type_store = module_type_store
    show_compilers.stypy_function_name = 'show_compilers'
    show_compilers.stypy_param_names_list = []
    show_compilers.stypy_varargs_param_name = None
    show_compilers.stypy_kwargs_param_name = None
    show_compilers.stypy_call_defaults = defaults
    show_compilers.stypy_call_varargs = varargs
    show_compilers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'show_compilers', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'show_compilers', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'show_compilers(...)' code ##################

    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 26, 4))
    
    # 'from distutils.ccompiler import show_compilers' statement (line 26)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
    import_17483 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 26, 4), 'distutils.ccompiler')

    if (type(import_17483) is not StypyTypeError):

        if (import_17483 != 'pyd_module'):
            __import__(import_17483)
            sys_modules_17484 = sys.modules[import_17483]
            import_from_module(stypy.reporting.localization.Localization(__file__, 26, 4), 'distutils.ccompiler', sys_modules_17484.module_type_store, module_type_store, ['show_compilers'])
            nest_module(stypy.reporting.localization.Localization(__file__, 26, 4), __file__, sys_modules_17484, sys_modules_17484.module_type_store, module_type_store)
        else:
            from distutils.ccompiler import show_compilers

            import_from_module(stypy.reporting.localization.Localization(__file__, 26, 4), 'distutils.ccompiler', None, module_type_store, ['show_compilers'], [show_compilers])

    else:
        # Assigning a type to the variable 'distutils.ccompiler' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'distutils.ccompiler', import_17483)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
    
    
    # Call to show_compilers(...): (line 27)
    # Processing the call keyword arguments (line 27)
    kwargs_17486 = {}
    # Getting the type of 'show_compilers' (line 27)
    show_compilers_17485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'show_compilers', False)
    # Calling show_compilers(args, kwargs) (line 27)
    show_compilers_call_result_17487 = invoke(stypy.reporting.localization.Localization(__file__, 27, 4), show_compilers_17485, *[], **kwargs_17486)
    
    
    # ################# End of 'show_compilers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show_compilers' in the type store
    # Getting the type of 'stypy_return_type' (line 25)
    stypy_return_type_17488 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_17488)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show_compilers'
    return stypy_return_type_17488

# Assigning a type to the variable 'show_compilers' (line 25)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 0), 'show_compilers', show_compilers)
# Declaration of the 'build_clib' class
# Getting the type of 'Command' (line 30)
Command_17489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 17), 'Command')

class build_clib(Command_17489, ):
    
    # Assigning a Str to a Name (line 32):
    
    # Assigning a List to a Name (line 34):
    
    # Assigning a List to a Name (line 47):
    
    # Assigning a List to a Name (line 49):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 54, 4, False)
        # Assigning a type to the variable 'self' (line 55)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build_clib.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.initialize_options.__dict__.__setitem__('stypy_function_name', 'build_clib.initialize_options')
        build_clib.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.initialize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize_options(...)' code ##################

        
        # Assigning a Name to a Attribute (line 55):
        
        # Assigning a Name to a Attribute (line 55):
        # Getting the type of 'None' (line 55)
        None_17490 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 26), 'None')
        # Getting the type of 'self' (line 55)
        self_17491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'build_clib' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_17491, 'build_clib', None_17490)
        
        # Assigning a Name to a Attribute (line 56):
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'None' (line 56)
        None_17492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 26), 'None')
        # Getting the type of 'self' (line 56)
        self_17493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'build_temp' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_17493, 'build_temp', None_17492)
        
        # Assigning a Name to a Attribute (line 59):
        
        # Assigning a Name to a Attribute (line 59):
        # Getting the type of 'None' (line 59)
        None_17494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'None')
        # Getting the type of 'self' (line 59)
        self_17495 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 59)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), self_17495, 'libraries', None_17494)
        
        # Assigning a Name to a Attribute (line 62):
        
        # Assigning a Name to a Attribute (line 62):
        # Getting the type of 'None' (line 62)
        None_17496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 28), 'None')
        # Getting the type of 'self' (line 62)
        self_17497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'self')
        # Setting the type of the member 'include_dirs' of a type (line 62)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 8), self_17497, 'include_dirs', None_17496)
        
        # Assigning a Name to a Attribute (line 63):
        
        # Assigning a Name to a Attribute (line 63):
        # Getting the type of 'None' (line 63)
        None_17498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 22), 'None')
        # Getting the type of 'self' (line 63)
        self_17499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member 'define' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_17499, 'define', None_17498)
        
        # Assigning a Name to a Attribute (line 64):
        
        # Assigning a Name to a Attribute (line 64):
        # Getting the type of 'None' (line 64)
        None_17500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 21), 'None')
        # Getting the type of 'self' (line 64)
        self_17501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'self')
        # Setting the type of the member 'undef' of a type (line 64)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 8), self_17501, 'undef', None_17500)
        
        # Assigning a Name to a Attribute (line 65):
        
        # Assigning a Name to a Attribute (line 65):
        # Getting the type of 'None' (line 65)
        None_17502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 21), 'None')
        # Getting the type of 'self' (line 65)
        self_17503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'self')
        # Setting the type of the member 'debug' of a type (line 65)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 8), self_17503, 'debug', None_17502)
        
        # Assigning a Num to a Attribute (line 66):
        
        # Assigning a Num to a Attribute (line 66):
        int_17504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 21), 'int')
        # Getting the type of 'self' (line 66)
        self_17505 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 8), 'self')
        # Setting the type of the member 'force' of a type (line 66)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 8), self_17505, 'force', int_17504)
        
        # Assigning a Name to a Attribute (line 67):
        
        # Assigning a Name to a Attribute (line 67):
        # Getting the type of 'None' (line 67)
        None_17506 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 24), 'None')
        # Getting the type of 'self' (line 67)
        self_17507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 67)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), self_17507, 'compiler', None_17506)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 54)
        stypy_return_type_17508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17508)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_17508


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build_clib.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.finalize_options.__dict__.__setitem__('stypy_function_name', 'build_clib.finalize_options')
        build_clib.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.finalize_options', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'finalize_options', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'finalize_options(...)' code ##################

        
        # Call to set_undefined_options(...): (line 76)
        # Processing the call arguments (line 76)
        str_17511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 77)
        tuple_17512 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 77)
        # Adding element type (line 77)
        str_17513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 36), 'str', 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 36), tuple_17512, str_17513)
        # Adding element type (line 77)
        str_17514 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 50), 'str', 'build_clib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 36), tuple_17512, str_17514)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 78)
        tuple_17515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 78)
        # Adding element type (line 78)
        str_17516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 36), 'str', 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 36), tuple_17515, str_17516)
        # Adding element type (line 78)
        str_17517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 50), 'str', 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 78, 36), tuple_17515, str_17517)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 79)
        tuple_17518 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 79)
        # Adding element type (line 79)
        str_17519 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 36), 'str', 'compiler')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 36), tuple_17518, str_17519)
        # Adding element type (line 79)
        str_17520 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 48), 'str', 'compiler')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 36), tuple_17518, str_17520)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 80)
        tuple_17521 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 80)
        # Adding element type (line 80)
        str_17522 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 36), 'str', 'debug')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 36), tuple_17521, str_17522)
        # Adding element type (line 80)
        str_17523 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 45), 'str', 'debug')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 36), tuple_17521, str_17523)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 81)
        tuple_17524 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 81)
        # Adding element type (line 81)
        str_17525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 36), tuple_17524, str_17525)
        # Adding element type (line 81)
        str_17526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 81, 36), tuple_17524, str_17526)
        
        # Processing the call keyword arguments (line 76)
        kwargs_17527 = {}
        # Getting the type of 'self' (line 76)
        self_17509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 76)
        set_undefined_options_17510 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 8), self_17509, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 76)
        set_undefined_options_call_result_17528 = invoke(stypy.reporting.localization.Localization(__file__, 76, 8), set_undefined_options_17510, *[str_17511, tuple_17512, tuple_17515, tuple_17518, tuple_17521, tuple_17524], **kwargs_17527)
        
        
        # Assigning a Attribute to a Attribute (line 83):
        
        # Assigning a Attribute to a Attribute (line 83):
        # Getting the type of 'self' (line 83)
        self_17529 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 25), 'self')
        # Obtaining the member 'distribution' of a type (line 83)
        distribution_17530 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 25), self_17529, 'distribution')
        # Obtaining the member 'libraries' of a type (line 83)
        libraries_17531 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 25), distribution_17530, 'libraries')
        # Getting the type of 'self' (line 83)
        self_17532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'self')
        # Setting the type of the member 'libraries' of a type (line 83)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 8), self_17532, 'libraries', libraries_17531)
        
        # Getting the type of 'self' (line 84)
        self_17533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 11), 'self')
        # Obtaining the member 'libraries' of a type (line 84)
        libraries_17534 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 84, 11), self_17533, 'libraries')
        # Testing the type of an if condition (line 84)
        if_condition_17535 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 84, 8), libraries_17534)
        # Assigning a type to the variable 'if_condition_17535' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'if_condition_17535', if_condition_17535)
        # SSA begins for if statement (line 84)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_library_list(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'self' (line 85)
        self_17538 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 36), 'self', False)
        # Obtaining the member 'libraries' of a type (line 85)
        libraries_17539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 36), self_17538, 'libraries')
        # Processing the call keyword arguments (line 85)
        kwargs_17540 = {}
        # Getting the type of 'self' (line 85)
        self_17536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'self', False)
        # Obtaining the member 'check_library_list' of a type (line 85)
        check_library_list_17537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 12), self_17536, 'check_library_list')
        # Calling check_library_list(args, kwargs) (line 85)
        check_library_list_call_result_17541 = invoke(stypy.reporting.localization.Localization(__file__, 85, 12), check_library_list_17537, *[libraries_17539], **kwargs_17540)
        
        # SSA join for if statement (line 84)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 87)
        # Getting the type of 'self' (line 87)
        self_17542 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 11), 'self')
        # Obtaining the member 'include_dirs' of a type (line 87)
        include_dirs_17543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 11), self_17542, 'include_dirs')
        # Getting the type of 'None' (line 87)
        None_17544 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 32), 'None')
        
        (may_be_17545, more_types_in_union_17546) = may_be_none(include_dirs_17543, None_17544)

        if may_be_17545:

            if more_types_in_union_17546:
                # Runtime conditional SSA (line 87)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BoolOp to a Attribute (line 88):
            
            # Assigning a BoolOp to a Attribute (line 88):
            
            # Evaluating a boolean operation
            # Getting the type of 'self' (line 88)
            self_17547 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 32), 'self')
            # Obtaining the member 'distribution' of a type (line 88)
            distribution_17548 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 32), self_17547, 'distribution')
            # Obtaining the member 'include_dirs' of a type (line 88)
            include_dirs_17549 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 32), distribution_17548, 'include_dirs')
            
            # Obtaining an instance of the builtin type 'list' (line 88)
            list_17550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 66), 'list')
            # Adding type elements to the builtin type 'list' instance (line 88)
            
            # Applying the binary operator 'or' (line 88)
            result_or_keyword_17551 = python_operator(stypy.reporting.localization.Localization(__file__, 88, 32), 'or', include_dirs_17549, list_17550)
            
            # Getting the type of 'self' (line 88)
            self_17552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 12), 'self')
            # Setting the type of the member 'include_dirs' of a type (line 88)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 12), self_17552, 'include_dirs', result_or_keyword_17551)

            if more_types_in_union_17546:
                # SSA join for if statement (line 87)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 89)
        # Getting the type of 'str' (line 89)
        str_17553 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 41), 'str')
        # Getting the type of 'self' (line 89)
        self_17554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 22), 'self')
        # Obtaining the member 'include_dirs' of a type (line 89)
        include_dirs_17555 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 22), self_17554, 'include_dirs')
        
        (may_be_17556, more_types_in_union_17557) = may_be_subtype(str_17553, include_dirs_17555)

        if may_be_17556:

            if more_types_in_union_17557:
                # Runtime conditional SSA (line 89)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Getting the type of 'self' (line 89)
            self_17558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
            # Obtaining the member 'include_dirs' of a type (line 89)
            include_dirs_17559 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_17558, 'include_dirs')
            # Setting the type of the member 'include_dirs' of a type (line 89)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_17558, 'include_dirs', remove_not_subtype_from_union(include_dirs_17555, str))
            
            # Assigning a Call to a Attribute (line 90):
            
            # Assigning a Call to a Attribute (line 90):
            
            # Call to split(...): (line 90)
            # Processing the call arguments (line 90)
            # Getting the type of 'os' (line 90)
            os_17563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 56), 'os', False)
            # Obtaining the member 'pathsep' of a type (line 90)
            pathsep_17564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 56), os_17563, 'pathsep')
            # Processing the call keyword arguments (line 90)
            kwargs_17565 = {}
            # Getting the type of 'self' (line 90)
            self_17560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 32), 'self', False)
            # Obtaining the member 'include_dirs' of a type (line 90)
            include_dirs_17561 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 32), self_17560, 'include_dirs')
            # Obtaining the member 'split' of a type (line 90)
            split_17562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 32), include_dirs_17561, 'split')
            # Calling split(args, kwargs) (line 90)
            split_call_result_17566 = invoke(stypy.reporting.localization.Localization(__file__, 90, 32), split_17562, *[pathsep_17564], **kwargs_17565)
            
            # Getting the type of 'self' (line 90)
            self_17567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 12), 'self')
            # Setting the type of the member 'include_dirs' of a type (line 90)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 12), self_17567, 'include_dirs', split_call_result_17566)

            if more_types_in_union_17557:
                # SSA join for if statement (line 89)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_17568 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17568)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_17568


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 95, 4, False)
        # Assigning a type to the variable 'self' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.run.__dict__.__setitem__('stypy_localization', localization)
        build_clib.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.run.__dict__.__setitem__('stypy_function_name', 'build_clib.run')
        build_clib.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 96)
        self_17569 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'self')
        # Obtaining the member 'libraries' of a type (line 96)
        libraries_17570 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), self_17569, 'libraries')
        # Applying the 'not' unary operator (line 96)
        result_not__17571 = python_operator(stypy.reporting.localization.Localization(__file__, 96, 11), 'not', libraries_17570)
        
        # Testing the type of an if condition (line 96)
        if_condition_17572 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 8), result_not__17571)
        # Assigning a type to the variable 'if_condition_17572' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'if_condition_17572', if_condition_17572)
        # SSA begins for if statement (line 96)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 96)
        module_type_store = module_type_store.join_ssa_context()
        
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 100, 8))
        
        # 'from distutils.ccompiler import new_compiler' statement (line 100)
        update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
        import_17573 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 100, 8), 'distutils.ccompiler')

        if (type(import_17573) is not StypyTypeError):

            if (import_17573 != 'pyd_module'):
                __import__(import_17573)
                sys_modules_17574 = sys.modules[import_17573]
                import_from_module(stypy.reporting.localization.Localization(__file__, 100, 8), 'distutils.ccompiler', sys_modules_17574.module_type_store, module_type_store, ['new_compiler'])
                nest_module(stypy.reporting.localization.Localization(__file__, 100, 8), __file__, sys_modules_17574, sys_modules_17574.module_type_store, module_type_store)
            else:
                from distutils.ccompiler import new_compiler

                import_from_module(stypy.reporting.localization.Localization(__file__, 100, 8), 'distutils.ccompiler', None, module_type_store, ['new_compiler'], [new_compiler])

        else:
            # Assigning a type to the variable 'distutils.ccompiler' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'distutils.ccompiler', import_17573)

        remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
        
        
        # Assigning a Call to a Attribute (line 101):
        
        # Assigning a Call to a Attribute (line 101):
        
        # Call to new_compiler(...): (line 101)
        # Processing the call keyword arguments (line 101)
        # Getting the type of 'self' (line 101)
        self_17576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 46), 'self', False)
        # Obtaining the member 'compiler' of a type (line 101)
        compiler_17577 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 46), self_17576, 'compiler')
        keyword_17578 = compiler_17577
        # Getting the type of 'self' (line 102)
        self_17579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 45), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 102)
        dry_run_17580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 45), self_17579, 'dry_run')
        keyword_17581 = dry_run_17580
        # Getting the type of 'self' (line 103)
        self_17582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 43), 'self', False)
        # Obtaining the member 'force' of a type (line 103)
        force_17583 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 43), self_17582, 'force')
        keyword_17584 = force_17583
        kwargs_17585 = {'force': keyword_17584, 'dry_run': keyword_17581, 'compiler': keyword_17578}
        # Getting the type of 'new_compiler' (line 101)
        new_compiler_17575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'new_compiler', False)
        # Calling new_compiler(args, kwargs) (line 101)
        new_compiler_call_result_17586 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), new_compiler_17575, *[], **kwargs_17585)
        
        # Getting the type of 'self' (line 101)
        self_17587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 101)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 8), self_17587, 'compiler', new_compiler_call_result_17586)
        
        # Call to customize_compiler(...): (line 104)
        # Processing the call arguments (line 104)
        # Getting the type of 'self' (line 104)
        self_17589 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 27), 'self', False)
        # Obtaining the member 'compiler' of a type (line 104)
        compiler_17590 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 27), self_17589, 'compiler')
        # Processing the call keyword arguments (line 104)
        kwargs_17591 = {}
        # Getting the type of 'customize_compiler' (line 104)
        customize_compiler_17588 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'customize_compiler', False)
        # Calling customize_compiler(args, kwargs) (line 104)
        customize_compiler_call_result_17592 = invoke(stypy.reporting.localization.Localization(__file__, 104, 8), customize_compiler_17588, *[compiler_17590], **kwargs_17591)
        
        
        
        # Getting the type of 'self' (line 106)
        self_17593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'self')
        # Obtaining the member 'include_dirs' of a type (line 106)
        include_dirs_17594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 11), self_17593, 'include_dirs')
        # Getting the type of 'None' (line 106)
        None_17595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 36), 'None')
        # Applying the binary operator 'isnot' (line 106)
        result_is_not_17596 = python_operator(stypy.reporting.localization.Localization(__file__, 106, 11), 'isnot', include_dirs_17594, None_17595)
        
        # Testing the type of an if condition (line 106)
        if_condition_17597 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), result_is_not_17596)
        # Assigning a type to the variable 'if_condition_17597' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_17597', if_condition_17597)
        # SSA begins for if statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to set_include_dirs(...): (line 107)
        # Processing the call arguments (line 107)
        # Getting the type of 'self' (line 107)
        self_17601 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 43), 'self', False)
        # Obtaining the member 'include_dirs' of a type (line 107)
        include_dirs_17602 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 43), self_17601, 'include_dirs')
        # Processing the call keyword arguments (line 107)
        kwargs_17603 = {}
        # Getting the type of 'self' (line 107)
        self_17598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'self', False)
        # Obtaining the member 'compiler' of a type (line 107)
        compiler_17599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), self_17598, 'compiler')
        # Obtaining the member 'set_include_dirs' of a type (line 107)
        set_include_dirs_17600 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 12), compiler_17599, 'set_include_dirs')
        # Calling set_include_dirs(args, kwargs) (line 107)
        set_include_dirs_call_result_17604 = invoke(stypy.reporting.localization.Localization(__file__, 107, 12), set_include_dirs_17600, *[include_dirs_17602], **kwargs_17603)
        
        # SSA join for if statement (line 106)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 108)
        self_17605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 11), 'self')
        # Obtaining the member 'define' of a type (line 108)
        define_17606 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 11), self_17605, 'define')
        # Getting the type of 'None' (line 108)
        None_17607 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 30), 'None')
        # Applying the binary operator 'isnot' (line 108)
        result_is_not_17608 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 11), 'isnot', define_17606, None_17607)
        
        # Testing the type of an if condition (line 108)
        if_condition_17609 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 108, 8), result_is_not_17608)
        # Assigning a type to the variable 'if_condition_17609' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 8), 'if_condition_17609', if_condition_17609)
        # SSA begins for if statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 110)
        self_17610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'self')
        # Obtaining the member 'define' of a type (line 110)
        define_17611 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 32), self_17610, 'define')
        # Testing the type of a for loop iterable (line 110)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 110, 12), define_17611)
        # Getting the type of the for loop variable (line 110)
        for_loop_var_17612 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 110, 12), define_17611)
        # Assigning a type to the variable 'name' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 12), for_loop_var_17612))
        # Assigning a type to the variable 'value' (line 110)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 12), 'value', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 110, 12), for_loop_var_17612))
        # SSA begins for a for statement (line 110)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to define_macro(...): (line 111)
        # Processing the call arguments (line 111)
        # Getting the type of 'name' (line 111)
        name_17616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 43), 'name', False)
        # Getting the type of 'value' (line 111)
        value_17617 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 49), 'value', False)
        # Processing the call keyword arguments (line 111)
        kwargs_17618 = {}
        # Getting the type of 'self' (line 111)
        self_17613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 16), 'self', False)
        # Obtaining the member 'compiler' of a type (line 111)
        compiler_17614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), self_17613, 'compiler')
        # Obtaining the member 'define_macro' of a type (line 111)
        define_macro_17615 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 16), compiler_17614, 'define_macro')
        # Calling define_macro(args, kwargs) (line 111)
        define_macro_call_result_17619 = invoke(stypy.reporting.localization.Localization(__file__, 111, 16), define_macro_17615, *[name_17616, value_17617], **kwargs_17618)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 108)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 112)
        self_17620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 11), 'self')
        # Obtaining the member 'undef' of a type (line 112)
        undef_17621 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 11), self_17620, 'undef')
        # Getting the type of 'None' (line 112)
        None_17622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 29), 'None')
        # Applying the binary operator 'isnot' (line 112)
        result_is_not_17623 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 11), 'isnot', undef_17621, None_17622)
        
        # Testing the type of an if condition (line 112)
        if_condition_17624 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 8), result_is_not_17623)
        # Assigning a type to the variable 'if_condition_17624' (line 112)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 8), 'if_condition_17624', if_condition_17624)
        # SSA begins for if statement (line 112)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 113)
        self_17625 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 25), 'self')
        # Obtaining the member 'undef' of a type (line 113)
        undef_17626 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 113, 25), self_17625, 'undef')
        # Testing the type of a for loop iterable (line 113)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 113, 12), undef_17626)
        # Getting the type of the for loop variable (line 113)
        for_loop_var_17627 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 113, 12), undef_17626)
        # Assigning a type to the variable 'macro' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'macro', for_loop_var_17627)
        # SSA begins for a for statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to undefine_macro(...): (line 114)
        # Processing the call arguments (line 114)
        # Getting the type of 'macro' (line 114)
        macro_17631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 45), 'macro', False)
        # Processing the call keyword arguments (line 114)
        kwargs_17632 = {}
        # Getting the type of 'self' (line 114)
        self_17628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 16), 'self', False)
        # Obtaining the member 'compiler' of a type (line 114)
        compiler_17629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), self_17628, 'compiler')
        # Obtaining the member 'undefine_macro' of a type (line 114)
        undefine_macro_17630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 16), compiler_17629, 'undefine_macro')
        # Calling undefine_macro(args, kwargs) (line 114)
        undefine_macro_call_result_17633 = invoke(stypy.reporting.localization.Localization(__file__, 114, 16), undefine_macro_17630, *[macro_17631], **kwargs_17632)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 112)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to build_libraries(...): (line 116)
        # Processing the call arguments (line 116)
        # Getting the type of 'self' (line 116)
        self_17636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 29), 'self', False)
        # Obtaining the member 'libraries' of a type (line 116)
        libraries_17637 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 29), self_17636, 'libraries')
        # Processing the call keyword arguments (line 116)
        kwargs_17638 = {}
        # Getting the type of 'self' (line 116)
        self_17634 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 8), 'self', False)
        # Obtaining the member 'build_libraries' of a type (line 116)
        build_libraries_17635 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 8), self_17634, 'build_libraries')
        # Calling build_libraries(args, kwargs) (line 116)
        build_libraries_call_result_17639 = invoke(stypy.reporting.localization.Localization(__file__, 116, 8), build_libraries_17635, *[libraries_17637], **kwargs_17638)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 95)
        stypy_return_type_17640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17640)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_17640


    @norecursion
    def check_library_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_library_list'
        module_type_store = module_type_store.open_function_context('check_library_list', 119, 4, False)
        # Assigning a type to the variable 'self' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.check_library_list.__dict__.__setitem__('stypy_localization', localization)
        build_clib.check_library_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.check_library_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.check_library_list.__dict__.__setitem__('stypy_function_name', 'build_clib.check_library_list')
        build_clib.check_library_list.__dict__.__setitem__('stypy_param_names_list', ['libraries'])
        build_clib.check_library_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.check_library_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.check_library_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.check_library_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.check_library_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.check_library_list.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.check_library_list', ['libraries'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_library_list', localization, ['libraries'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_library_list(...)' code ##################

        str_17641 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 128, (-1)), 'str', "Ensure that the list of libraries is valid.\n\n        `library` is presumably provided as a command option 'libraries'.\n        This method checks that it is a list of 2-tuples, where the tuples\n        are (library_name, build_info_dict).\n\n        Raise DistutilsSetupError if the structure is invalid anywhere;\n        just returns otherwise.\n        ")
        
        # Type idiom detected: calculating its left and rigth part (line 129)
        # Getting the type of 'list' (line 129)
        list_17642 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 37), 'list')
        # Getting the type of 'libraries' (line 129)
        libraries_17643 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 26), 'libraries')
        
        (may_be_17644, more_types_in_union_17645) = may_not_be_subtype(list_17642, libraries_17643)

        if may_be_17644:

            if more_types_in_union_17645:
                # Runtime conditional SSA (line 129)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'libraries' (line 129)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'libraries', remove_subtype_from_union(libraries_17643, list))
            # Getting the type of 'DistutilsSetupError' (line 130)
            DistutilsSetupError_17646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 18), 'DistutilsSetupError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 130, 12), DistutilsSetupError_17646, 'raise parameter', BaseException)

            if more_types_in_union_17645:
                # SSA join for if statement (line 129)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Getting the type of 'libraries' (line 133)
        libraries_17647 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'libraries')
        # Testing the type of a for loop iterable (line 133)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 133, 8), libraries_17647)
        # Getting the type of the for loop variable (line 133)
        for_loop_var_17648 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 133, 8), libraries_17647)
        # Assigning a type to the variable 'lib' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'lib', for_loop_var_17648)
        # SSA begins for a for statement (line 133)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Evaluating a boolean operation
        
        
        # Call to isinstance(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'lib' (line 134)
        lib_17650 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 30), 'lib', False)
        # Getting the type of 'tuple' (line 134)
        tuple_17651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 35), 'tuple', False)
        # Processing the call keyword arguments (line 134)
        kwargs_17652 = {}
        # Getting the type of 'isinstance' (line 134)
        isinstance_17649 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 19), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 134)
        isinstance_call_result_17653 = invoke(stypy.reporting.localization.Localization(__file__, 134, 19), isinstance_17649, *[lib_17650, tuple_17651], **kwargs_17652)
        
        # Applying the 'not' unary operator (line 134)
        result_not__17654 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), 'not', isinstance_call_result_17653)
        
        
        
        # Call to len(...): (line 134)
        # Processing the call arguments (line 134)
        # Getting the type of 'lib' (line 134)
        lib_17656 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 50), 'lib', False)
        # Processing the call keyword arguments (line 134)
        kwargs_17657 = {}
        # Getting the type of 'len' (line 134)
        len_17655 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 46), 'len', False)
        # Calling len(args, kwargs) (line 134)
        len_call_result_17658 = invoke(stypy.reporting.localization.Localization(__file__, 134, 46), len_17655, *[lib_17656], **kwargs_17657)
        
        int_17659 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 134, 58), 'int')
        # Applying the binary operator '!=' (line 134)
        result_ne_17660 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 46), '!=', len_call_result_17658, int_17659)
        
        # Applying the binary operator 'and' (line 134)
        result_and_keyword_17661 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), 'and', result_not__17654, result_ne_17660)
        
        # Testing the type of an if condition (line 134)
        if_condition_17662 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 12), result_and_keyword_17661)
        # Assigning a type to the variable 'if_condition_17662' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'if_condition_17662', if_condition_17662)
        # SSA begins for if statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsSetupError' (line 135)
        DistutilsSetupError_17663 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 22), 'DistutilsSetupError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 135, 16), DistutilsSetupError_17663, 'raise parameter', BaseException)
        # SSA join for if statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Name to a Tuple (line 138):
        
        # Assigning a Subscript to a Name (line 138):
        
        # Obtaining the type of the subscript
        int_17664 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 12), 'int')
        # Getting the type of 'lib' (line 138)
        lib_17665 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'lib')
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___17666 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), lib_17665, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_17667 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), getitem___17666, int_17664)
        
        # Assigning a type to the variable 'tuple_var_assignment_17473' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_17473', subscript_call_result_17667)
        
        # Assigning a Subscript to a Name (line 138):
        
        # Obtaining the type of the subscript
        int_17668 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 12), 'int')
        # Getting the type of 'lib' (line 138)
        lib_17669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 31), 'lib')
        # Obtaining the member '__getitem__' of a type (line 138)
        getitem___17670 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 12), lib_17669, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 138)
        subscript_call_result_17671 = invoke(stypy.reporting.localization.Localization(__file__, 138, 12), getitem___17670, int_17668)
        
        # Assigning a type to the variable 'tuple_var_assignment_17474' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_17474', subscript_call_result_17671)
        
        # Assigning a Name to a Name (line 138):
        # Getting the type of 'tuple_var_assignment_17473' (line 138)
        tuple_var_assignment_17473_17672 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_17473')
        # Assigning a type to the variable 'name' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'name', tuple_var_assignment_17473_17672)
        
        # Assigning a Name to a Name (line 138):
        # Getting the type of 'tuple_var_assignment_17474' (line 138)
        tuple_var_assignment_17474_17673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'tuple_var_assignment_17474')
        # Assigning a type to the variable 'build_info' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 18), 'build_info', tuple_var_assignment_17474_17673)
        
        # Type idiom detected: calculating its left and rigth part (line 140)
        # Getting the type of 'str' (line 140)
        str_17674 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 36), 'str')
        # Getting the type of 'name' (line 140)
        name_17675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 30), 'name')
        
        (may_be_17676, more_types_in_union_17677) = may_not_be_subtype(str_17674, name_17675)

        if may_be_17676:

            if more_types_in_union_17677:
                # Runtime conditional SSA (line 140)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'name' (line 140)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 140, 12), 'name', remove_subtype_from_union(name_17675, str))
            # Getting the type of 'DistutilsSetupError' (line 141)
            DistutilsSetupError_17678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 22), 'DistutilsSetupError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 141, 16), DistutilsSetupError_17678, 'raise parameter', BaseException)

            if more_types_in_union_17677:
                # SSA join for if statement (line 140)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Evaluating a boolean operation
        
        str_17679 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 15), 'str', '/')
        # Getting the type of 'name' (line 144)
        name_17680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 22), 'name')
        # Applying the binary operator 'in' (line 144)
        result_contains_17681 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), 'in', str_17679, name_17680)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'os' (line 144)
        os_17682 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 31), 'os')
        # Obtaining the member 'sep' of a type (line 144)
        sep_17683 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 31), os_17682, 'sep')
        str_17684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 41), 'str', '/')
        # Applying the binary operator '!=' (line 144)
        result_ne_17685 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 31), '!=', sep_17683, str_17684)
        
        
        # Getting the type of 'os' (line 144)
        os_17686 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 49), 'os')
        # Obtaining the member 'sep' of a type (line 144)
        sep_17687 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 49), os_17686, 'sep')
        # Getting the type of 'name' (line 144)
        name_17688 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 59), 'name')
        # Applying the binary operator 'in' (line 144)
        result_contains_17689 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 49), 'in', sep_17687, name_17688)
        
        # Applying the binary operator 'and' (line 144)
        result_and_keyword_17690 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 31), 'and', result_ne_17685, result_contains_17689)
        
        # Applying the binary operator 'or' (line 144)
        result_or_keyword_17691 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), 'or', result_contains_17681, result_and_keyword_17690)
        
        # Testing the type of an if condition (line 144)
        if_condition_17692 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 12), result_or_keyword_17691)
        # Assigning a type to the variable 'if_condition_17692' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'if_condition_17692', if_condition_17692)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsSetupError' (line 145)
        DistutilsSetupError_17693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 22), 'DistutilsSetupError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 145, 16), DistutilsSetupError_17693, 'raise parameter', BaseException)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Type idiom detected: calculating its left and rigth part (line 150)
        # Getting the type of 'dict' (line 150)
        dict_17694 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 42), 'dict')
        # Getting the type of 'build_info' (line 150)
        build_info_17695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 150, 30), 'build_info')
        
        (may_be_17696, more_types_in_union_17697) = may_not_be_subtype(dict_17694, build_info_17695)

        if may_be_17696:

            if more_types_in_union_17697:
                # Runtime conditional SSA (line 150)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'build_info' (line 150)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 150, 12), 'build_info', remove_subtype_from_union(build_info_17695, dict))
            # Getting the type of 'DistutilsSetupError' (line 151)
            DistutilsSetupError_17698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 151, 22), 'DistutilsSetupError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 151, 16), DistutilsSetupError_17698, 'raise parameter', BaseException)

            if more_types_in_union_17697:
                # SSA join for if statement (line 150)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_library_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_library_list' in the type store
        # Getting the type of 'stypy_return_type' (line 119)
        stypy_return_type_17699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17699)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_library_list'
        return stypy_return_type_17699


    @norecursion
    def get_library_names(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_library_names'
        module_type_store = module_type_store.open_function_context('get_library_names', 155, 4, False)
        # Assigning a type to the variable 'self' (line 156)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 156, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.get_library_names.__dict__.__setitem__('stypy_localization', localization)
        build_clib.get_library_names.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.get_library_names.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.get_library_names.__dict__.__setitem__('stypy_function_name', 'build_clib.get_library_names')
        build_clib.get_library_names.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.get_library_names.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.get_library_names.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.get_library_names.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.get_library_names.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.get_library_names.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.get_library_names.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.get_library_names', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_library_names', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_library_names(...)' code ##################

        
        
        # Getting the type of 'self' (line 158)
        self_17700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 158, 15), 'self')
        # Obtaining the member 'libraries' of a type (line 158)
        libraries_17701 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 158, 15), self_17700, 'libraries')
        # Applying the 'not' unary operator (line 158)
        result_not__17702 = python_operator(stypy.reporting.localization.Localization(__file__, 158, 11), 'not', libraries_17701)
        
        # Testing the type of an if condition (line 158)
        if_condition_17703 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 158, 8), result_not__17702)
        # Assigning a type to the variable 'if_condition_17703' (line 158)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 158, 8), 'if_condition_17703', if_condition_17703)
        # SSA begins for if statement (line 158)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'None' (line 159)
        None_17704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 159, 19), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 159)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 159, 12), 'stypy_return_type', None_17704)
        # SSA join for if statement (line 158)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a List to a Name (line 161):
        
        # Assigning a List to a Name (line 161):
        
        # Obtaining an instance of the builtin type 'list' (line 161)
        list_17705 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 161, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 161)
        
        # Assigning a type to the variable 'lib_names' (line 161)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 161, 8), 'lib_names', list_17705)
        
        # Getting the type of 'self' (line 162)
        self_17706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 162, 38), 'self')
        # Obtaining the member 'libraries' of a type (line 162)
        libraries_17707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 162, 38), self_17706, 'libraries')
        # Testing the type of a for loop iterable (line 162)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 162, 8), libraries_17707)
        # Getting the type of the for loop variable (line 162)
        for_loop_var_17708 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 162, 8), libraries_17707)
        # Assigning a type to the variable 'lib_name' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'lib_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), for_loop_var_17708))
        # Assigning a type to the variable 'build_info' (line 162)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 162, 8), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 162, 8), for_loop_var_17708))
        # SSA begins for a for statement (line 162)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Call to append(...): (line 163)
        # Processing the call arguments (line 163)
        # Getting the type of 'lib_name' (line 163)
        lib_name_17711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 29), 'lib_name', False)
        # Processing the call keyword arguments (line 163)
        kwargs_17712 = {}
        # Getting the type of 'lib_names' (line 163)
        lib_names_17709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 163, 12), 'lib_names', False)
        # Obtaining the member 'append' of a type (line 163)
        append_17710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 163, 12), lib_names_17709, 'append')
        # Calling append(args, kwargs) (line 163)
        append_call_result_17713 = invoke(stypy.reporting.localization.Localization(__file__, 163, 12), append_17710, *[lib_name_17711], **kwargs_17712)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'lib_names' (line 164)
        lib_names_17714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 164, 15), 'lib_names')
        # Assigning a type to the variable 'stypy_return_type' (line 164)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 164, 8), 'stypy_return_type', lib_names_17714)
        
        # ################# End of 'get_library_names(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_library_names' in the type store
        # Getting the type of 'stypy_return_type' (line 155)
        stypy_return_type_17715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 155, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17715)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_library_names'
        return stypy_return_type_17715


    @norecursion
    def get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_source_files'
        module_type_store = module_type_store.open_function_context('get_source_files', 167, 4, False)
        # Assigning a type to the variable 'self' (line 168)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 168, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.get_source_files.__dict__.__setitem__('stypy_localization', localization)
        build_clib.get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.get_source_files.__dict__.__setitem__('stypy_function_name', 'build_clib.get_source_files')
        build_clib.get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        build_clib.get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.get_source_files', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_source_files', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_source_files(...)' code ##################

        
        # Call to check_library_list(...): (line 168)
        # Processing the call arguments (line 168)
        # Getting the type of 'self' (line 168)
        self_17718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 32), 'self', False)
        # Obtaining the member 'libraries' of a type (line 168)
        libraries_17719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 32), self_17718, 'libraries')
        # Processing the call keyword arguments (line 168)
        kwargs_17720 = {}
        # Getting the type of 'self' (line 168)
        self_17716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 168, 8), 'self', False)
        # Obtaining the member 'check_library_list' of a type (line 168)
        check_library_list_17717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 168, 8), self_17716, 'check_library_list')
        # Calling check_library_list(args, kwargs) (line 168)
        check_library_list_call_result_17721 = invoke(stypy.reporting.localization.Localization(__file__, 168, 8), check_library_list_17717, *[libraries_17719], **kwargs_17720)
        
        
        # Assigning a List to a Name (line 169):
        
        # Assigning a List to a Name (line 169):
        
        # Obtaining an instance of the builtin type 'list' (line 169)
        list_17722 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 169, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 169)
        
        # Assigning a type to the variable 'filenames' (line 169)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 169, 8), 'filenames', list_17722)
        
        # Getting the type of 'self' (line 170)
        self_17723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 170, 38), 'self')
        # Obtaining the member 'libraries' of a type (line 170)
        libraries_17724 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 170, 38), self_17723, 'libraries')
        # Testing the type of a for loop iterable (line 170)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 170, 8), libraries_17724)
        # Getting the type of the for loop variable (line 170)
        for_loop_var_17725 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 170, 8), libraries_17724)
        # Assigning a type to the variable 'lib_name' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'lib_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), for_loop_var_17725))
        # Assigning a type to the variable 'build_info' (line 170)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 170, 8), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 170, 8), for_loop_var_17725))
        # SSA begins for a for statement (line 170)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 171):
        
        # Assigning a Call to a Name (line 171):
        
        # Call to get(...): (line 171)
        # Processing the call arguments (line 171)
        str_17728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 171, 37), 'str', 'sources')
        # Processing the call keyword arguments (line 171)
        kwargs_17729 = {}
        # Getting the type of 'build_info' (line 171)
        build_info_17726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 171, 22), 'build_info', False)
        # Obtaining the member 'get' of a type (line 171)
        get_17727 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 171, 22), build_info_17726, 'get')
        # Calling get(args, kwargs) (line 171)
        get_call_result_17730 = invoke(stypy.reporting.localization.Localization(__file__, 171, 22), get_17727, *[str_17728], **kwargs_17729)
        
        # Assigning a type to the variable 'sources' (line 171)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 171, 12), 'sources', get_call_result_17730)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sources' (line 172)
        sources_17731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 15), 'sources')
        # Getting the type of 'None' (line 172)
        None_17732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 26), 'None')
        # Applying the binary operator 'is' (line 172)
        result_is__17733 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), 'is', sources_17731, None_17732)
        
        
        
        # Call to isinstance(...): (line 172)
        # Processing the call arguments (line 172)
        # Getting the type of 'sources' (line 172)
        sources_17735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 49), 'sources', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 172)
        tuple_17736 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 172, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 172)
        # Adding element type (line 172)
        # Getting the type of 'list' (line 172)
        list_17737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 59), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 59), tuple_17736, list_17737)
        # Adding element type (line 172)
        # Getting the type of 'tuple' (line 172)
        tuple_17738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 65), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 172, 59), tuple_17736, tuple_17738)
        
        # Processing the call keyword arguments (line 172)
        kwargs_17739 = {}
        # Getting the type of 'isinstance' (line 172)
        isinstance_17734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 172, 38), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 172)
        isinstance_call_result_17740 = invoke(stypy.reporting.localization.Localization(__file__, 172, 38), isinstance_17734, *[sources_17735, tuple_17736], **kwargs_17739)
        
        # Applying the 'not' unary operator (line 172)
        result_not__17741 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 34), 'not', isinstance_call_result_17740)
        
        # Applying the binary operator 'or' (line 172)
        result_or_keyword_17742 = python_operator(stypy.reporting.localization.Localization(__file__, 172, 15), 'or', result_is__17733, result_not__17741)
        
        # Testing the type of an if condition (line 172)
        if_condition_17743 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 172, 12), result_or_keyword_17742)
        # Assigning a type to the variable 'if_condition_17743' (line 172)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 172, 12), 'if_condition_17743', if_condition_17743)
        # SSA begins for if statement (line 172)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsSetupError' (line 173)
        DistutilsSetupError_17744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 173, 22), 'DistutilsSetupError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 173, 16), DistutilsSetupError_17744, 'raise parameter', BaseException)
        # SSA join for if statement (line 172)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to extend(...): (line 178)
        # Processing the call arguments (line 178)
        # Getting the type of 'sources' (line 178)
        sources_17747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 29), 'sources', False)
        # Processing the call keyword arguments (line 178)
        kwargs_17748 = {}
        # Getting the type of 'filenames' (line 178)
        filenames_17745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 178, 12), 'filenames', False)
        # Obtaining the member 'extend' of a type (line 178)
        extend_17746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 178, 12), filenames_17745, 'extend')
        # Calling extend(args, kwargs) (line 178)
        extend_call_result_17749 = invoke(stypy.reporting.localization.Localization(__file__, 178, 12), extend_17746, *[sources_17747], **kwargs_17748)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'filenames' (line 179)
        filenames_17750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 179, 15), 'filenames')
        # Assigning a type to the variable 'stypy_return_type' (line 179)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 179, 8), 'stypy_return_type', filenames_17750)
        
        # ################# End of 'get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 167)
        stypy_return_type_17751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 167, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_source_files'
        return stypy_return_type_17751


    @norecursion
    def build_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'build_libraries'
        module_type_store = module_type_store.open_function_context('build_libraries', 181, 4, False)
        # Assigning a type to the variable 'self' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_clib.build_libraries.__dict__.__setitem__('stypy_localization', localization)
        build_clib.build_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_clib.build_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_clib.build_libraries.__dict__.__setitem__('stypy_function_name', 'build_clib.build_libraries')
        build_clib.build_libraries.__dict__.__setitem__('stypy_param_names_list', ['libraries'])
        build_clib.build_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_clib.build_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_clib.build_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_clib.build_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_clib.build_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_clib.build_libraries.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.build_libraries', ['libraries'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'build_libraries', localization, ['libraries'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'build_libraries(...)' code ##################

        
        # Getting the type of 'libraries' (line 182)
        libraries_17752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 182, 38), 'libraries')
        # Testing the type of a for loop iterable (line 182)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 182, 8), libraries_17752)
        # Getting the type of the for loop variable (line 182)
        for_loop_var_17753 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 182, 8), libraries_17752)
        # Assigning a type to the variable 'lib_name' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'lib_name', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 8), for_loop_var_17753))
        # Assigning a type to the variable 'build_info' (line 182)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 182, 8), 'build_info', get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 182, 8), for_loop_var_17753))
        # SSA begins for a for statement (line 182)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 183):
        
        # Assigning a Call to a Name (line 183):
        
        # Call to get(...): (line 183)
        # Processing the call arguments (line 183)
        str_17756 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 183, 37), 'str', 'sources')
        # Processing the call keyword arguments (line 183)
        kwargs_17757 = {}
        # Getting the type of 'build_info' (line 183)
        build_info_17754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 183, 22), 'build_info', False)
        # Obtaining the member 'get' of a type (line 183)
        get_17755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 183, 22), build_info_17754, 'get')
        # Calling get(args, kwargs) (line 183)
        get_call_result_17758 = invoke(stypy.reporting.localization.Localization(__file__, 183, 22), get_17755, *[str_17756], **kwargs_17757)
        
        # Assigning a type to the variable 'sources' (line 183)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 183, 12), 'sources', get_call_result_17758)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'sources' (line 184)
        sources_17759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 15), 'sources')
        # Getting the type of 'None' (line 184)
        None_17760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 26), 'None')
        # Applying the binary operator 'is' (line 184)
        result_is__17761 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), 'is', sources_17759, None_17760)
        
        
        
        # Call to isinstance(...): (line 184)
        # Processing the call arguments (line 184)
        # Getting the type of 'sources' (line 184)
        sources_17763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 49), 'sources', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 184)
        tuple_17764 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 184, 59), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 184)
        # Adding element type (line 184)
        # Getting the type of 'list' (line 184)
        list_17765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 59), 'list', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 59), tuple_17764, list_17765)
        # Adding element type (line 184)
        # Getting the type of 'tuple' (line 184)
        tuple_17766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 65), 'tuple', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 184, 59), tuple_17764, tuple_17766)
        
        # Processing the call keyword arguments (line 184)
        kwargs_17767 = {}
        # Getting the type of 'isinstance' (line 184)
        isinstance_17762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 184, 38), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 184)
        isinstance_call_result_17768 = invoke(stypy.reporting.localization.Localization(__file__, 184, 38), isinstance_17762, *[sources_17763, tuple_17764], **kwargs_17767)
        
        # Applying the 'not' unary operator (line 184)
        result_not__17769 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 34), 'not', isinstance_call_result_17768)
        
        # Applying the binary operator 'or' (line 184)
        result_or_keyword_17770 = python_operator(stypy.reporting.localization.Localization(__file__, 184, 15), 'or', result_is__17761, result_not__17769)
        
        # Testing the type of an if condition (line 184)
        if_condition_17771 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 184, 12), result_or_keyword_17770)
        # Assigning a type to the variable 'if_condition_17771' (line 184)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 184, 12), 'if_condition_17771', if_condition_17771)
        # SSA begins for if statement (line 184)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'DistutilsSetupError' (line 185)
        DistutilsSetupError_17772 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 185, 22), 'DistutilsSetupError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 185, 16), DistutilsSetupError_17772, 'raise parameter', BaseException)
        # SSA join for if statement (line 184)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 189):
        
        # Assigning a Call to a Name (line 189):
        
        # Call to list(...): (line 189)
        # Processing the call arguments (line 189)
        # Getting the type of 'sources' (line 189)
        sources_17774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 27), 'sources', False)
        # Processing the call keyword arguments (line 189)
        kwargs_17775 = {}
        # Getting the type of 'list' (line 189)
        list_17773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 189, 22), 'list', False)
        # Calling list(args, kwargs) (line 189)
        list_call_result_17776 = invoke(stypy.reporting.localization.Localization(__file__, 189, 22), list_17773, *[sources_17774], **kwargs_17775)
        
        # Assigning a type to the variable 'sources' (line 189)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 189, 12), 'sources', list_call_result_17776)
        
        # Call to info(...): (line 191)
        # Processing the call arguments (line 191)
        str_17779 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 191, 21), 'str', "building '%s' library")
        # Getting the type of 'lib_name' (line 191)
        lib_name_17780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 46), 'lib_name', False)
        # Processing the call keyword arguments (line 191)
        kwargs_17781 = {}
        # Getting the type of 'log' (line 191)
        log_17777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 191, 12), 'log', False)
        # Obtaining the member 'info' of a type (line 191)
        info_17778 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 191, 12), log_17777, 'info')
        # Calling info(args, kwargs) (line 191)
        info_call_result_17782 = invoke(stypy.reporting.localization.Localization(__file__, 191, 12), info_17778, *[str_17779, lib_name_17780], **kwargs_17781)
        
        
        # Assigning a Call to a Name (line 196):
        
        # Assigning a Call to a Name (line 196):
        
        # Call to get(...): (line 196)
        # Processing the call arguments (line 196)
        str_17785 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 196, 36), 'str', 'macros')
        # Processing the call keyword arguments (line 196)
        kwargs_17786 = {}
        # Getting the type of 'build_info' (line 196)
        build_info_17783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 196, 21), 'build_info', False)
        # Obtaining the member 'get' of a type (line 196)
        get_17784 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 196, 21), build_info_17783, 'get')
        # Calling get(args, kwargs) (line 196)
        get_call_result_17787 = invoke(stypy.reporting.localization.Localization(__file__, 196, 21), get_17784, *[str_17785], **kwargs_17786)
        
        # Assigning a type to the variable 'macros' (line 196)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 196, 12), 'macros', get_call_result_17787)
        
        # Assigning a Call to a Name (line 197):
        
        # Assigning a Call to a Name (line 197):
        
        # Call to get(...): (line 197)
        # Processing the call arguments (line 197)
        str_17790 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 197, 42), 'str', 'include_dirs')
        # Processing the call keyword arguments (line 197)
        kwargs_17791 = {}
        # Getting the type of 'build_info' (line 197)
        build_info_17788 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 197, 27), 'build_info', False)
        # Obtaining the member 'get' of a type (line 197)
        get_17789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 197, 27), build_info_17788, 'get')
        # Calling get(args, kwargs) (line 197)
        get_call_result_17792 = invoke(stypy.reporting.localization.Localization(__file__, 197, 27), get_17789, *[str_17790], **kwargs_17791)
        
        # Assigning a type to the variable 'include_dirs' (line 197)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 197, 12), 'include_dirs', get_call_result_17792)
        
        # Assigning a Call to a Name (line 198):
        
        # Assigning a Call to a Name (line 198):
        
        # Call to compile(...): (line 198)
        # Processing the call arguments (line 198)
        # Getting the type of 'sources' (line 198)
        sources_17796 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 44), 'sources', False)
        # Processing the call keyword arguments (line 198)
        # Getting the type of 'self' (line 199)
        self_17797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 199, 55), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 199)
        build_temp_17798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 199, 55), self_17797, 'build_temp')
        keyword_17799 = build_temp_17798
        # Getting the type of 'macros' (line 200)
        macros_17800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 200, 51), 'macros', False)
        keyword_17801 = macros_17800
        # Getting the type of 'include_dirs' (line 201)
        include_dirs_17802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 201, 57), 'include_dirs', False)
        keyword_17803 = include_dirs_17802
        # Getting the type of 'self' (line 202)
        self_17804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 202, 50), 'self', False)
        # Obtaining the member 'debug' of a type (line 202)
        debug_17805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 202, 50), self_17804, 'debug')
        keyword_17806 = debug_17805
        kwargs_17807 = {'debug': keyword_17806, 'macros': keyword_17801, 'output_dir': keyword_17799, 'include_dirs': keyword_17803}
        # Getting the type of 'self' (line 198)
        self_17793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 198, 22), 'self', False)
        # Obtaining the member 'compiler' of a type (line 198)
        compiler_17794 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 22), self_17793, 'compiler')
        # Obtaining the member 'compile' of a type (line 198)
        compile_17795 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 198, 22), compiler_17794, 'compile')
        # Calling compile(args, kwargs) (line 198)
        compile_call_result_17808 = invoke(stypy.reporting.localization.Localization(__file__, 198, 22), compile_17795, *[sources_17796], **kwargs_17807)
        
        # Assigning a type to the variable 'objects' (line 198)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 198, 12), 'objects', compile_call_result_17808)
        
        # Call to create_static_lib(...): (line 207)
        # Processing the call arguments (line 207)
        # Getting the type of 'objects' (line 207)
        objects_17812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 44), 'objects', False)
        # Getting the type of 'lib_name' (line 207)
        lib_name_17813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 53), 'lib_name', False)
        # Processing the call keyword arguments (line 207)
        # Getting the type of 'self' (line 208)
        self_17814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 208, 55), 'self', False)
        # Obtaining the member 'build_clib' of a type (line 208)
        build_clib_17815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 208, 55), self_17814, 'build_clib')
        keyword_17816 = build_clib_17815
        # Getting the type of 'self' (line 209)
        self_17817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 209, 50), 'self', False)
        # Obtaining the member 'debug' of a type (line 209)
        debug_17818 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 209, 50), self_17817, 'debug')
        keyword_17819 = debug_17818
        kwargs_17820 = {'debug': keyword_17819, 'output_dir': keyword_17816}
        # Getting the type of 'self' (line 207)
        self_17809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 207, 12), 'self', False)
        # Obtaining the member 'compiler' of a type (line 207)
        compiler_17810 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), self_17809, 'compiler')
        # Obtaining the member 'create_static_lib' of a type (line 207)
        create_static_lib_17811 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 207, 12), compiler_17810, 'create_static_lib')
        # Calling create_static_lib(args, kwargs) (line 207)
        create_static_lib_call_result_17821 = invoke(stypy.reporting.localization.Localization(__file__, 207, 12), create_static_lib_17811, *[objects_17812, lib_name_17813], **kwargs_17820)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'build_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'build_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 181)
        stypy_return_type_17822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 181, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_17822)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'build_libraries'
        return stypy_return_type_17822


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 30, 0, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_clib.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_clib' (line 30)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'build_clib', build_clib)

# Assigning a Str to a Name (line 32):
str_17823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 18), 'str', 'build C/C++ libraries used by Python extensions')
# Getting the type of 'build_clib'
build_clib_17824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_clib')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_clib_17824, 'description', str_17823)

# Assigning a List to a Name (line 34):

# Obtaining an instance of the builtin type 'list' (line 34)
list_17825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 34)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_17826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
str_17827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'str', 'build-clib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_17826, str_17827)
# Adding element type (line 35)
str_17828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_17826, str_17828)
# Adding element type (line 35)
str_17829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'str', 'directory to build C/C++ libraries to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_17826, str_17829)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), list_17825, tuple_17826)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 37)
tuple_17830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 37)
# Adding element type (line 37)
str_17831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 9), 'str', 'build-temp=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_17830, str_17831)
# Adding element type (line 37)
str_17832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 24), 'str', 't')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_17830, str_17832)
# Adding element type (line 37)
str_17833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 9), 'str', 'directory to put temporary build by-products')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 9), tuple_17830, str_17833)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), list_17825, tuple_17830)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 39)
tuple_17834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 39)
# Adding element type (line 39)
str_17835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 9), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_17834, str_17835)
# Adding element type (line 39)
str_17836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 18), 'str', 'g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_17834, str_17836)
# Adding element type (line 39)
str_17837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'str', 'compile with debugging information')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 9), tuple_17834, str_17837)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), list_17825, tuple_17834)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_17838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
str_17839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 9), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_17838, str_17839)
# Adding element type (line 41)
str_17840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 18), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_17838, str_17840)
# Adding element type (line 41)
str_17841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 9), 'str', 'forcibly build everything (ignore file timestamps)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 9), tuple_17838, str_17841)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), list_17825, tuple_17838)
# Adding element type (line 34)

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_17842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
str_17843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 9), 'str', 'compiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_17842, str_17843)
# Adding element type (line 43)
str_17844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 22), 'str', 'c')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_17842, str_17844)
# Adding element type (line 43)
str_17845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 9), 'str', 'specify the compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 9), tuple_17842, str_17845)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 19), list_17825, tuple_17842)

# Getting the type of 'build_clib'
build_clib_17846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_clib')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_clib_17846, 'user_options', list_17825)

# Assigning a List to a Name (line 47):

# Obtaining an instance of the builtin type 'list' (line 47)
list_17847 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 47)
# Adding element type (line 47)
str_17848 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 23), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 22), list_17847, str_17848)
# Adding element type (line 47)
str_17849 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 32), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 22), list_17847, str_17849)

# Getting the type of 'build_clib'
build_clib_17850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_clib')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_clib_17850, 'boolean_options', list_17847)

# Assigning a List to a Name (line 49):

# Obtaining an instance of the builtin type 'list' (line 49)
list_17851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 49)
# Adding element type (line 49)

# Obtaining an instance of the builtin type 'tuple' (line 50)
tuple_17852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 50)
# Adding element type (line 50)
str_17853 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 9), 'str', 'help-compiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_17852, str_17853)
# Adding element type (line 50)
# Getting the type of 'None' (line 50)
None_17854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 26), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_17852, None_17854)
# Adding element type (line 50)
str_17855 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 9), 'str', 'list available compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_17852, str_17855)
# Adding element type (line 50)
# Getting the type of 'show_compilers' (line 51)
show_compilers_17856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 37), 'show_compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 50, 9), tuple_17852, show_compilers_17856)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 19), list_17851, tuple_17852)

# Getting the type of 'build_clib'
build_clib_17857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build_clib')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_clib_17857, 'help_options', list_17851)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
