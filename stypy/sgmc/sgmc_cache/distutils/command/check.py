
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.check
2: 
3: Implements the Distutils 'check' command.
4: '''
5: __revision__ = "$Id$"
6: 
7: from distutils.core import Command
8: from distutils.dist import PKG_INFO_ENCODING
9: from distutils.errors import DistutilsSetupError
10: 
11: try:
12:     # docutils is installed
13:     from docutils.utils import Reporter
14:     from docutils.parsers.rst import Parser
15:     from docutils import frontend
16:     from docutils import nodes
17:     from StringIO import StringIO
18: 
19:     class SilentReporter(Reporter):
20: 
21:         def __init__(self, source, report_level, halt_level, stream=None,
22:                      debug=0, encoding='ascii', error_handler='replace'):
23:             self.messages = []
24:             Reporter.__init__(self, source, report_level, halt_level, stream,
25:                               debug, encoding, error_handler)
26: 
27:         def system_message(self, level, message, *children, **kwargs):
28:             self.messages.append((level, message, children, kwargs))
29:             return nodes.system_message(message, level=level,
30:                                         type=self.levels[level],
31:                                         *children, **kwargs)
32: 
33:     HAS_DOCUTILS = True
34: except ImportError:
35:     # docutils is not installed
36:     HAS_DOCUTILS = False
37: 
38: class check(Command):
39:     '''This command checks the meta-data of the package.
40:     '''
41:     description = ("perform some checks on the package")
42:     user_options = [('metadata', 'm', 'Verify meta-data'),
43:                     ('restructuredtext', 'r',
44:                      ('Checks if long string meta-data syntax '
45:                       'are reStructuredText-compliant')),
46:                     ('strict', 's',
47:                      'Will exit with an error if a check fails')]
48: 
49:     boolean_options = ['metadata', 'restructuredtext', 'strict']
50: 
51:     def initialize_options(self):
52:         '''Sets default values for options.'''
53:         self.restructuredtext = 0
54:         self.metadata = 1
55:         self.strict = 0
56:         self._warnings = 0
57: 
58:     def finalize_options(self):
59:         pass
60: 
61:     def warn(self, msg):
62:         '''Counts the number of warnings that occurs.'''
63:         self._warnings += 1
64:         return Command.warn(self, msg)
65: 
66:     def run(self):
67:         '''Runs the command.'''
68:         # perform the various tests
69:         if self.metadata:
70:             self.check_metadata()
71:         if self.restructuredtext:
72:             if HAS_DOCUTILS:
73:                 self.check_restructuredtext()
74:             elif self.strict:
75:                 raise DistutilsSetupError('The docutils package is needed.')
76: 
77:         # let's raise an error in strict mode, if we have at least
78:         # one warning
79:         if self.strict and self._warnings > 0:
80:             raise DistutilsSetupError('Please correct your package.')
81: 
82:     def check_metadata(self):
83:         '''Ensures that all required elements of meta-data are supplied.
84: 
85:         name, version, URL, (author and author_email) or
86:         (maintainer and maintainer_email)).
87: 
88:         Warns if any are missing.
89:         '''
90:         metadata = self.distribution.metadata
91: 
92:         missing = []
93:         for attr in ('name', 'version', 'url'):
94:             if not (hasattr(metadata, attr) and getattr(metadata, attr)):
95:                 missing.append(attr)
96: 
97:         if missing:
98:             self.warn("missing required meta-data: %s"  % ', '.join(missing))
99:         if metadata.author:
100:             if not metadata.author_email:
101:                 self.warn("missing meta-data: if 'author' supplied, " +
102:                           "'author_email' must be supplied too")
103:         elif metadata.maintainer:
104:             if not metadata.maintainer_email:
105:                 self.warn("missing meta-data: if 'maintainer' supplied, " +
106:                           "'maintainer_email' must be supplied too")
107:         else:
108:             self.warn("missing meta-data: either (author and author_email) " +
109:                       "or (maintainer and maintainer_email) " +
110:                       "must be supplied")
111: 
112:     def check_restructuredtext(self):
113:         '''Checks if the long string fields are reST-compliant.'''
114:         data = self.distribution.get_long_description()
115:         if not isinstance(data, unicode):
116:             data = data.decode(PKG_INFO_ENCODING)
117:         for warning in self._check_rst_data(data):
118:             line = warning[-1].get('line')
119:             if line is None:
120:                 warning = warning[1]
121:             else:
122:                 warning = '%s (line %s)' % (warning[1], line)
123:             self.warn(warning)
124: 
125:     def _check_rst_data(self, data):
126:         '''Returns warnings when the provided data doesn't compile.'''
127:         source_path = StringIO()
128:         parser = Parser()
129:         settings = frontend.OptionParser(components=(Parser,)).get_default_values()
130:         settings.tab_width = 4
131:         settings.pep_references = None
132:         settings.rfc_references = None
133:         reporter = SilentReporter(source_path,
134:                           settings.report_level,
135:                           settings.halt_level,
136:                           stream=settings.warning_stream,
137:                           debug=settings.debug,
138:                           encoding=settings.error_encoding,
139:                           error_handler=settings.error_encoding_error_handler)
140: 
141:         document = nodes.document(settings, reporter, source=source_path)
142:         document.note_source(source_path, -1)
143:         try:
144:             parser.parse(data, document)
145:         except AttributeError as e:
146:             reporter.messages.append(
147:                 (-1, 'Could not finish the parsing: %s.' % e, '', {}))
148: 
149:         return reporter.messages
150: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_20940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.check\n\nImplements the Distutils 'check' command.\n")

# Assigning a Str to a Name (line 5):
str_20941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), '__revision__', str_20941)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.core import Command' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20942 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core')

if (type(import_20942) is not StypyTypeError):

    if (import_20942 != 'pyd_module'):
        __import__(import_20942)
        sys_modules_20943 = sys.modules[import_20942]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', sys_modules_20943.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_20943, sys_modules_20943.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.core', import_20942)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.dist import PKG_INFO_ENCODING' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20944 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.dist')

if (type(import_20944) is not StypyTypeError):

    if (import_20944 != 'pyd_module'):
        __import__(import_20944)
        sys_modules_20945 = sys.modules[import_20944]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.dist', sys_modules_20945.module_type_store, module_type_store, ['PKG_INFO_ENCODING'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_20945, sys_modules_20945.module_type_store, module_type_store)
    else:
        from distutils.dist import PKG_INFO_ENCODING

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.dist', None, module_type_store, ['PKG_INFO_ENCODING'], [PKG_INFO_ENCODING])

else:
    # Assigning a type to the variable 'distutils.dist' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.dist', import_20944)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'from distutils.errors import DistutilsSetupError' statement (line 9)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20946 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors')

if (type(import_20946) is not StypyTypeError):

    if (import_20946 != 'pyd_module'):
        __import__(import_20946)
        sys_modules_20947 = sys.modules[import_20946]
        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', sys_modules_20947.module_type_store, module_type_store, ['DistutilsSetupError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 9, 0), __file__, sys_modules_20947, sys_modules_20947.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsSetupError

        import_from_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', None, module_type_store, ['DistutilsSetupError'], [DistutilsSetupError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 9)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'distutils.errors', import_20946)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')



# SSA begins for try-except statement (line 11)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))

# 'from docutils.utils import Reporter' statement (line 13)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20948 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'docutils.utils')

if (type(import_20948) is not StypyTypeError):

    if (import_20948 != 'pyd_module'):
        __import__(import_20948)
        sys_modules_20949 = sys.modules[import_20948]
        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'docutils.utils', sys_modules_20949.module_type_store, module_type_store, ['Reporter'])
        nest_module(stypy.reporting.localization.Localization(__file__, 13, 4), __file__, sys_modules_20949, sys_modules_20949.module_type_store, module_type_store)
    else:
        from docutils.utils import Reporter

        import_from_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'docutils.utils', None, module_type_store, ['Reporter'], [Reporter])

else:
    # Assigning a type to the variable 'docutils.utils' (line 13)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'docutils.utils', import_20948)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 14, 4))

# 'from docutils.parsers.rst import Parser' statement (line 14)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20950 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'docutils.parsers.rst')

if (type(import_20950) is not StypyTypeError):

    if (import_20950 != 'pyd_module'):
        __import__(import_20950)
        sys_modules_20951 = sys.modules[import_20950]
        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'docutils.parsers.rst', sys_modules_20951.module_type_store, module_type_store, ['Parser'])
        nest_module(stypy.reporting.localization.Localization(__file__, 14, 4), __file__, sys_modules_20951, sys_modules_20951.module_type_store, module_type_store)
    else:
        from docutils.parsers.rst import Parser

        import_from_module(stypy.reporting.localization.Localization(__file__, 14, 4), 'docutils.parsers.rst', None, module_type_store, ['Parser'], [Parser])

else:
    # Assigning a type to the variable 'docutils.parsers.rst' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'docutils.parsers.rst', import_20950)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 15, 4))

# 'from docutils import frontend' statement (line 15)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20952 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'docutils')

if (type(import_20952) is not StypyTypeError):

    if (import_20952 != 'pyd_module'):
        __import__(import_20952)
        sys_modules_20953 = sys.modules[import_20952]
        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'docutils', sys_modules_20953.module_type_store, module_type_store, ['frontend'])
        nest_module(stypy.reporting.localization.Localization(__file__, 15, 4), __file__, sys_modules_20953, sys_modules_20953.module_type_store, module_type_store)
    else:
        from docutils import frontend

        import_from_module(stypy.reporting.localization.Localization(__file__, 15, 4), 'docutils', None, module_type_store, ['frontend'], [frontend])

else:
    # Assigning a type to the variable 'docutils' (line 15)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'docutils', import_20952)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 16, 4))

# 'from docutils import nodes' statement (line 16)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_20954 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'docutils')

if (type(import_20954) is not StypyTypeError):

    if (import_20954 != 'pyd_module'):
        __import__(import_20954)
        sys_modules_20955 = sys.modules[import_20954]
        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'docutils', sys_modules_20955.module_type_store, module_type_store, ['nodes'])
        nest_module(stypy.reporting.localization.Localization(__file__, 16, 4), __file__, sys_modules_20955, sys_modules_20955.module_type_store, module_type_store)
    else:
        from docutils import nodes

        import_from_module(stypy.reporting.localization.Localization(__file__, 16, 4), 'docutils', None, module_type_store, ['nodes'], [nodes])

else:
    # Assigning a type to the variable 'docutils' (line 16)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'docutils', import_20954)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 17, 4))

# 'from StringIO import StringIO' statement (line 17)
try:
    from StringIO import StringIO

except:
    StringIO = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 17, 4), 'StringIO', None, module_type_store, ['StringIO'], [StringIO])

# Declaration of the 'SilentReporter' class
# Getting the type of 'Reporter' (line 19)
Reporter_20956 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'Reporter')

class SilentReporter(Reporter_20956, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 21)
        None_20957 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 68), 'None')
        int_20958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 27), 'int')
        str_20959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 39), 'str', 'ascii')
        str_20960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 62), 'str', 'replace')
        defaults = [None_20957, int_20958, str_20959, str_20960]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 21, 8, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SilentReporter.__init__', ['source', 'report_level', 'halt_level', 'stream', 'debug', 'encoding', 'error_handler'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['source', 'report_level', 'halt_level', 'stream', 'debug', 'encoding', 'error_handler'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Assigning a List to a Attribute (line 23):
        
        # Obtaining an instance of the builtin type 'list' (line 23)
        list_20961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 28), 'list')
        # Adding type elements to the builtin type 'list' instance (line 23)
        
        # Getting the type of 'self' (line 23)
        self_20962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 12), 'self')
        # Setting the type of the member 'messages' of a type (line 23)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 12), self_20962, 'messages', list_20961)
        
        # Call to __init__(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'self' (line 24)
        self_20965 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 30), 'self', False)
        # Getting the type of 'source' (line 24)
        source_20966 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 36), 'source', False)
        # Getting the type of 'report_level' (line 24)
        report_level_20967 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 44), 'report_level', False)
        # Getting the type of 'halt_level' (line 24)
        halt_level_20968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 58), 'halt_level', False)
        # Getting the type of 'stream' (line 24)
        stream_20969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 70), 'stream', False)
        # Getting the type of 'debug' (line 25)
        debug_20970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 30), 'debug', False)
        # Getting the type of 'encoding' (line 25)
        encoding_20971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 37), 'encoding', False)
        # Getting the type of 'error_handler' (line 25)
        error_handler_20972 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 47), 'error_handler', False)
        # Processing the call keyword arguments (line 24)
        kwargs_20973 = {}
        # Getting the type of 'Reporter' (line 24)
        Reporter_20963 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 12), 'Reporter', False)
        # Obtaining the member '__init__' of a type (line 24)
        init___20964 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 12), Reporter_20963, '__init__')
        # Calling __init__(args, kwargs) (line 24)
        init___call_result_20974 = invoke(stypy.reporting.localization.Localization(__file__, 24, 12), init___20964, *[self_20965, source_20966, report_level_20967, halt_level_20968, stream_20969, debug_20970, encoding_20971, error_handler_20972], **kwargs_20973)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def system_message(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'system_message'
        module_type_store = module_type_store.open_function_context('system_message', 27, 8, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self', type_of_self)
        
        # Passed parameters checking function
        SilentReporter.system_message.__dict__.__setitem__('stypy_localization', localization)
        SilentReporter.system_message.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SilentReporter.system_message.__dict__.__setitem__('stypy_type_store', module_type_store)
        SilentReporter.system_message.__dict__.__setitem__('stypy_function_name', 'SilentReporter.system_message')
        SilentReporter.system_message.__dict__.__setitem__('stypy_param_names_list', ['level', 'message'])
        SilentReporter.system_message.__dict__.__setitem__('stypy_varargs_param_name', 'children')
        SilentReporter.system_message.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        SilentReporter.system_message.__dict__.__setitem__('stypy_call_defaults', defaults)
        SilentReporter.system_message.__dict__.__setitem__('stypy_call_varargs', varargs)
        SilentReporter.system_message.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SilentReporter.system_message.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SilentReporter.system_message', ['level', 'message'], 'children', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'system_message', localization, ['level', 'message'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'system_message(...)' code ##################

        
        # Call to append(...): (line 28)
        # Processing the call arguments (line 28)
        
        # Obtaining an instance of the builtin type 'tuple' (line 28)
        tuple_20978 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 34), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 28)
        # Adding element type (line 28)
        # Getting the type of 'level' (line 28)
        level_20979 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 34), 'level', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_20978, level_20979)
        # Adding element type (line 28)
        # Getting the type of 'message' (line 28)
        message_20980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 41), 'message', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_20978, message_20980)
        # Adding element type (line 28)
        # Getting the type of 'children' (line 28)
        children_20981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 50), 'children', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_20978, children_20981)
        # Adding element type (line 28)
        # Getting the type of 'kwargs' (line 28)
        kwargs_20982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 60), 'kwargs', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 34), tuple_20978, kwargs_20982)
        
        # Processing the call keyword arguments (line 28)
        kwargs_20983 = {}
        # Getting the type of 'self' (line 28)
        self_20975 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 12), 'self', False)
        # Obtaining the member 'messages' of a type (line 28)
        messages_20976 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), self_20975, 'messages')
        # Obtaining the member 'append' of a type (line 28)
        append_20977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 12), messages_20976, 'append')
        # Calling append(args, kwargs) (line 28)
        append_call_result_20984 = invoke(stypy.reporting.localization.Localization(__file__, 28, 12), append_20977, *[tuple_20978], **kwargs_20983)
        
        
        # Call to system_message(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'message' (line 29)
        message_20987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 40), 'message', False)
        # Getting the type of 'children' (line 31)
        children_20988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 41), 'children', False)
        # Processing the call keyword arguments (line 29)
        # Getting the type of 'level' (line 29)
        level_20989 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 55), 'level', False)
        keyword_20990 = level_20989
        
        # Obtaining the type of the subscript
        # Getting the type of 'level' (line 30)
        level_20991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 57), 'level', False)
        # Getting the type of 'self' (line 30)
        self_20992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 45), 'self', False)
        # Obtaining the member 'levels' of a type (line 30)
        levels_20993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 45), self_20992, 'levels')
        # Obtaining the member '__getitem__' of a type (line 30)
        getitem___20994 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 45), levels_20993, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 30)
        subscript_call_result_20995 = invoke(stypy.reporting.localization.Localization(__file__, 30, 45), getitem___20994, level_20991)
        
        keyword_20996 = subscript_call_result_20995
        # Getting the type of 'kwargs' (line 31)
        kwargs_20997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 53), 'kwargs', False)
        kwargs_20998 = {'kwargs_20997': kwargs_20997, 'type': keyword_20996, 'level': keyword_20990}
        # Getting the type of 'nodes' (line 29)
        nodes_20985 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 19), 'nodes', False)
        # Obtaining the member 'system_message' of a type (line 29)
        system_message_20986 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 19), nodes_20985, 'system_message')
        # Calling system_message(args, kwargs) (line 29)
        system_message_call_result_20999 = invoke(stypy.reporting.localization.Localization(__file__, 29, 19), system_message_20986, *[message_20987, children_20988], **kwargs_20998)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'stypy_return_type', system_message_call_result_20999)
        
        # ################# End of 'system_message(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'system_message' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_21000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21000)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'system_message'
        return stypy_return_type_21000


# Assigning a type to the variable 'SilentReporter' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'SilentReporter', SilentReporter)

# Assigning a Name to a Name (line 33):
# Getting the type of 'True' (line 33)
True_21001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 19), 'True')
# Assigning a type to the variable 'HAS_DOCUTILS' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'HAS_DOCUTILS', True_21001)
# SSA branch for the except part of a try statement (line 11)
# SSA branch for the except 'ImportError' branch of a try statement (line 11)
module_type_store.open_ssa_branch('except')

# Assigning a Name to a Name (line 36):
# Getting the type of 'False' (line 36)
False_21002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'False')
# Assigning a type to the variable 'HAS_DOCUTILS' (line 36)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'HAS_DOCUTILS', False_21002)
# SSA join for try-except statement (line 11)
module_type_store = module_type_store.join_ssa_context()

# Declaration of the 'check' class
# Getting the type of 'Command' (line 38)
Command_21003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'Command')

class check(Command_21003, ):
    str_21004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, (-1)), 'str', 'This command checks the meta-data of the package.\n    ')

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        check.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        check.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        check.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        check.initialize_options.__dict__.__setitem__('stypy_function_name', 'check.initialize_options')
        check.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        check.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        check.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        check.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        check.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        check.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        check.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'check.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        str_21005 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 8), 'str', 'Sets default values for options.')
        
        # Assigning a Num to a Attribute (line 53):
        int_21006 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 32), 'int')
        # Getting the type of 'self' (line 53)
        self_21007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'restructuredtext' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_21007, 'restructuredtext', int_21006)
        
        # Assigning a Num to a Attribute (line 54):
        int_21008 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 24), 'int')
        # Getting the type of 'self' (line 54)
        self_21009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member 'metadata' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_21009, 'metadata', int_21008)
        
        # Assigning a Num to a Attribute (line 55):
        int_21010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 22), 'int')
        # Getting the type of 'self' (line 55)
        self_21011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'strict' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_21011, 'strict', int_21010)
        
        # Assigning a Num to a Attribute (line 56):
        int_21012 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, 25), 'int')
        # Getting the type of 'self' (line 56)
        self_21013 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member '_warnings' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_21013, '_warnings', int_21012)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_21014 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21014)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_21014


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        check.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        check.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        check.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        check.finalize_options.__dict__.__setitem__('stypy_function_name', 'check.finalize_options')
        check.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        check.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        check.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        check.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        check.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        check.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        check.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'check.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        pass
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_21015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21015)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_21015


    @norecursion
    def warn(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'warn'
        module_type_store = module_type_store.open_function_context('warn', 61, 4, False)
        # Assigning a type to the variable 'self' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        check.warn.__dict__.__setitem__('stypy_localization', localization)
        check.warn.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        check.warn.__dict__.__setitem__('stypy_type_store', module_type_store)
        check.warn.__dict__.__setitem__('stypy_function_name', 'check.warn')
        check.warn.__dict__.__setitem__('stypy_param_names_list', ['msg'])
        check.warn.__dict__.__setitem__('stypy_varargs_param_name', None)
        check.warn.__dict__.__setitem__('stypy_kwargs_param_name', None)
        check.warn.__dict__.__setitem__('stypy_call_defaults', defaults)
        check.warn.__dict__.__setitem__('stypy_call_varargs', varargs)
        check.warn.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        check.warn.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'check.warn', ['msg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'warn', localization, ['msg'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'warn(...)' code ##################

        str_21016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 8), 'str', 'Counts the number of warnings that occurs.')
        
        # Getting the type of 'self' (line 63)
        self_21017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Obtaining the member '_warnings' of a type (line 63)
        _warnings_21018 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_21017, '_warnings')
        int_21019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 26), 'int')
        # Applying the binary operator '+=' (line 63)
        result_iadd_21020 = python_operator(stypy.reporting.localization.Localization(__file__, 63, 8), '+=', _warnings_21018, int_21019)
        # Getting the type of 'self' (line 63)
        self_21021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'self')
        # Setting the type of the member '_warnings' of a type (line 63)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 8), self_21021, '_warnings', result_iadd_21020)
        
        
        # Call to warn(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'self' (line 64)
        self_21024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 28), 'self', False)
        # Getting the type of 'msg' (line 64)
        msg_21025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'msg', False)
        # Processing the call keyword arguments (line 64)
        kwargs_21026 = {}
        # Getting the type of 'Command' (line 64)
        Command_21022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'Command', False)
        # Obtaining the member 'warn' of a type (line 64)
        warn_21023 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 15), Command_21022, 'warn')
        # Calling warn(args, kwargs) (line 64)
        warn_call_result_21027 = invoke(stypy.reporting.localization.Localization(__file__, 64, 15), warn_21023, *[self_21024, msg_21025], **kwargs_21026)
        
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', warn_call_result_21027)
        
        # ################# End of 'warn(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'warn' in the type store
        # Getting the type of 'stypy_return_type' (line 61)
        stypy_return_type_21028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21028)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'warn'
        return stypy_return_type_21028


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        check.run.__dict__.__setitem__('stypy_localization', localization)
        check.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        check.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        check.run.__dict__.__setitem__('stypy_function_name', 'check.run')
        check.run.__dict__.__setitem__('stypy_param_names_list', [])
        check.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        check.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        check.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        check.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        check.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        check.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'check.run', [], None, None, defaults, varargs, kwargs)

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

        str_21029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 8), 'str', 'Runs the command.')
        
        # Getting the type of 'self' (line 69)
        self_21030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 11), 'self')
        # Obtaining the member 'metadata' of a type (line 69)
        metadata_21031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 11), self_21030, 'metadata')
        # Testing the type of an if condition (line 69)
        if_condition_21032 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 69, 8), metadata_21031)
        # Assigning a type to the variable 'if_condition_21032' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'if_condition_21032', if_condition_21032)
        # SSA begins for if statement (line 69)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_metadata(...): (line 70)
        # Processing the call keyword arguments (line 70)
        kwargs_21035 = {}
        # Getting the type of 'self' (line 70)
        self_21033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'self', False)
        # Obtaining the member 'check_metadata' of a type (line 70)
        check_metadata_21034 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 12), self_21033, 'check_metadata')
        # Calling check_metadata(args, kwargs) (line 70)
        check_metadata_call_result_21036 = invoke(stypy.reporting.localization.Localization(__file__, 70, 12), check_metadata_21034, *[], **kwargs_21035)
        
        # SSA join for if statement (line 69)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 71)
        self_21037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 11), 'self')
        # Obtaining the member 'restructuredtext' of a type (line 71)
        restructuredtext_21038 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 11), self_21037, 'restructuredtext')
        # Testing the type of an if condition (line 71)
        if_condition_21039 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 8), restructuredtext_21038)
        # Assigning a type to the variable 'if_condition_21039' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 8), 'if_condition_21039', if_condition_21039)
        # SSA begins for if statement (line 71)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'HAS_DOCUTILS' (line 72)
        HAS_DOCUTILS_21040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 15), 'HAS_DOCUTILS')
        # Testing the type of an if condition (line 72)
        if_condition_21041 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 12), HAS_DOCUTILS_21040)
        # Assigning a type to the variable 'if_condition_21041' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'if_condition_21041', if_condition_21041)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to check_restructuredtext(...): (line 73)
        # Processing the call keyword arguments (line 73)
        kwargs_21044 = {}
        # Getting the type of 'self' (line 73)
        self_21042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 16), 'self', False)
        # Obtaining the member 'check_restructuredtext' of a type (line 73)
        check_restructuredtext_21043 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 16), self_21042, 'check_restructuredtext')
        # Calling check_restructuredtext(args, kwargs) (line 73)
        check_restructuredtext_call_result_21045 = invoke(stypy.reporting.localization.Localization(__file__, 73, 16), check_restructuredtext_21043, *[], **kwargs_21044)
        
        # SSA branch for the else part of an if statement (line 72)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'self' (line 74)
        self_21046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'self')
        # Obtaining the member 'strict' of a type (line 74)
        strict_21047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 74, 17), self_21046, 'strict')
        # Testing the type of an if condition (line 74)
        if_condition_21048 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 17), strict_21047)
        # Assigning a type to the variable 'if_condition_21048' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 17), 'if_condition_21048', if_condition_21048)
        # SSA begins for if statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 75)
        # Processing the call arguments (line 75)
        str_21050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 42), 'str', 'The docutils package is needed.')
        # Processing the call keyword arguments (line 75)
        kwargs_21051 = {}
        # Getting the type of 'DistutilsSetupError' (line 75)
        DistutilsSetupError_21049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 22), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 75)
        DistutilsSetupError_call_result_21052 = invoke(stypy.reporting.localization.Localization(__file__, 75, 22), DistutilsSetupError_21049, *[str_21050], **kwargs_21051)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 75, 16), DistutilsSetupError_call_result_21052, 'raise parameter', BaseException)
        # SSA join for if statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 71)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 79)
        self_21053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 11), 'self')
        # Obtaining the member 'strict' of a type (line 79)
        strict_21054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 11), self_21053, 'strict')
        
        # Getting the type of 'self' (line 79)
        self_21055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 27), 'self')
        # Obtaining the member '_warnings' of a type (line 79)
        _warnings_21056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 27), self_21055, '_warnings')
        int_21057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 44), 'int')
        # Applying the binary operator '>' (line 79)
        result_gt_21058 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 27), '>', _warnings_21056, int_21057)
        
        # Applying the binary operator 'and' (line 79)
        result_and_keyword_21059 = python_operator(stypy.reporting.localization.Localization(__file__, 79, 11), 'and', strict_21054, result_gt_21058)
        
        # Testing the type of an if condition (line 79)
        if_condition_21060 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 79, 8), result_and_keyword_21059)
        # Assigning a type to the variable 'if_condition_21060' (line 79)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 8), 'if_condition_21060', if_condition_21060)
        # SSA begins for if statement (line 79)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to DistutilsSetupError(...): (line 80)
        # Processing the call arguments (line 80)
        str_21062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 38), 'str', 'Please correct your package.')
        # Processing the call keyword arguments (line 80)
        kwargs_21063 = {}
        # Getting the type of 'DistutilsSetupError' (line 80)
        DistutilsSetupError_21061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 18), 'DistutilsSetupError', False)
        # Calling DistutilsSetupError(args, kwargs) (line 80)
        DistutilsSetupError_call_result_21064 = invoke(stypy.reporting.localization.Localization(__file__, 80, 18), DistutilsSetupError_21061, *[str_21062], **kwargs_21063)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 80, 12), DistutilsSetupError_call_result_21064, 'raise parameter', BaseException)
        # SSA join for if statement (line 79)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 66)
        stypy_return_type_21065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21065)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_21065


    @norecursion
    def check_metadata(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_metadata'
        module_type_store = module_type_store.open_function_context('check_metadata', 82, 4, False)
        # Assigning a type to the variable 'self' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        check.check_metadata.__dict__.__setitem__('stypy_localization', localization)
        check.check_metadata.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        check.check_metadata.__dict__.__setitem__('stypy_type_store', module_type_store)
        check.check_metadata.__dict__.__setitem__('stypy_function_name', 'check.check_metadata')
        check.check_metadata.__dict__.__setitem__('stypy_param_names_list', [])
        check.check_metadata.__dict__.__setitem__('stypy_varargs_param_name', None)
        check.check_metadata.__dict__.__setitem__('stypy_kwargs_param_name', None)
        check.check_metadata.__dict__.__setitem__('stypy_call_defaults', defaults)
        check.check_metadata.__dict__.__setitem__('stypy_call_varargs', varargs)
        check.check_metadata.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        check.check_metadata.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'check.check_metadata', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_metadata', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_metadata(...)' code ##################

        str_21066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, (-1)), 'str', 'Ensures that all required elements of meta-data are supplied.\n\n        name, version, URL, (author and author_email) or\n        (maintainer and maintainer_email)).\n\n        Warns if any are missing.\n        ')
        
        # Assigning a Attribute to a Name (line 90):
        # Getting the type of 'self' (line 90)
        self_21067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 19), 'self')
        # Obtaining the member 'distribution' of a type (line 90)
        distribution_21068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), self_21067, 'distribution')
        # Obtaining the member 'metadata' of a type (line 90)
        metadata_21069 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 19), distribution_21068, 'metadata')
        # Assigning a type to the variable 'metadata' (line 90)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'metadata', metadata_21069)
        
        # Assigning a List to a Name (line 92):
        
        # Obtaining an instance of the builtin type 'list' (line 92)
        list_21070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 18), 'list')
        # Adding type elements to the builtin type 'list' instance (line 92)
        
        # Assigning a type to the variable 'missing' (line 92)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 92, 8), 'missing', list_21070)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 93)
        tuple_21071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 93)
        # Adding element type (line 93)
        str_21072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 21), 'str', 'name')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), tuple_21071, str_21072)
        # Adding element type (line 93)
        str_21073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 29), 'str', 'version')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), tuple_21071, str_21073)
        # Adding element type (line 93)
        str_21074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 40), 'str', 'url')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 21), tuple_21071, str_21074)
        
        # Testing the type of a for loop iterable (line 93)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 93, 8), tuple_21071)
        # Getting the type of the for loop variable (line 93)
        for_loop_var_21075 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 93, 8), tuple_21071)
        # Assigning a type to the variable 'attr' (line 93)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 93, 8), 'attr', for_loop_var_21075)
        # SSA begins for a for statement (line 93)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'metadata' (line 94)
        metadata_21077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 28), 'metadata', False)
        # Getting the type of 'attr' (line 94)
        attr_21078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 38), 'attr', False)
        # Processing the call keyword arguments (line 94)
        kwargs_21079 = {}
        # Getting the type of 'hasattr' (line 94)
        hasattr_21076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 20), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 94)
        hasattr_call_result_21080 = invoke(stypy.reporting.localization.Localization(__file__, 94, 20), hasattr_21076, *[metadata_21077, attr_21078], **kwargs_21079)
        
        
        # Call to getattr(...): (line 94)
        # Processing the call arguments (line 94)
        # Getting the type of 'metadata' (line 94)
        metadata_21082 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 56), 'metadata', False)
        # Getting the type of 'attr' (line 94)
        attr_21083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 66), 'attr', False)
        # Processing the call keyword arguments (line 94)
        kwargs_21084 = {}
        # Getting the type of 'getattr' (line 94)
        getattr_21081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 94, 48), 'getattr', False)
        # Calling getattr(args, kwargs) (line 94)
        getattr_call_result_21085 = invoke(stypy.reporting.localization.Localization(__file__, 94, 48), getattr_21081, *[metadata_21082, attr_21083], **kwargs_21084)
        
        # Applying the binary operator 'and' (line 94)
        result_and_keyword_21086 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 20), 'and', hasattr_call_result_21080, getattr_call_result_21085)
        
        # Applying the 'not' unary operator (line 94)
        result_not__21087 = python_operator(stypy.reporting.localization.Localization(__file__, 94, 15), 'not', result_and_keyword_21086)
        
        # Testing the type of an if condition (line 94)
        if_condition_21088 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 94, 12), result_not__21087)
        # Assigning a type to the variable 'if_condition_21088' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 12), 'if_condition_21088', if_condition_21088)
        # SSA begins for if statement (line 94)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'attr' (line 95)
        attr_21091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 31), 'attr', False)
        # Processing the call keyword arguments (line 95)
        kwargs_21092 = {}
        # Getting the type of 'missing' (line 95)
        missing_21089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 16), 'missing', False)
        # Obtaining the member 'append' of a type (line 95)
        append_21090 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 16), missing_21089, 'append')
        # Calling append(args, kwargs) (line 95)
        append_call_result_21093 = invoke(stypy.reporting.localization.Localization(__file__, 95, 16), append_21090, *[attr_21091], **kwargs_21092)
        
        # SSA join for if statement (line 94)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'missing' (line 97)
        missing_21094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 11), 'missing')
        # Testing the type of an if condition (line 97)
        if_condition_21095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 97, 8), missing_21094)
        # Assigning a type to the variable 'if_condition_21095' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'if_condition_21095', if_condition_21095)
        # SSA begins for if statement (line 97)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 98)
        # Processing the call arguments (line 98)
        str_21098 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 22), 'str', 'missing required meta-data: %s')
        
        # Call to join(...): (line 98)
        # Processing the call arguments (line 98)
        # Getting the type of 'missing' (line 98)
        missing_21101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 68), 'missing', False)
        # Processing the call keyword arguments (line 98)
        kwargs_21102 = {}
        str_21099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, 58), 'str', ', ')
        # Obtaining the member 'join' of a type (line 98)
        join_21100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 58), str_21099, 'join')
        # Calling join(args, kwargs) (line 98)
        join_call_result_21103 = invoke(stypy.reporting.localization.Localization(__file__, 98, 58), join_21100, *[missing_21101], **kwargs_21102)
        
        # Applying the binary operator '%' (line 98)
        result_mod_21104 = python_operator(stypy.reporting.localization.Localization(__file__, 98, 22), '%', str_21098, join_call_result_21103)
        
        # Processing the call keyword arguments (line 98)
        kwargs_21105 = {}
        # Getting the type of 'self' (line 98)
        self_21096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 98)
        warn_21097 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 98, 12), self_21096, 'warn')
        # Calling warn(args, kwargs) (line 98)
        warn_call_result_21106 = invoke(stypy.reporting.localization.Localization(__file__, 98, 12), warn_21097, *[result_mod_21104], **kwargs_21105)
        
        # SSA join for if statement (line 97)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'metadata' (line 99)
        metadata_21107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 11), 'metadata')
        # Obtaining the member 'author' of a type (line 99)
        author_21108 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 11), metadata_21107, 'author')
        # Testing the type of an if condition (line 99)
        if_condition_21109 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 99, 8), author_21108)
        # Assigning a type to the variable 'if_condition_21109' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'if_condition_21109', if_condition_21109)
        # SSA begins for if statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'metadata' (line 100)
        metadata_21110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 19), 'metadata')
        # Obtaining the member 'author_email' of a type (line 100)
        author_email_21111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 19), metadata_21110, 'author_email')
        # Applying the 'not' unary operator (line 100)
        result_not__21112 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'not', author_email_21111)
        
        # Testing the type of an if condition (line 100)
        if_condition_21113 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_not__21112)
        # Assigning a type to the variable 'if_condition_21113' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_21113', if_condition_21113)
        # SSA begins for if statement (line 100)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 101)
        # Processing the call arguments (line 101)
        str_21116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 26), 'str', "missing meta-data: if 'author' supplied, ")
        str_21117 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 26), 'str', "'author_email' must be supplied too")
        # Applying the binary operator '+' (line 101)
        result_add_21118 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 26), '+', str_21116, str_21117)
        
        # Processing the call keyword arguments (line 101)
        kwargs_21119 = {}
        # Getting the type of 'self' (line 101)
        self_21114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 16), 'self', False)
        # Obtaining the member 'warn' of a type (line 101)
        warn_21115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 16), self_21114, 'warn')
        # Calling warn(args, kwargs) (line 101)
        warn_call_result_21120 = invoke(stypy.reporting.localization.Localization(__file__, 101, 16), warn_21115, *[result_add_21118], **kwargs_21119)
        
        # SSA join for if statement (line 100)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 99)
        module_type_store.open_ssa_branch('else')
        
        # Getting the type of 'metadata' (line 103)
        metadata_21121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'metadata')
        # Obtaining the member 'maintainer' of a type (line 103)
        maintainer_21122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 13), metadata_21121, 'maintainer')
        # Testing the type of an if condition (line 103)
        if_condition_21123 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 13), maintainer_21122)
        # Assigning a type to the variable 'if_condition_21123' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 13), 'if_condition_21123', if_condition_21123)
        # SSA begins for if statement (line 103)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Getting the type of 'metadata' (line 104)
        metadata_21124 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 19), 'metadata')
        # Obtaining the member 'maintainer_email' of a type (line 104)
        maintainer_email_21125 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 19), metadata_21124, 'maintainer_email')
        # Applying the 'not' unary operator (line 104)
        result_not__21126 = python_operator(stypy.reporting.localization.Localization(__file__, 104, 15), 'not', maintainer_email_21125)
        
        # Testing the type of an if condition (line 104)
        if_condition_21127 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 104, 12), result_not__21126)
        # Assigning a type to the variable 'if_condition_21127' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'if_condition_21127', if_condition_21127)
        # SSA begins for if statement (line 104)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 105)
        # Processing the call arguments (line 105)
        str_21130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 26), 'str', "missing meta-data: if 'maintainer' supplied, ")
        str_21131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 26), 'str', "'maintainer_email' must be supplied too")
        # Applying the binary operator '+' (line 105)
        result_add_21132 = python_operator(stypy.reporting.localization.Localization(__file__, 105, 26), '+', str_21130, str_21131)
        
        # Processing the call keyword arguments (line 105)
        kwargs_21133 = {}
        # Getting the type of 'self' (line 105)
        self_21128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 16), 'self', False)
        # Obtaining the member 'warn' of a type (line 105)
        warn_21129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 16), self_21128, 'warn')
        # Calling warn(args, kwargs) (line 105)
        warn_call_result_21134 = invoke(stypy.reporting.localization.Localization(__file__, 105, 16), warn_21129, *[result_add_21132], **kwargs_21133)
        
        # SSA join for if statement (line 104)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA branch for the else part of an if statement (line 103)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 108)
        # Processing the call arguments (line 108)
        str_21137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 108, 22), 'str', 'missing meta-data: either (author and author_email) ')
        str_21138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 22), 'str', 'or (maintainer and maintainer_email) ')
        # Applying the binary operator '+' (line 108)
        result_add_21139 = python_operator(stypy.reporting.localization.Localization(__file__, 108, 22), '+', str_21137, str_21138)
        
        str_21140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 110, 22), 'str', 'must be supplied')
        # Applying the binary operator '+' (line 109)
        result_add_21141 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 62), '+', result_add_21139, str_21140)
        
        # Processing the call keyword arguments (line 108)
        kwargs_21142 = {}
        # Getting the type of 'self' (line 108)
        self_21135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 108)
        warn_21136 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 108, 12), self_21135, 'warn')
        # Calling warn(args, kwargs) (line 108)
        warn_call_result_21143 = invoke(stypy.reporting.localization.Localization(__file__, 108, 12), warn_21136, *[result_add_21141], **kwargs_21142)
        
        # SSA join for if statement (line 103)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_metadata(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_metadata' in the type store
        # Getting the type of 'stypy_return_type' (line 82)
        stypy_return_type_21144 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21144)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_metadata'
        return stypy_return_type_21144


    @norecursion
    def check_restructuredtext(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'check_restructuredtext'
        module_type_store = module_type_store.open_function_context('check_restructuredtext', 112, 4, False)
        # Assigning a type to the variable 'self' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        check.check_restructuredtext.__dict__.__setitem__('stypy_localization', localization)
        check.check_restructuredtext.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        check.check_restructuredtext.__dict__.__setitem__('stypy_type_store', module_type_store)
        check.check_restructuredtext.__dict__.__setitem__('stypy_function_name', 'check.check_restructuredtext')
        check.check_restructuredtext.__dict__.__setitem__('stypy_param_names_list', [])
        check.check_restructuredtext.__dict__.__setitem__('stypy_varargs_param_name', None)
        check.check_restructuredtext.__dict__.__setitem__('stypy_kwargs_param_name', None)
        check.check_restructuredtext.__dict__.__setitem__('stypy_call_defaults', defaults)
        check.check_restructuredtext.__dict__.__setitem__('stypy_call_varargs', varargs)
        check.check_restructuredtext.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        check.check_restructuredtext.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'check.check_restructuredtext', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'check_restructuredtext', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'check_restructuredtext(...)' code ##################

        str_21145 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 8), 'str', 'Checks if the long string fields are reST-compliant.')
        
        # Assigning a Call to a Name (line 114):
        
        # Call to get_long_description(...): (line 114)
        # Processing the call keyword arguments (line 114)
        kwargs_21149 = {}
        # Getting the type of 'self' (line 114)
        self_21146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 15), 'self', False)
        # Obtaining the member 'distribution' of a type (line 114)
        distribution_21147 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 15), self_21146, 'distribution')
        # Obtaining the member 'get_long_description' of a type (line 114)
        get_long_description_21148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 15), distribution_21147, 'get_long_description')
        # Calling get_long_description(args, kwargs) (line 114)
        get_long_description_call_result_21150 = invoke(stypy.reporting.localization.Localization(__file__, 114, 15), get_long_description_21148, *[], **kwargs_21149)
        
        # Assigning a type to the variable 'data' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 8), 'data', get_long_description_call_result_21150)
        
        # Type idiom detected: calculating its left and rigth part (line 115)
        # Getting the type of 'unicode' (line 115)
        unicode_21151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'unicode')
        # Getting the type of 'data' (line 115)
        data_21152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 26), 'data')
        
        (may_be_21153, more_types_in_union_21154) = may_not_be_subtype(unicode_21151, data_21152)

        if may_be_21153:

            if more_types_in_union_21154:
                # Runtime conditional SSA (line 115)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            # Assigning a type to the variable 'data' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 8), 'data', remove_subtype_from_union(data_21152, unicode))
            
            # Assigning a Call to a Name (line 116):
            
            # Call to decode(...): (line 116)
            # Processing the call arguments (line 116)
            # Getting the type of 'PKG_INFO_ENCODING' (line 116)
            PKG_INFO_ENCODING_21157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 31), 'PKG_INFO_ENCODING', False)
            # Processing the call keyword arguments (line 116)
            kwargs_21158 = {}
            # Getting the type of 'data' (line 116)
            data_21155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 19), 'data', False)
            # Obtaining the member 'decode' of a type (line 116)
            decode_21156 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 19), data_21155, 'decode')
            # Calling decode(args, kwargs) (line 116)
            decode_call_result_21159 = invoke(stypy.reporting.localization.Localization(__file__, 116, 19), decode_21156, *[PKG_INFO_ENCODING_21157], **kwargs_21158)
            
            # Assigning a type to the variable 'data' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'data', decode_call_result_21159)

            if more_types_in_union_21154:
                # SSA join for if statement (line 115)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Call to _check_rst_data(...): (line 117)
        # Processing the call arguments (line 117)
        # Getting the type of 'data' (line 117)
        data_21162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 44), 'data', False)
        # Processing the call keyword arguments (line 117)
        kwargs_21163 = {}
        # Getting the type of 'self' (line 117)
        self_21160 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 23), 'self', False)
        # Obtaining the member '_check_rst_data' of a type (line 117)
        _check_rst_data_21161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 117, 23), self_21160, '_check_rst_data')
        # Calling _check_rst_data(args, kwargs) (line 117)
        _check_rst_data_call_result_21164 = invoke(stypy.reporting.localization.Localization(__file__, 117, 23), _check_rst_data_21161, *[data_21162], **kwargs_21163)
        
        # Testing the type of a for loop iterable (line 117)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 117, 8), _check_rst_data_call_result_21164)
        # Getting the type of the for loop variable (line 117)
        for_loop_var_21165 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 117, 8), _check_rst_data_call_result_21164)
        # Assigning a type to the variable 'warning' (line 117)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 8), 'warning', for_loop_var_21165)
        # SSA begins for a for statement (line 117)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 118):
        
        # Call to get(...): (line 118)
        # Processing the call arguments (line 118)
        str_21171 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 35), 'str', 'line')
        # Processing the call keyword arguments (line 118)
        kwargs_21172 = {}
        
        # Obtaining the type of the subscript
        int_21166 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 118, 27), 'int')
        # Getting the type of 'warning' (line 118)
        warning_21167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 19), 'warning', False)
        # Obtaining the member '__getitem__' of a type (line 118)
        getitem___21168 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), warning_21167, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 118)
        subscript_call_result_21169 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), getitem___21168, int_21166)
        
        # Obtaining the member 'get' of a type (line 118)
        get_21170 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 19), subscript_call_result_21169, 'get')
        # Calling get(args, kwargs) (line 118)
        get_call_result_21173 = invoke(stypy.reporting.localization.Localization(__file__, 118, 19), get_21170, *[str_21171], **kwargs_21172)
        
        # Assigning a type to the variable 'line' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'line', get_call_result_21173)
        
        # Type idiom detected: calculating its left and rigth part (line 119)
        # Getting the type of 'line' (line 119)
        line_21174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 15), 'line')
        # Getting the type of 'None' (line 119)
        None_21175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 23), 'None')
        
        (may_be_21176, more_types_in_union_21177) = may_be_none(line_21174, None_21175)

        if may_be_21176:

            if more_types_in_union_21177:
                # Runtime conditional SSA (line 119)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Subscript to a Name (line 120):
            
            # Obtaining the type of the subscript
            int_21178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 120, 34), 'int')
            # Getting the type of 'warning' (line 120)
            warning_21179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 26), 'warning')
            # Obtaining the member '__getitem__' of a type (line 120)
            getitem___21180 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 26), warning_21179, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 120)
            subscript_call_result_21181 = invoke(stypy.reporting.localization.Localization(__file__, 120, 26), getitem___21180, int_21178)
            
            # Assigning a type to the variable 'warning' (line 120)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'warning', subscript_call_result_21181)

            if more_types_in_union_21177:
                # Runtime conditional SSA for else branch (line 119)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_21176) or more_types_in_union_21177):
            
            # Assigning a BinOp to a Name (line 122):
            str_21182 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 26), 'str', '%s (line %s)')
            
            # Obtaining an instance of the builtin type 'tuple' (line 122)
            tuple_21183 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 44), 'tuple')
            # Adding type elements to the builtin type 'tuple' instance (line 122)
            # Adding element type (line 122)
            
            # Obtaining the type of the subscript
            int_21184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 52), 'int')
            # Getting the type of 'warning' (line 122)
            warning_21185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 44), 'warning')
            # Obtaining the member '__getitem__' of a type (line 122)
            getitem___21186 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 44), warning_21185, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 122)
            subscript_call_result_21187 = invoke(stypy.reporting.localization.Localization(__file__, 122, 44), getitem___21186, int_21184)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 44), tuple_21183, subscript_call_result_21187)
            # Adding element type (line 122)
            # Getting the type of 'line' (line 122)
            line_21188 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 56), 'line')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 122, 44), tuple_21183, line_21188)
            
            # Applying the binary operator '%' (line 122)
            result_mod_21189 = python_operator(stypy.reporting.localization.Localization(__file__, 122, 26), '%', str_21182, tuple_21183)
            
            # Assigning a type to the variable 'warning' (line 122)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 16), 'warning', result_mod_21189)

            if (may_be_21176 and more_types_in_union_21177):
                # SSA join for if statement (line 119)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to warn(...): (line 123)
        # Processing the call arguments (line 123)
        # Getting the type of 'warning' (line 123)
        warning_21192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 22), 'warning', False)
        # Processing the call keyword arguments (line 123)
        kwargs_21193 = {}
        # Getting the type of 'self' (line 123)
        self_21190 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 12), 'self', False)
        # Obtaining the member 'warn' of a type (line 123)
        warn_21191 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 12), self_21190, 'warn')
        # Calling warn(args, kwargs) (line 123)
        warn_call_result_21194 = invoke(stypy.reporting.localization.Localization(__file__, 123, 12), warn_21191, *[warning_21192], **kwargs_21193)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'check_restructuredtext(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'check_restructuredtext' in the type store
        # Getting the type of 'stypy_return_type' (line 112)
        stypy_return_type_21195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21195)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'check_restructuredtext'
        return stypy_return_type_21195


    @norecursion
    def _check_rst_data(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '_check_rst_data'
        module_type_store = module_type_store.open_function_context('_check_rst_data', 125, 4, False)
        # Assigning a type to the variable 'self' (line 126)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        check._check_rst_data.__dict__.__setitem__('stypy_localization', localization)
        check._check_rst_data.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        check._check_rst_data.__dict__.__setitem__('stypy_type_store', module_type_store)
        check._check_rst_data.__dict__.__setitem__('stypy_function_name', 'check._check_rst_data')
        check._check_rst_data.__dict__.__setitem__('stypy_param_names_list', ['data'])
        check._check_rst_data.__dict__.__setitem__('stypy_varargs_param_name', None)
        check._check_rst_data.__dict__.__setitem__('stypy_kwargs_param_name', None)
        check._check_rst_data.__dict__.__setitem__('stypy_call_defaults', defaults)
        check._check_rst_data.__dict__.__setitem__('stypy_call_varargs', varargs)
        check._check_rst_data.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        check._check_rst_data.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'check._check_rst_data', ['data'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '_check_rst_data', localization, ['data'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '_check_rst_data(...)' code ##################

        str_21196 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 8), 'str', "Returns warnings when the provided data doesn't compile.")
        
        # Assigning a Call to a Name (line 127):
        
        # Call to StringIO(...): (line 127)
        # Processing the call keyword arguments (line 127)
        kwargs_21198 = {}
        # Getting the type of 'StringIO' (line 127)
        StringIO_21197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 22), 'StringIO', False)
        # Calling StringIO(args, kwargs) (line 127)
        StringIO_call_result_21199 = invoke(stypy.reporting.localization.Localization(__file__, 127, 22), StringIO_21197, *[], **kwargs_21198)
        
        # Assigning a type to the variable 'source_path' (line 127)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 8), 'source_path', StringIO_call_result_21199)
        
        # Assigning a Call to a Name (line 128):
        
        # Call to Parser(...): (line 128)
        # Processing the call keyword arguments (line 128)
        kwargs_21201 = {}
        # Getting the type of 'Parser' (line 128)
        Parser_21200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 17), 'Parser', False)
        # Calling Parser(args, kwargs) (line 128)
        Parser_call_result_21202 = invoke(stypy.reporting.localization.Localization(__file__, 128, 17), Parser_21200, *[], **kwargs_21201)
        
        # Assigning a type to the variable 'parser' (line 128)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 128, 8), 'parser', Parser_call_result_21202)
        
        # Assigning a Call to a Name (line 129):
        
        # Call to get_default_values(...): (line 129)
        # Processing the call keyword arguments (line 129)
        kwargs_21211 = {}
        
        # Call to OptionParser(...): (line 129)
        # Processing the call keyword arguments (line 129)
        
        # Obtaining an instance of the builtin type 'tuple' (line 129)
        tuple_21205 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 129, 53), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 129)
        # Adding element type (line 129)
        # Getting the type of 'Parser' (line 129)
        Parser_21206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 53), 'Parser', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 53), tuple_21205, Parser_21206)
        
        keyword_21207 = tuple_21205
        kwargs_21208 = {'components': keyword_21207}
        # Getting the type of 'frontend' (line 129)
        frontend_21203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 19), 'frontend', False)
        # Obtaining the member 'OptionParser' of a type (line 129)
        OptionParser_21204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 19), frontend_21203, 'OptionParser')
        # Calling OptionParser(args, kwargs) (line 129)
        OptionParser_call_result_21209 = invoke(stypy.reporting.localization.Localization(__file__, 129, 19), OptionParser_21204, *[], **kwargs_21208)
        
        # Obtaining the member 'get_default_values' of a type (line 129)
        get_default_values_21210 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 19), OptionParser_call_result_21209, 'get_default_values')
        # Calling get_default_values(args, kwargs) (line 129)
        get_default_values_call_result_21212 = invoke(stypy.reporting.localization.Localization(__file__, 129, 19), get_default_values_21210, *[], **kwargs_21211)
        
        # Assigning a type to the variable 'settings' (line 129)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 129, 8), 'settings', get_default_values_call_result_21212)
        
        # Assigning a Num to a Attribute (line 130):
        int_21213 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 130, 29), 'int')
        # Getting the type of 'settings' (line 130)
        settings_21214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 130, 8), 'settings')
        # Setting the type of the member 'tab_width' of a type (line 130)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 130, 8), settings_21214, 'tab_width', int_21213)
        
        # Assigning a Name to a Attribute (line 131):
        # Getting the type of 'None' (line 131)
        None_21215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 34), 'None')
        # Getting the type of 'settings' (line 131)
        settings_21216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'settings')
        # Setting the type of the member 'pep_references' of a type (line 131)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 8), settings_21216, 'pep_references', None_21215)
        
        # Assigning a Name to a Attribute (line 132):
        # Getting the type of 'None' (line 132)
        None_21217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 34), 'None')
        # Getting the type of 'settings' (line 132)
        settings_21218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 8), 'settings')
        # Setting the type of the member 'rfc_references' of a type (line 132)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 8), settings_21218, 'rfc_references', None_21217)
        
        # Assigning a Call to a Name (line 133):
        
        # Call to SilentReporter(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'source_path' (line 133)
        source_path_21220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 34), 'source_path', False)
        # Getting the type of 'settings' (line 134)
        settings_21221 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 26), 'settings', False)
        # Obtaining the member 'report_level' of a type (line 134)
        report_level_21222 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 26), settings_21221, 'report_level')
        # Getting the type of 'settings' (line 135)
        settings_21223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 26), 'settings', False)
        # Obtaining the member 'halt_level' of a type (line 135)
        halt_level_21224 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 26), settings_21223, 'halt_level')
        # Processing the call keyword arguments (line 133)
        # Getting the type of 'settings' (line 136)
        settings_21225 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 136, 33), 'settings', False)
        # Obtaining the member 'warning_stream' of a type (line 136)
        warning_stream_21226 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 136, 33), settings_21225, 'warning_stream')
        keyword_21227 = warning_stream_21226
        # Getting the type of 'settings' (line 137)
        settings_21228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 137, 32), 'settings', False)
        # Obtaining the member 'debug' of a type (line 137)
        debug_21229 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 137, 32), settings_21228, 'debug')
        keyword_21230 = debug_21229
        # Getting the type of 'settings' (line 138)
        settings_21231 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 35), 'settings', False)
        # Obtaining the member 'error_encoding' of a type (line 138)
        error_encoding_21232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 138, 35), settings_21231, 'error_encoding')
        keyword_21233 = error_encoding_21232
        # Getting the type of 'settings' (line 139)
        settings_21234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 40), 'settings', False)
        # Obtaining the member 'error_encoding_error_handler' of a type (line 139)
        error_encoding_error_handler_21235 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 40), settings_21234, 'error_encoding_error_handler')
        keyword_21236 = error_encoding_error_handler_21235
        kwargs_21237 = {'debug': keyword_21230, 'error_handler': keyword_21236, 'stream': keyword_21227, 'encoding': keyword_21233}
        # Getting the type of 'SilentReporter' (line 133)
        SilentReporter_21219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 19), 'SilentReporter', False)
        # Calling SilentReporter(args, kwargs) (line 133)
        SilentReporter_call_result_21238 = invoke(stypy.reporting.localization.Localization(__file__, 133, 19), SilentReporter_21219, *[source_path_21220, report_level_21222, halt_level_21224], **kwargs_21237)
        
        # Assigning a type to the variable 'reporter' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 8), 'reporter', SilentReporter_call_result_21238)
        
        # Assigning a Call to a Name (line 141):
        
        # Call to document(...): (line 141)
        # Processing the call arguments (line 141)
        # Getting the type of 'settings' (line 141)
        settings_21241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 34), 'settings', False)
        # Getting the type of 'reporter' (line 141)
        reporter_21242 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 44), 'reporter', False)
        # Processing the call keyword arguments (line 141)
        # Getting the type of 'source_path' (line 141)
        source_path_21243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 61), 'source_path', False)
        keyword_21244 = source_path_21243
        kwargs_21245 = {'source': keyword_21244}
        # Getting the type of 'nodes' (line 141)
        nodes_21239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 141, 19), 'nodes', False)
        # Obtaining the member 'document' of a type (line 141)
        document_21240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 141, 19), nodes_21239, 'document')
        # Calling document(args, kwargs) (line 141)
        document_call_result_21246 = invoke(stypy.reporting.localization.Localization(__file__, 141, 19), document_21240, *[settings_21241, reporter_21242], **kwargs_21245)
        
        # Assigning a type to the variable 'document' (line 141)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 141, 8), 'document', document_call_result_21246)
        
        # Call to note_source(...): (line 142)
        # Processing the call arguments (line 142)
        # Getting the type of 'source_path' (line 142)
        source_path_21249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 29), 'source_path', False)
        int_21250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 142, 42), 'int')
        # Processing the call keyword arguments (line 142)
        kwargs_21251 = {}
        # Getting the type of 'document' (line 142)
        document_21247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 142, 8), 'document', False)
        # Obtaining the member 'note_source' of a type (line 142)
        note_source_21248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 142, 8), document_21247, 'note_source')
        # Calling note_source(args, kwargs) (line 142)
        note_source_call_result_21252 = invoke(stypy.reporting.localization.Localization(__file__, 142, 8), note_source_21248, *[source_path_21249, int_21250], **kwargs_21251)
        
        
        
        # SSA begins for try-except statement (line 143)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to parse(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'data' (line 144)
        data_21255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 25), 'data', False)
        # Getting the type of 'document' (line 144)
        document_21256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 31), 'document', False)
        # Processing the call keyword arguments (line 144)
        kwargs_21257 = {}
        # Getting the type of 'parser' (line 144)
        parser_21253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'parser', False)
        # Obtaining the member 'parse' of a type (line 144)
        parse_21254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 12), parser_21253, 'parse')
        # Calling parse(args, kwargs) (line 144)
        parse_call_result_21258 = invoke(stypy.reporting.localization.Localization(__file__, 144, 12), parse_21254, *[data_21255, document_21256], **kwargs_21257)
        
        # SSA branch for the except part of a try statement (line 143)
        # SSA branch for the except 'AttributeError' branch of a try statement (line 143)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'AttributeError' (line 145)
        AttributeError_21259 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 15), 'AttributeError')
        # Assigning a type to the variable 'e' (line 145)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 8), 'e', AttributeError_21259)
        
        # Call to append(...): (line 146)
        # Processing the call arguments (line 146)
        
        # Obtaining an instance of the builtin type 'tuple' (line 147)
        tuple_21263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 17), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 147)
        # Adding element type (line 147)
        int_21264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 17), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 17), tuple_21263, int_21264)
        # Adding element type (line 147)
        str_21265 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 21), 'str', 'Could not finish the parsing: %s.')
        # Getting the type of 'e' (line 147)
        e_21266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 59), 'e', False)
        # Applying the binary operator '%' (line 147)
        result_mod_21267 = python_operator(stypy.reporting.localization.Localization(__file__, 147, 21), '%', str_21265, e_21266)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 17), tuple_21263, result_mod_21267)
        # Adding element type (line 147)
        str_21268 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 62), 'str', '')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 17), tuple_21263, str_21268)
        # Adding element type (line 147)
        
        # Obtaining an instance of the builtin type 'dict' (line 147)
        dict_21269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 147, 66), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 147)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 147, 17), tuple_21263, dict_21269)
        
        # Processing the call keyword arguments (line 146)
        kwargs_21270 = {}
        # Getting the type of 'reporter' (line 146)
        reporter_21260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'reporter', False)
        # Obtaining the member 'messages' of a type (line 146)
        messages_21261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), reporter_21260, 'messages')
        # Obtaining the member 'append' of a type (line 146)
        append_21262 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), messages_21261, 'append')
        # Calling append(args, kwargs) (line 146)
        append_call_result_21271 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), append_21262, *[tuple_21263], **kwargs_21270)
        
        # SSA join for try-except statement (line 143)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'reporter' (line 149)
        reporter_21272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 15), 'reporter')
        # Obtaining the member 'messages' of a type (line 149)
        messages_21273 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 15), reporter_21272, 'messages')
        # Assigning a type to the variable 'stypy_return_type' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'stypy_return_type', messages_21273)
        
        # ################# End of '_check_rst_data(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '_check_rst_data' in the type store
        # Getting the type of 'stypy_return_type' (line 125)
        stypy_return_type_21274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 125, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21274)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '_check_rst_data'
        return stypy_return_type_21274


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 0, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'check.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'check' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'check', check)

# Assigning a Str to a Name (line 41):
str_21275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 19), 'str', 'perform some checks on the package')
# Getting the type of 'check'
check_21276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'check')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), check_21276, 'description', str_21275)

# Assigning a List to a Name (line 42):

# Obtaining an instance of the builtin type 'list' (line 42)
list_21277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 42)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'tuple' (line 42)
tuple_21278 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 42)
# Adding element type (line 42)
str_21279 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'str', 'metadata')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), tuple_21278, str_21279)
# Adding element type (line 42)
str_21280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 33), 'str', 'm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), tuple_21278, str_21280)
# Adding element type (line 42)
str_21281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 38), 'str', 'Verify meta-data')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 21), tuple_21278, str_21281)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_21277, tuple_21278)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_21282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
str_21283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'str', 'restructuredtext')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_21282, str_21283)
# Adding element type (line 43)
str_21284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 41), 'str', 'r')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_21282, str_21284)
# Adding element type (line 43)
str_21285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'str', 'Checks if long string meta-data syntax are reStructuredText-compliant')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_21282, str_21285)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_21277, tuple_21282)
# Adding element type (line 42)

# Obtaining an instance of the builtin type 'tuple' (line 46)
tuple_21286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 46)
# Adding element type (line 46)
str_21287 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'str', 'strict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_21286, str_21287)
# Adding element type (line 46)
str_21288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 31), 'str', 's')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_21286, str_21288)
# Adding element type (line 46)
str_21289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'str', 'Will exit with an error if a check fails')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_21286, str_21289)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 42, 19), list_21277, tuple_21286)

# Getting the type of 'check'
check_21290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'check')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), check_21290, 'user_options', list_21277)

# Assigning a List to a Name (line 49):

# Obtaining an instance of the builtin type 'list' (line 49)
list_21291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 49)
# Adding element type (line 49)
str_21292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 23), 'str', 'metadata')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 22), list_21291, str_21292)
# Adding element type (line 49)
str_21293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 35), 'str', 'restructuredtext')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 22), list_21291, str_21293)
# Adding element type (line 49)
str_21294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 55), 'str', 'strict')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 22), list_21291, str_21294)

# Getting the type of 'check'
check_21295 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'check')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), check_21295, 'boolean_options', list_21291)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
