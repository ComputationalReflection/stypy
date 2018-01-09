
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.bdist
2: 
3: Implements the Distutils 'bdist' command (create a built [binary]
4: distribution).'''
5: 
6: __revision__ = "$Id$"
7: 
8: import os
9: 
10: from distutils.util import get_platform
11: from distutils.core import Command
12: from distutils.errors import DistutilsPlatformError, DistutilsOptionError
13: 
14: 
15: def show_formats():
16:     '''Print list of available formats (arguments to "--format" option).
17:     '''
18:     from distutils.fancy_getopt import FancyGetopt
19:     formats = []
20:     for format in bdist.format_commands:
21:         formats.append(("formats=" + format, None,
22:                         bdist.format_command[format][1]))
23:     pretty_printer = FancyGetopt(formats)
24:     pretty_printer.print_help("List of available distribution formats:")
25: 
26: 
27: class bdist(Command):
28: 
29:     description = "create a built (binary) distribution"
30: 
31:     user_options = [('bdist-base=', 'b',
32:                      "temporary directory for creating built distributions"),
33:                     ('plat-name=', 'p',
34:                      "platform name to embed in generated filenames "
35:                      "(default: %s)" % get_platform()),
36:                     ('formats=', None,
37:                      "formats for distribution (comma-separated list)"),
38:                     ('dist-dir=', 'd',
39:                      "directory to put final built distributions in "
40:                      "[default: dist]"),
41:                     ('skip-build', None,
42:                      "skip rebuilding everything (for testing/debugging)"),
43:                     ('owner=', 'u',
44:                      "Owner name used when creating a tar file"
45:                      " [default: current user]"),
46:                     ('group=', 'g',
47:                      "Group name used when creating a tar file"
48:                      " [default: current group]"),
49:                    ]
50: 
51:     boolean_options = ['skip-build']
52: 
53:     help_options = [
54:         ('help-formats', None,
55:          "lists available distribution formats", show_formats),
56:         ]
57: 
58:     # The following commands do not take a format option from bdist
59:     no_format_option = ('bdist_rpm',)
60: 
61:     # This won't do in reality: will need to distinguish RPM-ish Linux,
62:     # Debian-ish Linux, Solaris, FreeBSD, ..., Windows, Mac OS.
63:     default_format = {'posix': 'gztar',
64:                       'nt': 'zip',
65:                       'os2': 'zip'}
66: 
67:     # Establish the preferred order (for the --help-formats option).
68:     format_commands = ['rpm', 'gztar', 'bztar', 'ztar', 'tar',
69:                        'wininst', 'zip', 'msi']
70: 
71:     # And the real information.
72:     format_command = {'rpm':   ('bdist_rpm',  "RPM distribution"),
73:                       'gztar': ('bdist_dumb', "gzip'ed tar file"),
74:                       'bztar': ('bdist_dumb', "bzip2'ed tar file"),
75:                       'ztar':  ('bdist_dumb', "compressed tar file"),
76:                       'tar':   ('bdist_dumb', "tar file"),
77:                       'wininst': ('bdist_wininst',
78:                                   "Windows executable installer"),
79:                       'zip':   ('bdist_dumb', "ZIP file"),
80:                       'msi':   ('bdist_msi',  "Microsoft Installer")
81:                       }
82: 
83: 
84:     def initialize_options(self):
85:         self.bdist_base = None
86:         self.plat_name = None
87:         self.formats = None
88:         self.dist_dir = None
89:         self.skip_build = 0
90:         self.group = None
91:         self.owner = None
92: 
93:     def finalize_options(self):
94:         # have to finalize 'plat_name' before 'bdist_base'
95:         if self.plat_name is None:
96:             if self.skip_build:
97:                 self.plat_name = get_platform()
98:             else:
99:                 self.plat_name = self.get_finalized_command('build').plat_name
100: 
101:         # 'bdist_base' -- parent of per-built-distribution-format
102:         # temporary directories (eg. we'll probably have
103:         # "build/bdist.<plat>/dumb", "build/bdist.<plat>/rpm", etc.)
104:         if self.bdist_base is None:
105:             build_base = self.get_finalized_command('build').build_base
106:             self.bdist_base = os.path.join(build_base,
107:                                            'bdist.' + self.plat_name)
108: 
109:         self.ensure_string_list('formats')
110:         if self.formats is None:
111:             try:
112:                 self.formats = [self.default_format[os.name]]
113:             except KeyError:
114:                 raise DistutilsPlatformError, \
115:                       "don't know how to create built distributions " + \
116:                       "on platform %s" % os.name
117: 
118:         if self.dist_dir is None:
119:             self.dist_dir = "dist"
120: 
121:     def run(self):
122:         # Figure out which sub-commands we need to run.
123:         commands = []
124:         for format in self.formats:
125:             try:
126:                 commands.append(self.format_command[format][0])
127:             except KeyError:
128:                 raise DistutilsOptionError, "invalid format '%s'" % format
129: 
130:         # Reinitialize and run each command.
131:         for i in range(len(self.formats)):
132:             cmd_name = commands[i]
133:             sub_cmd = self.reinitialize_command(cmd_name)
134:             if cmd_name not in self.no_format_option:
135:                 sub_cmd.format = self.formats[i]
136: 
137:             # passing the owner and group names for tar archiving
138:             if cmd_name == 'bdist_dumb':
139:                 sub_cmd.owner = self.owner
140:                 sub_cmd.group = self.group
141: 
142:             # If we're going to need to run this command again, tell it to
143:             # keep its temporary files around so subsequent runs go faster.
144:             if cmd_name in commands[i+1:]:
145:                 sub_cmd.keep_temp = 1
146:             self.run_command(cmd_name)
147: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_11677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.bdist\n\nImplements the Distutils 'bdist' command (create a built [binary]\ndistribution).")

# Assigning a Str to a Name (line 6):
str_11678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_11678)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'import os' statement (line 8)
import os

import_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.util import get_platform' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_11679 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.util')

if (type(import_11679) is not StypyTypeError):

    if (import_11679 != 'pyd_module'):
        __import__(import_11679)
        sys_modules_11680 = sys.modules[import_11679]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.util', sys_modules_11680.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_11680, sys_modules_11680.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.util', import_11679)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.core import Command' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_11681 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core')

if (type(import_11681) is not StypyTypeError):

    if (import_11681 != 'pyd_module'):
        __import__(import_11681)
        sys_modules_11682 = sys.modules[import_11681]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', sys_modules_11682.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_11682, sys_modules_11682.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.core', import_11681)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils.errors import DistutilsPlatformError, DistutilsOptionError' statement (line 12)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_11683 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors')

if (type(import_11683) is not StypyTypeError):

    if (import_11683 != 'pyd_module'):
        __import__(import_11683)
        sys_modules_11684 = sys.modules[import_11683]
        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', sys_modules_11684.module_type_store, module_type_store, ['DistutilsPlatformError', 'DistutilsOptionError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 12, 0), __file__, sys_modules_11684, sys_modules_11684.module_type_store, module_type_store)
    else:
        from distutils.errors import DistutilsPlatformError, DistutilsOptionError

        import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', None, module_type_store, ['DistutilsPlatformError', 'DistutilsOptionError'], [DistutilsPlatformError, DistutilsOptionError])

else:
    # Assigning a type to the variable 'distutils.errors' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils.errors', import_11683)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')


@norecursion
def show_formats(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'show_formats'
    module_type_store = module_type_store.open_function_context('show_formats', 15, 0, False)
    
    # Passed parameters checking function
    show_formats.stypy_localization = localization
    show_formats.stypy_type_of_self = None
    show_formats.stypy_type_store = module_type_store
    show_formats.stypy_function_name = 'show_formats'
    show_formats.stypy_param_names_list = []
    show_formats.stypy_varargs_param_name = None
    show_formats.stypy_kwargs_param_name = None
    show_formats.stypy_call_defaults = defaults
    show_formats.stypy_call_varargs = varargs
    show_formats.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'show_formats', [], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'show_formats', localization, [], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'show_formats(...)' code ##################

    str_11685 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', 'Print list of available formats (arguments to "--format" option).\n    ')
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 18, 4))
    
    # 'from distutils.fancy_getopt import FancyGetopt' statement (line 18)
    update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
    import_11686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'distutils.fancy_getopt')

    if (type(import_11686) is not StypyTypeError):

        if (import_11686 != 'pyd_module'):
            __import__(import_11686)
            sys_modules_11687 = sys.modules[import_11686]
            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'distutils.fancy_getopt', sys_modules_11687.module_type_store, module_type_store, ['FancyGetopt'])
            nest_module(stypy.reporting.localization.Localization(__file__, 18, 4), __file__, sys_modules_11687, sys_modules_11687.module_type_store, module_type_store)
        else:
            from distutils.fancy_getopt import FancyGetopt

            import_from_module(stypy.reporting.localization.Localization(__file__, 18, 4), 'distutils.fancy_getopt', None, module_type_store, ['FancyGetopt'], [FancyGetopt])

    else:
        # Assigning a type to the variable 'distutils.fancy_getopt' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'distutils.fancy_getopt', import_11686)

    remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')
    
    
    # Assigning a List to a Name (line 19):
    
    # Obtaining an instance of the builtin type 'list' (line 19)
    list_11688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 14), 'list')
    # Adding type elements to the builtin type 'list' instance (line 19)
    
    # Assigning a type to the variable 'formats' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'formats', list_11688)
    
    # Getting the type of 'bdist' (line 20)
    bdist_11689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 18), 'bdist')
    # Obtaining the member 'format_commands' of a type (line 20)
    format_commands_11690 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 18), bdist_11689, 'format_commands')
    # Testing the type of a for loop iterable (line 20)
    is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 20, 4), format_commands_11690)
    # Getting the type of the for loop variable (line 20)
    for_loop_var_11691 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 20, 4), format_commands_11690)
    # Assigning a type to the variable 'format' (line 20)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'format', for_loop_var_11691)
    # SSA begins for a for statement (line 20)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
    
    # Call to append(...): (line 21)
    # Processing the call arguments (line 21)
    
    # Obtaining an instance of the builtin type 'tuple' (line 21)
    tuple_11694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'tuple')
    # Adding type elements to the builtin type 'tuple' instance (line 21)
    # Adding element type (line 21)
    str_11695 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 24), 'str', 'formats=')
    # Getting the type of 'format' (line 21)
    format_11696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 37), 'format', False)
    # Applying the binary operator '+' (line 21)
    result_add_11697 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 24), '+', str_11695, format_11696)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 24), tuple_11694, result_add_11697)
    # Adding element type (line 21)
    # Getting the type of 'None' (line 21)
    None_11698 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 45), 'None', False)
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 24), tuple_11694, None_11698)
    # Adding element type (line 21)
    
    # Obtaining the type of the subscript
    int_11699 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 53), 'int')
    
    # Obtaining the type of the subscript
    # Getting the type of 'format' (line 22)
    format_11700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 45), 'format', False)
    # Getting the type of 'bdist' (line 22)
    bdist_11701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'bdist', False)
    # Obtaining the member 'format_command' of a type (line 22)
    format_command_11702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), bdist_11701, 'format_command')
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___11703 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), format_command_11702, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_11704 = invoke(stypy.reporting.localization.Localization(__file__, 22, 24), getitem___11703, format_11700)
    
    # Obtaining the member '__getitem__' of a type (line 22)
    getitem___11705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 24), subscript_call_result_11704, '__getitem__')
    # Calling the subscript (__getitem__) to obtain the elements type (line 22)
    subscript_call_result_11706 = invoke(stypy.reporting.localization.Localization(__file__, 22, 24), getitem___11705, int_11699)
    
    add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 24), tuple_11694, subscript_call_result_11706)
    
    # Processing the call keyword arguments (line 21)
    kwargs_11707 = {}
    # Getting the type of 'formats' (line 21)
    formats_11692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'formats', False)
    # Obtaining the member 'append' of a type (line 21)
    append_11693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 8), formats_11692, 'append')
    # Calling append(args, kwargs) (line 21)
    append_call_result_11708 = invoke(stypy.reporting.localization.Localization(__file__, 21, 8), append_11693, *[tuple_11694], **kwargs_11707)
    
    # SSA join for a for statement
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Assigning a Call to a Name (line 23):
    
    # Call to FancyGetopt(...): (line 23)
    # Processing the call arguments (line 23)
    # Getting the type of 'formats' (line 23)
    formats_11710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 33), 'formats', False)
    # Processing the call keyword arguments (line 23)
    kwargs_11711 = {}
    # Getting the type of 'FancyGetopt' (line 23)
    FancyGetopt_11709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 21), 'FancyGetopt', False)
    # Calling FancyGetopt(args, kwargs) (line 23)
    FancyGetopt_call_result_11712 = invoke(stypy.reporting.localization.Localization(__file__, 23, 21), FancyGetopt_11709, *[formats_11710], **kwargs_11711)
    
    # Assigning a type to the variable 'pretty_printer' (line 23)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'pretty_printer', FancyGetopt_call_result_11712)
    
    # Call to print_help(...): (line 24)
    # Processing the call arguments (line 24)
    str_11715 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 30), 'str', 'List of available distribution formats:')
    # Processing the call keyword arguments (line 24)
    kwargs_11716 = {}
    # Getting the type of 'pretty_printer' (line 24)
    pretty_printer_11713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'pretty_printer', False)
    # Obtaining the member 'print_help' of a type (line 24)
    print_help_11714 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 4), pretty_printer_11713, 'print_help')
    # Calling print_help(args, kwargs) (line 24)
    print_help_call_result_11717 = invoke(stypy.reporting.localization.Localization(__file__, 24, 4), print_help_11714, *[str_11715], **kwargs_11716)
    
    
    # ################# End of 'show_formats(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show_formats' in the type store
    # Getting the type of 'stypy_return_type' (line 15)
    stypy_return_type_11718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_11718)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show_formats'
    return stypy_return_type_11718

# Assigning a type to the variable 'show_formats' (line 15)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'show_formats', show_formats)
# Declaration of the 'bdist' class
# Getting the type of 'Command' (line 27)
Command_11719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'Command')

class bdist(Command_11719, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 84, 4, False)
        # Assigning a type to the variable 'self' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist.initialize_options.__dict__.__setitem__('stypy_function_name', 'bdist.initialize_options')
        bdist.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 85):
        # Getting the type of 'None' (line 85)
        None_11720 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 26), 'None')
        # Getting the type of 'self' (line 85)
        self_11721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 8), 'self')
        # Setting the type of the member 'bdist_base' of a type (line 85)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 85, 8), self_11721, 'bdist_base', None_11720)
        
        # Assigning a Name to a Attribute (line 86):
        # Getting the type of 'None' (line 86)
        None_11722 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 25), 'None')
        # Getting the type of 'self' (line 86)
        self_11723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 8), 'self')
        # Setting the type of the member 'plat_name' of a type (line 86)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 8), self_11723, 'plat_name', None_11722)
        
        # Assigning a Name to a Attribute (line 87):
        # Getting the type of 'None' (line 87)
        None_11724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'None')
        # Getting the type of 'self' (line 87)
        self_11725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 8), 'self')
        # Setting the type of the member 'formats' of a type (line 87)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 8), self_11725, 'formats', None_11724)
        
        # Assigning a Name to a Attribute (line 88):
        # Getting the type of 'None' (line 88)
        None_11726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'None')
        # Getting the type of 'self' (line 88)
        self_11727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 8), 'self')
        # Setting the type of the member 'dist_dir' of a type (line 88)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 8), self_11727, 'dist_dir', None_11726)
        
        # Assigning a Num to a Attribute (line 89):
        int_11728 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 89, 26), 'int')
        # Getting the type of 'self' (line 89)
        self_11729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 8), 'self')
        # Setting the type of the member 'skip_build' of a type (line 89)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 8), self_11729, 'skip_build', int_11728)
        
        # Assigning a Name to a Attribute (line 90):
        # Getting the type of 'None' (line 90)
        None_11730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 21), 'None')
        # Getting the type of 'self' (line 90)
        self_11731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 90, 8), 'self')
        # Setting the type of the member 'group' of a type (line 90)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 90, 8), self_11731, 'group', None_11730)
        
        # Assigning a Name to a Attribute (line 91):
        # Getting the type of 'None' (line 91)
        None_11732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 21), 'None')
        # Getting the type of 'self' (line 91)
        self_11733 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 8), 'self')
        # Setting the type of the member 'owner' of a type (line 91)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 8), self_11733, 'owner', None_11732)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 84)
        stypy_return_type_11734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_11734


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 93, 4, False)
        # Assigning a type to the variable 'self' (line 94)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 94, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        bdist.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist.finalize_options.__dict__.__setitem__('stypy_function_name', 'bdist.finalize_options')
        bdist.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        bdist.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Type idiom detected: calculating its left and rigth part (line 95)
        # Getting the type of 'self' (line 95)
        self_11735 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 11), 'self')
        # Obtaining the member 'plat_name' of a type (line 95)
        plat_name_11736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 11), self_11735, 'plat_name')
        # Getting the type of 'None' (line 95)
        None_11737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 29), 'None')
        
        (may_be_11738, more_types_in_union_11739) = may_be_none(plat_name_11736, None_11737)

        if may_be_11738:

            if more_types_in_union_11739:
                # Runtime conditional SSA (line 95)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Getting the type of 'self' (line 96)
            self_11740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 15), 'self')
            # Obtaining the member 'skip_build' of a type (line 96)
            skip_build_11741 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 15), self_11740, 'skip_build')
            # Testing the type of an if condition (line 96)
            if_condition_11742 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 96, 12), skip_build_11741)
            # Assigning a type to the variable 'if_condition_11742' (line 96)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 12), 'if_condition_11742', if_condition_11742)
            # SSA begins for if statement (line 96)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Attribute (line 97):
            
            # Call to get_platform(...): (line 97)
            # Processing the call keyword arguments (line 97)
            kwargs_11744 = {}
            # Getting the type of 'get_platform' (line 97)
            get_platform_11743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 33), 'get_platform', False)
            # Calling get_platform(args, kwargs) (line 97)
            get_platform_call_result_11745 = invoke(stypy.reporting.localization.Localization(__file__, 97, 33), get_platform_11743, *[], **kwargs_11744)
            
            # Getting the type of 'self' (line 97)
            self_11746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 16), 'self')
            # Setting the type of the member 'plat_name' of a type (line 97)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 16), self_11746, 'plat_name', get_platform_call_result_11745)
            # SSA branch for the else part of an if statement (line 96)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Attribute (line 99):
            
            # Call to get_finalized_command(...): (line 99)
            # Processing the call arguments (line 99)
            str_11749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 60), 'str', 'build')
            # Processing the call keyword arguments (line 99)
            kwargs_11750 = {}
            # Getting the type of 'self' (line 99)
            self_11747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 33), 'self', False)
            # Obtaining the member 'get_finalized_command' of a type (line 99)
            get_finalized_command_11748 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 33), self_11747, 'get_finalized_command')
            # Calling get_finalized_command(args, kwargs) (line 99)
            get_finalized_command_call_result_11751 = invoke(stypy.reporting.localization.Localization(__file__, 99, 33), get_finalized_command_11748, *[str_11749], **kwargs_11750)
            
            # Obtaining the member 'plat_name' of a type (line 99)
            plat_name_11752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 33), get_finalized_command_call_result_11751, 'plat_name')
            # Getting the type of 'self' (line 99)
            self_11753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 16), 'self')
            # Setting the type of the member 'plat_name' of a type (line 99)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 16), self_11753, 'plat_name', plat_name_11752)
            # SSA join for if statement (line 96)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_11739:
                # SSA join for if statement (line 95)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 104)
        # Getting the type of 'self' (line 104)
        self_11754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 11), 'self')
        # Obtaining the member 'bdist_base' of a type (line 104)
        bdist_base_11755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 11), self_11754, 'bdist_base')
        # Getting the type of 'None' (line 104)
        None_11756 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 30), 'None')
        
        (may_be_11757, more_types_in_union_11758) = may_be_none(bdist_base_11755, None_11756)

        if may_be_11757:

            if more_types_in_union_11758:
                # Runtime conditional SSA (line 104)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Attribute to a Name (line 105):
            
            # Call to get_finalized_command(...): (line 105)
            # Processing the call arguments (line 105)
            str_11761 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 52), 'str', 'build')
            # Processing the call keyword arguments (line 105)
            kwargs_11762 = {}
            # Getting the type of 'self' (line 105)
            self_11759 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 25), 'self', False)
            # Obtaining the member 'get_finalized_command' of a type (line 105)
            get_finalized_command_11760 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), self_11759, 'get_finalized_command')
            # Calling get_finalized_command(args, kwargs) (line 105)
            get_finalized_command_call_result_11763 = invoke(stypy.reporting.localization.Localization(__file__, 105, 25), get_finalized_command_11760, *[str_11761], **kwargs_11762)
            
            # Obtaining the member 'build_base' of a type (line 105)
            build_base_11764 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 25), get_finalized_command_call_result_11763, 'build_base')
            # Assigning a type to the variable 'build_base' (line 105)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'build_base', build_base_11764)
            
            # Assigning a Call to a Attribute (line 106):
            
            # Call to join(...): (line 106)
            # Processing the call arguments (line 106)
            # Getting the type of 'build_base' (line 106)
            build_base_11768 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 43), 'build_base', False)
            str_11769 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 43), 'str', 'bdist.')
            # Getting the type of 'self' (line 107)
            self_11770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 54), 'self', False)
            # Obtaining the member 'plat_name' of a type (line 107)
            plat_name_11771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 107, 54), self_11770, 'plat_name')
            # Applying the binary operator '+' (line 107)
            result_add_11772 = python_operator(stypy.reporting.localization.Localization(__file__, 107, 43), '+', str_11769, plat_name_11771)
            
            # Processing the call keyword arguments (line 106)
            kwargs_11773 = {}
            # Getting the type of 'os' (line 106)
            os_11765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 30), 'os', False)
            # Obtaining the member 'path' of a type (line 106)
            path_11766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 30), os_11765, 'path')
            # Obtaining the member 'join' of a type (line 106)
            join_11767 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 30), path_11766, 'join')
            # Calling join(args, kwargs) (line 106)
            join_call_result_11774 = invoke(stypy.reporting.localization.Localization(__file__, 106, 30), join_11767, *[build_base_11768, result_add_11772], **kwargs_11773)
            
            # Getting the type of 'self' (line 106)
            self_11775 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 12), 'self')
            # Setting the type of the member 'bdist_base' of a type (line 106)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 106, 12), self_11775, 'bdist_base', join_call_result_11774)

            if more_types_in_union_11758:
                # SSA join for if statement (line 104)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Call to ensure_string_list(...): (line 109)
        # Processing the call arguments (line 109)
        str_11778 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 109, 32), 'str', 'formats')
        # Processing the call keyword arguments (line 109)
        kwargs_11779 = {}
        # Getting the type of 'self' (line 109)
        self_11776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 8), 'self', False)
        # Obtaining the member 'ensure_string_list' of a type (line 109)
        ensure_string_list_11777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 8), self_11776, 'ensure_string_list')
        # Calling ensure_string_list(args, kwargs) (line 109)
        ensure_string_list_call_result_11780 = invoke(stypy.reporting.localization.Localization(__file__, 109, 8), ensure_string_list_11777, *[str_11778], **kwargs_11779)
        
        
        # Type idiom detected: calculating its left and rigth part (line 110)
        # Getting the type of 'self' (line 110)
        self_11781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'self')
        # Obtaining the member 'formats' of a type (line 110)
        formats_11782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 110, 11), self_11781, 'formats')
        # Getting the type of 'None' (line 110)
        None_11783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'None')
        
        (may_be_11784, more_types_in_union_11785) = may_be_none(formats_11782, None_11783)

        if may_be_11784:

            if more_types_in_union_11785:
                # Runtime conditional SSA (line 110)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # SSA begins for try-except statement (line 111)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a List to a Attribute (line 112):
            
            # Obtaining an instance of the builtin type 'list' (line 112)
            list_11786 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 31), 'list')
            # Adding type elements to the builtin type 'list' instance (line 112)
            # Adding element type (line 112)
            
            # Obtaining the type of the subscript
            # Getting the type of 'os' (line 112)
            os_11787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 52), 'os')
            # Obtaining the member 'name' of a type (line 112)
            name_11788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 52), os_11787, 'name')
            # Getting the type of 'self' (line 112)
            self_11789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'self')
            # Obtaining the member 'default_format' of a type (line 112)
            default_format_11790 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 32), self_11789, 'default_format')
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___11791 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 32), default_format_11790, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_11792 = invoke(stypy.reporting.localization.Localization(__file__, 112, 32), getitem___11791, name_11788)
            
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 112, 31), list_11786, subscript_call_result_11792)
            
            # Getting the type of 'self' (line 112)
            self_11793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'self')
            # Setting the type of the member 'formats' of a type (line 112)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 16), self_11793, 'formats', list_11786)
            # SSA branch for the except part of a try statement (line 111)
            # SSA branch for the except 'KeyError' branch of a try statement (line 111)
            module_type_store.open_ssa_branch('except')
            # Getting the type of 'DistutilsPlatformError' (line 114)
            DistutilsPlatformError_11794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 22), 'DistutilsPlatformError')
            ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 114, 16), DistutilsPlatformError_11794, 'raise parameter', BaseException)
            # SSA join for try-except statement (line 111)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_11785:
                # SSA join for if statement (line 110)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 118)
        # Getting the type of 'self' (line 118)
        self_11795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 11), 'self')
        # Obtaining the member 'dist_dir' of a type (line 118)
        dist_dir_11796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 118, 11), self_11795, 'dist_dir')
        # Getting the type of 'None' (line 118)
        None_11797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 28), 'None')
        
        (may_be_11798, more_types_in_union_11799) = may_be_none(dist_dir_11796, None_11797)

        if may_be_11798:

            if more_types_in_union_11799:
                # Runtime conditional SSA (line 118)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Str to a Attribute (line 119):
            str_11800 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 119, 28), 'str', 'dist')
            # Getting the type of 'self' (line 119)
            self_11801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'self')
            # Setting the type of the member 'dist_dir' of a type (line 119)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 12), self_11801, 'dist_dir', str_11800)

            if more_types_in_union_11799:
                # SSA join for if statement (line 118)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 93)
        stypy_return_type_11802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11802)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_11802


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 121, 4, False)
        # Assigning a type to the variable 'self' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        bdist.run.__dict__.__setitem__('stypy_localization', localization)
        bdist.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        bdist.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        bdist.run.__dict__.__setitem__('stypy_function_name', 'bdist.run')
        bdist.run.__dict__.__setitem__('stypy_param_names_list', [])
        bdist.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        bdist.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        bdist.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        bdist.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        bdist.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        bdist.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Name (line 123):
        
        # Obtaining an instance of the builtin type 'list' (line 123)
        list_11803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 123, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 123)
        
        # Assigning a type to the variable 'commands' (line 123)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 123, 8), 'commands', list_11803)
        
        # Getting the type of 'self' (line 124)
        self_11804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 22), 'self')
        # Obtaining the member 'formats' of a type (line 124)
        formats_11805 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 124, 22), self_11804, 'formats')
        # Testing the type of a for loop iterable (line 124)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 124, 8), formats_11805)
        # Getting the type of the for loop variable (line 124)
        for_loop_var_11806 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 124, 8), formats_11805)
        # Assigning a type to the variable 'format' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 8), 'format', for_loop_var_11806)
        # SSA begins for a for statement (line 124)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # SSA begins for try-except statement (line 125)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to append(...): (line 126)
        # Processing the call arguments (line 126)
        
        # Obtaining the type of the subscript
        int_11809 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 126, 60), 'int')
        
        # Obtaining the type of the subscript
        # Getting the type of 'format' (line 126)
        format_11810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 52), 'format', False)
        # Getting the type of 'self' (line 126)
        self_11811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'self', False)
        # Obtaining the member 'format_command' of a type (line 126)
        format_command_11812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 32), self_11811, 'format_command')
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___11813 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 32), format_command_11812, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_11814 = invoke(stypy.reporting.localization.Localization(__file__, 126, 32), getitem___11813, format_11810)
        
        # Obtaining the member '__getitem__' of a type (line 126)
        getitem___11815 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 32), subscript_call_result_11814, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 126)
        subscript_call_result_11816 = invoke(stypy.reporting.localization.Localization(__file__, 126, 32), getitem___11815, int_11809)
        
        # Processing the call keyword arguments (line 126)
        kwargs_11817 = {}
        # Getting the type of 'commands' (line 126)
        commands_11807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'commands', False)
        # Obtaining the member 'append' of a type (line 126)
        append_11808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 126, 16), commands_11807, 'append')
        # Calling append(args, kwargs) (line 126)
        append_call_result_11818 = invoke(stypy.reporting.localization.Localization(__file__, 126, 16), append_11808, *[subscript_call_result_11816], **kwargs_11817)
        
        # SSA branch for the except part of a try statement (line 125)
        # SSA branch for the except 'KeyError' branch of a try statement (line 125)
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'DistutilsOptionError' (line 128)
        DistutilsOptionError_11819 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 128, 22), 'DistutilsOptionError')
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 128, 16), DistutilsOptionError_11819, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 125)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to range(...): (line 131)
        # Processing the call arguments (line 131)
        
        # Call to len(...): (line 131)
        # Processing the call arguments (line 131)
        # Getting the type of 'self' (line 131)
        self_11822 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 27), 'self', False)
        # Obtaining the member 'formats' of a type (line 131)
        formats_11823 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 131, 27), self_11822, 'formats')
        # Processing the call keyword arguments (line 131)
        kwargs_11824 = {}
        # Getting the type of 'len' (line 131)
        len_11821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 23), 'len', False)
        # Calling len(args, kwargs) (line 131)
        len_call_result_11825 = invoke(stypy.reporting.localization.Localization(__file__, 131, 23), len_11821, *[formats_11823], **kwargs_11824)
        
        # Processing the call keyword arguments (line 131)
        kwargs_11826 = {}
        # Getting the type of 'range' (line 131)
        range_11820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 17), 'range', False)
        # Calling range(args, kwargs) (line 131)
        range_call_result_11827 = invoke(stypy.reporting.localization.Localization(__file__, 131, 17), range_11820, *[len_call_result_11825], **kwargs_11826)
        
        # Testing the type of a for loop iterable (line 131)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 131, 8), range_call_result_11827)
        # Getting the type of the for loop variable (line 131)
        for_loop_var_11828 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 131, 8), range_call_result_11827)
        # Assigning a type to the variable 'i' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'i', for_loop_var_11828)
        # SSA begins for a for statement (line 131)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Subscript to a Name (line 132):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 132)
        i_11829 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 32), 'i')
        # Getting the type of 'commands' (line 132)
        commands_11830 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 132, 23), 'commands')
        # Obtaining the member '__getitem__' of a type (line 132)
        getitem___11831 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 132, 23), commands_11830, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 132)
        subscript_call_result_11832 = invoke(stypy.reporting.localization.Localization(__file__, 132, 23), getitem___11831, i_11829)
        
        # Assigning a type to the variable 'cmd_name' (line 132)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 132, 12), 'cmd_name', subscript_call_result_11832)
        
        # Assigning a Call to a Name (line 133):
        
        # Call to reinitialize_command(...): (line 133)
        # Processing the call arguments (line 133)
        # Getting the type of 'cmd_name' (line 133)
        cmd_name_11835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 48), 'cmd_name', False)
        # Processing the call keyword arguments (line 133)
        kwargs_11836 = {}
        # Getting the type of 'self' (line 133)
        self_11833 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 22), 'self', False)
        # Obtaining the member 'reinitialize_command' of a type (line 133)
        reinitialize_command_11834 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 133, 22), self_11833, 'reinitialize_command')
        # Calling reinitialize_command(args, kwargs) (line 133)
        reinitialize_command_call_result_11837 = invoke(stypy.reporting.localization.Localization(__file__, 133, 22), reinitialize_command_11834, *[cmd_name_11835], **kwargs_11836)
        
        # Assigning a type to the variable 'sub_cmd' (line 133)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 133, 12), 'sub_cmd', reinitialize_command_call_result_11837)
        
        
        # Getting the type of 'cmd_name' (line 134)
        cmd_name_11838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 15), 'cmd_name')
        # Getting the type of 'self' (line 134)
        self_11839 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 134, 31), 'self')
        # Obtaining the member 'no_format_option' of a type (line 134)
        no_format_option_11840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 134, 31), self_11839, 'no_format_option')
        # Applying the binary operator 'notin' (line 134)
        result_contains_11841 = python_operator(stypy.reporting.localization.Localization(__file__, 134, 15), 'notin', cmd_name_11838, no_format_option_11840)
        
        # Testing the type of an if condition (line 134)
        if_condition_11842 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 134, 12), result_contains_11841)
        # Assigning a type to the variable 'if_condition_11842' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 12), 'if_condition_11842', if_condition_11842)
        # SSA begins for if statement (line 134)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Subscript to a Attribute (line 135):
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 135)
        i_11843 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 46), 'i')
        # Getting the type of 'self' (line 135)
        self_11844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 33), 'self')
        # Obtaining the member 'formats' of a type (line 135)
        formats_11845 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 33), self_11844, 'formats')
        # Obtaining the member '__getitem__' of a type (line 135)
        getitem___11846 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 33), formats_11845, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 135)
        subscript_call_result_11847 = invoke(stypy.reporting.localization.Localization(__file__, 135, 33), getitem___11846, i_11843)
        
        # Getting the type of 'sub_cmd' (line 135)
        sub_cmd_11848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 135, 16), 'sub_cmd')
        # Setting the type of the member 'format' of a type (line 135)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 135, 16), sub_cmd_11848, 'format', subscript_call_result_11847)
        # SSA join for if statement (line 134)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'cmd_name' (line 138)
        cmd_name_11849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 138, 15), 'cmd_name')
        str_11850 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 138, 27), 'str', 'bdist_dumb')
        # Applying the binary operator '==' (line 138)
        result_eq_11851 = python_operator(stypy.reporting.localization.Localization(__file__, 138, 15), '==', cmd_name_11849, str_11850)
        
        # Testing the type of an if condition (line 138)
        if_condition_11852 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 138, 12), result_eq_11851)
        # Assigning a type to the variable 'if_condition_11852' (line 138)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 138, 12), 'if_condition_11852', if_condition_11852)
        # SSA begins for if statement (line 138)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Attribute to a Attribute (line 139):
        # Getting the type of 'self' (line 139)
        self_11853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 32), 'self')
        # Obtaining the member 'owner' of a type (line 139)
        owner_11854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 32), self_11853, 'owner')
        # Getting the type of 'sub_cmd' (line 139)
        sub_cmd_11855 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 139, 16), 'sub_cmd')
        # Setting the type of the member 'owner' of a type (line 139)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 139, 16), sub_cmd_11855, 'owner', owner_11854)
        
        # Assigning a Attribute to a Attribute (line 140):
        # Getting the type of 'self' (line 140)
        self_11856 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 32), 'self')
        # Obtaining the member 'group' of a type (line 140)
        group_11857 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 32), self_11856, 'group')
        # Getting the type of 'sub_cmd' (line 140)
        sub_cmd_11858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 140, 16), 'sub_cmd')
        # Setting the type of the member 'group' of a type (line 140)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 140, 16), sub_cmd_11858, 'group', group_11857)
        # SSA join for if statement (line 138)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'cmd_name' (line 144)
        cmd_name_11859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 15), 'cmd_name')
        
        # Obtaining the type of the subscript
        # Getting the type of 'i' (line 144)
        i_11860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 36), 'i')
        int_11861 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 144, 38), 'int')
        # Applying the binary operator '+' (line 144)
        result_add_11862 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 36), '+', i_11860, int_11861)
        
        slice_11863 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 144, 27), result_add_11862, None, None)
        # Getting the type of 'commands' (line 144)
        commands_11864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'commands')
        # Obtaining the member '__getitem__' of a type (line 144)
        getitem___11865 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 27), commands_11864, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 144)
        subscript_call_result_11866 = invoke(stypy.reporting.localization.Localization(__file__, 144, 27), getitem___11865, slice_11863)
        
        # Applying the binary operator 'in' (line 144)
        result_contains_11867 = python_operator(stypy.reporting.localization.Localization(__file__, 144, 15), 'in', cmd_name_11859, subscript_call_result_11866)
        
        # Testing the type of an if condition (line 144)
        if_condition_11868 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 12), result_contains_11867)
        # Assigning a type to the variable 'if_condition_11868' (line 144)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 12), 'if_condition_11868', if_condition_11868)
        # SSA begins for if statement (line 144)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Num to a Attribute (line 145):
        int_11869 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 36), 'int')
        # Getting the type of 'sub_cmd' (line 145)
        sub_cmd_11870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 16), 'sub_cmd')
        # Setting the type of the member 'keep_temp' of a type (line 145)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 16), sub_cmd_11870, 'keep_temp', int_11869)
        # SSA join for if statement (line 144)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to run_command(...): (line 146)
        # Processing the call arguments (line 146)
        # Getting the type of 'cmd_name' (line 146)
        cmd_name_11873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 29), 'cmd_name', False)
        # Processing the call keyword arguments (line 146)
        kwargs_11874 = {}
        # Getting the type of 'self' (line 146)
        self_11871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 146, 12), 'self', False)
        # Obtaining the member 'run_command' of a type (line 146)
        run_command_11872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 146, 12), self_11871, 'run_command')
        # Calling run_command(args, kwargs) (line 146)
        run_command_call_result_11875 = invoke(stypy.reporting.localization.Localization(__file__, 146, 12), run_command_11872, *[cmd_name_11873], **kwargs_11874)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 121)
        stypy_return_type_11876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_11876)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_11876


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 27, 0, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'bdist.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'bdist' (line 27)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'bdist', bdist)

# Assigning a Str to a Name (line 29):
str_11877 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 18), 'str', 'create a built (binary) distribution')
# Getting the type of 'bdist'
bdist_11878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_11878, 'description', str_11877)

# Assigning a List to a Name (line 31):

# Obtaining an instance of the builtin type 'list' (line 31)
list_11879 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 31)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 31)
tuple_11880 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 31)
# Adding element type (line 31)
str_11881 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 21), 'str', 'bdist-base=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 21), tuple_11880, str_11881)
# Adding element type (line 31)
str_11882 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 21), tuple_11880, str_11882)
# Adding element type (line 31)
str_11883 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 21), 'str', 'temporary directory for creating built distributions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 21), tuple_11880, str_11883)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_11879, tuple_11880)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_11884 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_11885 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 21), 'str', 'plat-name=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_11884, str_11885)
# Adding element type (line 33)
str_11886 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 35), 'str', 'p')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_11884, str_11886)
# Adding element type (line 33)
str_11887 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 21), 'str', 'platform name to embed in generated filenames (default: %s)')

# Call to get_platform(...): (line 35)
# Processing the call keyword arguments (line 35)
kwargs_11889 = {}
# Getting the type of 'get_platform' (line 35)
get_platform_11888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 39), 'get_platform', False)
# Calling get_platform(args, kwargs) (line 35)
get_platform_call_result_11890 = invoke(stypy.reporting.localization.Localization(__file__, 35, 39), get_platform_11888, *[], **kwargs_11889)

# Applying the binary operator '%' (line 34)
result_mod_11891 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 21), '%', str_11887, get_platform_call_result_11890)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 21), tuple_11884, result_mod_11891)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_11879, tuple_11884)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_11892 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
str_11893 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 21), 'str', 'formats=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 21), tuple_11892, str_11893)
# Adding element type (line 36)
# Getting the type of 'None' (line 36)
None_11894 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 33), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 21), tuple_11892, None_11894)
# Adding element type (line 36)
str_11895 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 21), 'str', 'formats for distribution (comma-separated list)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 21), tuple_11892, str_11895)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_11879, tuple_11892)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 38)
tuple_11896 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 38)
# Adding element type (line 38)
str_11897 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 21), 'str', 'dist-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 21), tuple_11896, str_11897)
# Adding element type (line 38)
str_11898 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 34), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 21), tuple_11896, str_11898)
# Adding element type (line 38)
str_11899 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 21), 'str', 'directory to put final built distributions in [default: dist]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 21), tuple_11896, str_11899)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_11879, tuple_11896)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 41)
tuple_11900 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 41)
# Adding element type (line 41)
str_11901 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 21), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 21), tuple_11900, str_11901)
# Adding element type (line 41)
# Getting the type of 'None' (line 41)
None_11902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 35), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 21), tuple_11900, None_11902)
# Adding element type (line 41)
str_11903 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 21), 'str', 'skip rebuilding everything (for testing/debugging)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 21), tuple_11900, str_11903)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_11879, tuple_11900)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 43)
tuple_11904 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 43)
# Adding element type (line 43)
str_11905 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 21), 'str', 'owner=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_11904, str_11905)
# Adding element type (line 43)
str_11906 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 31), 'str', 'u')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_11904, str_11906)
# Adding element type (line 43)
str_11907 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 21), 'str', 'Owner name used when creating a tar file [default: current user]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 21), tuple_11904, str_11907)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_11879, tuple_11904)
# Adding element type (line 31)

# Obtaining an instance of the builtin type 'tuple' (line 46)
tuple_11908 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 46)
# Adding element type (line 46)
str_11909 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 21), 'str', 'group=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_11908, str_11909)
# Adding element type (line 46)
str_11910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 31), 'str', 'g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_11908, str_11910)
# Adding element type (line 46)
str_11911 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 21), 'str', 'Group name used when creating a tar file [default: current group]')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 21), tuple_11908, str_11911)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 19), list_11879, tuple_11908)

# Getting the type of 'bdist'
bdist_11912 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_11912, 'user_options', list_11879)

# Assigning a List to a Name (line 51):

# Obtaining an instance of the builtin type 'list' (line 51)
list_11913 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 51)
# Adding element type (line 51)
str_11914 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 23), 'str', 'skip-build')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 51, 22), list_11913, str_11914)

# Getting the type of 'bdist'
bdist_11915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_11915, 'boolean_options', list_11913)

# Assigning a List to a Name (line 53):

# Obtaining an instance of the builtin type 'list' (line 53)
list_11916 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 53)
# Adding element type (line 53)

# Obtaining an instance of the builtin type 'tuple' (line 54)
tuple_11917 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 54)
# Adding element type (line 54)
str_11918 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 9), 'str', 'help-formats')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_11917, str_11918)
# Adding element type (line 54)
# Getting the type of 'None' (line 54)
None_11919 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 25), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_11917, None_11919)
# Adding element type (line 54)
str_11920 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 55, 9), 'str', 'lists available distribution formats')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_11917, str_11920)
# Adding element type (line 54)
# Getting the type of 'show_formats' (line 55)
show_formats_11921 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 49), 'show_formats')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 54, 9), tuple_11917, show_formats_11921)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 53, 19), list_11916, tuple_11917)

# Getting the type of 'bdist'
bdist_11922 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_11922, 'help_options', list_11916)

# Assigning a Tuple to a Name (line 59):

# Obtaining an instance of the builtin type 'tuple' (line 59)
tuple_11923 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 59)
# Adding element type (line 59)
str_11924 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 24), 'str', 'bdist_rpm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 59, 24), tuple_11923, str_11924)

# Getting the type of 'bdist'
bdist_11925 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist')
# Setting the type of the member 'no_format_option' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_11925, 'no_format_option', tuple_11923)

# Assigning a Dict to a Name (line 63):

# Obtaining an instance of the builtin type 'dict' (line 63)
dict_11926 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 63)
# Adding element type (key, value) (line 63)
str_11927 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 22), 'str', 'posix')
str_11928 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 31), 'str', 'gztar')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), dict_11926, (str_11927, str_11928))
# Adding element type (key, value) (line 63)
str_11929 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 22), 'str', 'nt')
str_11930 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 28), 'str', 'zip')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), dict_11926, (str_11929, str_11930))
# Adding element type (key, value) (line 63)
str_11931 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 22), 'str', 'os2')
str_11932 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 29), 'str', 'zip')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 63, 21), dict_11926, (str_11931, str_11932))

# Getting the type of 'bdist'
bdist_11933 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist')
# Setting the type of the member 'default_format' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_11933, 'default_format', dict_11926)

# Assigning a List to a Name (line 68):

# Obtaining an instance of the builtin type 'list' (line 68)
list_11934 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 68)
# Adding element type (line 68)
str_11935 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 23), 'str', 'rpm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_11934, str_11935)
# Adding element type (line 68)
str_11936 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 30), 'str', 'gztar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_11934, str_11936)
# Adding element type (line 68)
str_11937 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 39), 'str', 'bztar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_11934, str_11937)
# Adding element type (line 68)
str_11938 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 48), 'str', 'ztar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_11934, str_11938)
# Adding element type (line 68)
str_11939 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 56), 'str', 'tar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_11934, str_11939)
# Adding element type (line 68)
str_11940 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 23), 'str', 'wininst')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_11934, str_11940)
# Adding element type (line 68)
str_11941 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'str', 'zip')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_11934, str_11941)
# Adding element type (line 68)
str_11942 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 41), 'str', 'msi')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 68, 22), list_11934, str_11942)

# Getting the type of 'bdist'
bdist_11943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist')
# Setting the type of the member 'format_commands' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_11943, 'format_commands', list_11934)

# Assigning a Dict to a Name (line 72):

# Obtaining an instance of the builtin type 'dict' (line 72)
dict_11944 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 21), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 72)
# Adding element type (key, value) (line 72)
str_11945 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 22), 'str', 'rpm')

# Obtaining an instance of the builtin type 'tuple' (line 72)
tuple_11946 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 72)
# Adding element type (line 72)
str_11947 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 32), 'str', 'bdist_rpm')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 32), tuple_11946, str_11947)
# Adding element type (line 72)
str_11948 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 46), 'str', 'RPM distribution')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 32), tuple_11946, str_11948)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), dict_11944, (str_11945, tuple_11946))
# Adding element type (key, value) (line 72)
str_11949 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 22), 'str', 'gztar')

# Obtaining an instance of the builtin type 'tuple' (line 73)
tuple_11950 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 73)
# Adding element type (line 73)
str_11951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 32), 'str', 'bdist_dumb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 32), tuple_11950, str_11951)
# Adding element type (line 73)
str_11952 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 46), 'str', "gzip'ed tar file")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 73, 32), tuple_11950, str_11952)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), dict_11944, (str_11949, tuple_11950))
# Adding element type (key, value) (line 72)
str_11953 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 22), 'str', 'bztar')

# Obtaining an instance of the builtin type 'tuple' (line 74)
tuple_11954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 74)
# Adding element type (line 74)
str_11955 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 32), 'str', 'bdist_dumb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 32), tuple_11954, str_11955)
# Adding element type (line 74)
str_11956 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 46), 'str', "bzip2'ed tar file")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 74, 32), tuple_11954, str_11956)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), dict_11944, (str_11953, tuple_11954))
# Adding element type (key, value) (line 72)
str_11957 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 22), 'str', 'ztar')

# Obtaining an instance of the builtin type 'tuple' (line 75)
tuple_11958 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 75)
# Adding element type (line 75)
str_11959 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 32), 'str', 'bdist_dumb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 32), tuple_11958, str_11959)
# Adding element type (line 75)
str_11960 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 46), 'str', 'compressed tar file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 75, 32), tuple_11958, str_11960)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), dict_11944, (str_11957, tuple_11958))
# Adding element type (key, value) (line 72)
str_11961 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 22), 'str', 'tar')

# Obtaining an instance of the builtin type 'tuple' (line 76)
tuple_11962 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 76)
# Adding element type (line 76)
str_11963 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 32), 'str', 'bdist_dumb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 32), tuple_11962, str_11963)
# Adding element type (line 76)
str_11964 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 46), 'str', 'tar file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 32), tuple_11962, str_11964)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), dict_11944, (str_11961, tuple_11962))
# Adding element type (key, value) (line 72)
str_11965 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 22), 'str', 'wininst')

# Obtaining an instance of the builtin type 'tuple' (line 77)
tuple_11966 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 77)
# Adding element type (line 77)
str_11967 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 34), 'str', 'bdist_wininst')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_11966, str_11967)
# Adding element type (line 77)
str_11968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 34), 'str', 'Windows executable installer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 77, 34), tuple_11966, str_11968)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), dict_11944, (str_11965, tuple_11966))
# Adding element type (key, value) (line 72)
str_11969 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 22), 'str', 'zip')

# Obtaining an instance of the builtin type 'tuple' (line 79)
tuple_11970 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 79)
# Adding element type (line 79)
str_11971 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 32), 'str', 'bdist_dumb')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 32), tuple_11970, str_11971)
# Adding element type (line 79)
str_11972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 46), 'str', 'ZIP file')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 32), tuple_11970, str_11972)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), dict_11944, (str_11969, tuple_11970))
# Adding element type (key, value) (line 72)
str_11973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 22), 'str', 'msi')

# Obtaining an instance of the builtin type 'tuple' (line 80)
tuple_11974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 32), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 80)
# Adding element type (line 80)
str_11975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 32), 'str', 'bdist_msi')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 32), tuple_11974, str_11975)
# Adding element type (line 80)
str_11976 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 46), 'str', 'Microsoft Installer')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 80, 32), tuple_11974, str_11976)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 72, 21), dict_11944, (str_11973, tuple_11974))

# Getting the type of 'bdist'
bdist_11977 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'bdist')
# Setting the type of the member 'format_command' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), bdist_11977, 'format_command', dict_11944)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
