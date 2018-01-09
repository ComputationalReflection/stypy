
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from distutils.core import Command
4: from numpy.distutils import log
5: 
6: #XXX: Linker flags
7: 
8: def show_fortran_compilers(_cache=[]):
9:     # Using cache to prevent infinite recursion
10:     if _cache: return
11:     _cache.append(1)
12:     from numpy.distutils.fcompiler import show_fcompilers
13:     import distutils.core
14:     dist = distutils.core._setup_distribution
15:     show_fcompilers(dist)
16: 
17: class config_fc(Command):
18:     ''' Distutils command to hold user specified options
19:     to Fortran compilers.
20: 
21:     config_fc command is used by the FCompiler.customize() method.
22:     '''
23: 
24:     description = "specify Fortran 77/Fortran 90 compiler information"
25: 
26:     user_options = [
27:         ('fcompiler=', None, "specify Fortran compiler type"),
28:         ('f77exec=', None, "specify F77 compiler command"),
29:         ('f90exec=', None, "specify F90 compiler command"),
30:         ('f77flags=', None, "specify F77 compiler flags"),
31:         ('f90flags=', None, "specify F90 compiler flags"),
32:         ('opt=', None, "specify optimization flags"),
33:         ('arch=', None, "specify architecture specific optimization flags"),
34:         ('debug', 'g', "compile with debugging information"),
35:         ('noopt', None, "compile without optimization"),
36:         ('noarch', None, "compile without arch-dependent optimization"),
37:         ]
38: 
39:     help_options = [
40:         ('help-fcompiler', None, "list available Fortran compilers",
41:          show_fortran_compilers),
42:         ]
43: 
44:     boolean_options = ['debug', 'noopt', 'noarch']
45: 
46:     def initialize_options(self):
47:         self.fcompiler = None
48:         self.f77exec = None
49:         self.f90exec = None
50:         self.f77flags = None
51:         self.f90flags = None
52:         self.opt = None
53:         self.arch = None
54:         self.debug = None
55:         self.noopt = None
56:         self.noarch = None
57: 
58:     def finalize_options(self):
59:         log.info('unifing config_fc, config, build_clib, build_ext, build commands --fcompiler options')
60:         build_clib = self.get_finalized_command('build_clib')
61:         build_ext = self.get_finalized_command('build_ext')
62:         config = self.get_finalized_command('config')
63:         build = self.get_finalized_command('build')
64:         cmd_list = [self, config, build_clib, build_ext, build]
65:         for a in ['fcompiler']:
66:             l = []
67:             for c in cmd_list:
68:                 v = getattr(c, a)
69:                 if v is not None:
70:                     if not isinstance(v, str): v = v.compiler_type
71:                     if v not in l: l.append(v)
72:             if not l: v1 = None
73:             else: v1 = l[0]
74:             if len(l)>1:
75:                 log.warn('  commands have different --%s options: %s'\
76:                          ', using first in list as default' % (a, l))
77:             if v1:
78:                 for c in cmd_list:
79:                     if getattr(c, a) is None: setattr(c, a, v1)
80: 
81:     def run(self):
82:         # Do nothing.
83:         return
84: 
85: class config_cc(Command):
86:     ''' Distutils command to hold user specified options
87:     to C/C++ compilers.
88:     '''
89: 
90:     description = "specify C/C++ compiler information"
91: 
92:     user_options = [
93:         ('compiler=', None, "specify C/C++ compiler type"),
94:         ]
95: 
96:     def initialize_options(self):
97:         self.compiler = None
98: 
99:     def finalize_options(self):
100:         log.info('unifing config_cc, config, build_clib, build_ext, build commands --compiler options')
101:         build_clib = self.get_finalized_command('build_clib')
102:         build_ext = self.get_finalized_command('build_ext')
103:         config = self.get_finalized_command('config')
104:         build = self.get_finalized_command('build')
105:         cmd_list = [self, config, build_clib, build_ext, build]
106:         for a in ['compiler']:
107:             l = []
108:             for c in cmd_list:
109:                 v = getattr(c, a)
110:                 if v is not None:
111:                     if not isinstance(v, str): v = v.compiler_type
112:                     if v not in l: l.append(v)
113:             if not l: v1 = None
114:             else: v1 = l[0]
115:             if len(l)>1:
116:                 log.warn('  commands have different --%s options: %s'\
117:                          ', using first in list as default' % (a, l))
118:             if v1:
119:                 for c in cmd_list:
120:                     if getattr(c, a) is None: setattr(c, a, v1)
121:         return
122: 
123:     def run(self):
124:         # Do nothing.
125:         return
126: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from distutils.core import Command' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58964 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.core')

if (type(import_58964) is not StypyTypeError):

    if (import_58964 != 'pyd_module'):
        __import__(import_58964)
        sys_modules_58965 = sys.modules[import_58964]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.core', sys_modules_58965.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_58965, sys_modules_58965.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.core', import_58964)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.distutils import log' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_58966 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils')

if (type(import_58966) is not StypyTypeError):

    if (import_58966 != 'pyd_module'):
        __import__(import_58966)
        sys_modules_58967 = sys.modules[import_58966]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils', sys_modules_58967.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_58967, sys_modules_58967.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils', import_58966)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')


@norecursion
def show_fortran_compilers(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    
    # Obtaining an instance of the builtin type 'list' (line 8)
    list_58968 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, 34), 'list')
    # Adding type elements to the builtin type 'list' instance (line 8)
    
    defaults = [list_58968]
    # Create a new context for function 'show_fortran_compilers'
    module_type_store = module_type_store.open_function_context('show_fortran_compilers', 8, 0, False)
    
    # Passed parameters checking function
    show_fortran_compilers.stypy_localization = localization
    show_fortran_compilers.stypy_type_of_self = None
    show_fortran_compilers.stypy_type_store = module_type_store
    show_fortran_compilers.stypy_function_name = 'show_fortran_compilers'
    show_fortran_compilers.stypy_param_names_list = ['_cache']
    show_fortran_compilers.stypy_varargs_param_name = None
    show_fortran_compilers.stypy_kwargs_param_name = None
    show_fortran_compilers.stypy_call_defaults = defaults
    show_fortran_compilers.stypy_call_varargs = varargs
    show_fortran_compilers.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'show_fortran_compilers', ['_cache'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'show_fortran_compilers', localization, ['_cache'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'show_fortran_compilers(...)' code ##################

    
    # Getting the type of '_cache' (line 10)
    _cache_58969 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 7), '_cache')
    # Testing the type of an if condition (line 10)
    if_condition_58970 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 4), _cache_58969)
    # Assigning a type to the variable 'if_condition_58970' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'if_condition_58970', if_condition_58970)
    # SSA begins for if statement (line 10)
    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
    # Assigning a type to the variable 'stypy_return_type' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 15), 'stypy_return_type', types.NoneType)
    # SSA join for if statement (line 10)
    module_type_store = module_type_store.join_ssa_context()
    
    
    # Call to append(...): (line 11)
    # Processing the call arguments (line 11)
    int_58973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'int')
    # Processing the call keyword arguments (line 11)
    kwargs_58974 = {}
    # Getting the type of '_cache' (line 11)
    _cache_58971 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), '_cache', False)
    # Obtaining the member 'append' of a type (line 11)
    append_58972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 4), _cache_58971, 'append')
    # Calling append(args, kwargs) (line 11)
    append_call_result_58975 = invoke(stypy.reporting.localization.Localization(__file__, 11, 4), append_58972, *[int_58973], **kwargs_58974)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 4))
    
    # 'from numpy.distutils.fcompiler import show_fcompilers' statement (line 12)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
    import_58976 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.fcompiler')

    if (type(import_58976) is not StypyTypeError):

        if (import_58976 != 'pyd_module'):
            __import__(import_58976)
            sys_modules_58977 = sys.modules[import_58976]
            import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.fcompiler', sys_modules_58977.module_type_store, module_type_store, ['show_fcompilers'])
            nest_module(stypy.reporting.localization.Localization(__file__, 12, 4), __file__, sys_modules_58977, sys_modules_58977.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import show_fcompilers

            import_from_module(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['show_fcompilers'], [show_fcompilers])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'numpy.distutils.fcompiler', import_58976)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 13, 4))
    
    # 'import distutils.core' statement (line 13)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
    import_58978 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'distutils.core')

    if (type(import_58978) is not StypyTypeError):

        if (import_58978 != 'pyd_module'):
            __import__(import_58978)
            sys_modules_58979 = sys.modules[import_58978]
            import_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'distutils.core', sys_modules_58979.module_type_store, module_type_store)
        else:
            import distutils.core

            import_module(stypy.reporting.localization.Localization(__file__, 13, 4), 'distutils.core', distutils.core, module_type_store)

    else:
        # Assigning a type to the variable 'distutils.core' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'distutils.core', import_58978)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
    
    
    # Assigning a Attribute to a Name (line 14):
    # Getting the type of 'distutils' (line 14)
    distutils_58980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 11), 'distutils')
    # Obtaining the member 'core' of a type (line 14)
    core_58981 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), distutils_58980, 'core')
    # Obtaining the member '_setup_distribution' of a type (line 14)
    _setup_distribution_58982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 11), core_58981, '_setup_distribution')
    # Assigning a type to the variable 'dist' (line 14)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'dist', _setup_distribution_58982)
    
    # Call to show_fcompilers(...): (line 15)
    # Processing the call arguments (line 15)
    # Getting the type of 'dist' (line 15)
    dist_58984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 20), 'dist', False)
    # Processing the call keyword arguments (line 15)
    kwargs_58985 = {}
    # Getting the type of 'show_fcompilers' (line 15)
    show_fcompilers_58983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'show_fcompilers', False)
    # Calling show_fcompilers(args, kwargs) (line 15)
    show_fcompilers_call_result_58986 = invoke(stypy.reporting.localization.Localization(__file__, 15, 4), show_fcompilers_58983, *[dist_58984], **kwargs_58985)
    
    
    # ################# End of 'show_fortran_compilers(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'show_fortran_compilers' in the type store
    # Getting the type of 'stypy_return_type' (line 8)
    stypy_return_type_58987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_58987)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'show_fortran_compilers'
    return stypy_return_type_58987

# Assigning a type to the variable 'show_fortran_compilers' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'show_fortran_compilers', show_fortran_compilers)
# Declaration of the 'config_fc' class
# Getting the type of 'Command' (line 17)
Command_58988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'Command')

class config_fc(Command_58988, ):
    str_58989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', ' Distutils command to hold user specified options\n    to Fortran compilers.\n\n    config_fc command is used by the FCompiler.customize() method.\n    ')

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config_fc.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        config_fc.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config_fc.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        config_fc.initialize_options.__dict__.__setitem__('stypy_function_name', 'config_fc.initialize_options')
        config_fc.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        config_fc.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        config_fc.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config_fc.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        config_fc.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        config_fc.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config_fc.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config_fc.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'None' (line 47)
        None_58990 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 25), 'None')
        # Getting the type of 'self' (line 47)
        self_58991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'fcompiler' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_58991, 'fcompiler', None_58990)
        
        # Assigning a Name to a Attribute (line 48):
        # Getting the type of 'None' (line 48)
        None_58992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 23), 'None')
        # Getting the type of 'self' (line 48)
        self_58993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self')
        # Setting the type of the member 'f77exec' of a type (line 48)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_58993, 'f77exec', None_58992)
        
        # Assigning a Name to a Attribute (line 49):
        # Getting the type of 'None' (line 49)
        None_58994 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 23), 'None')
        # Getting the type of 'self' (line 49)
        self_58995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'self')
        # Setting the type of the member 'f90exec' of a type (line 49)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 8), self_58995, 'f90exec', None_58994)
        
        # Assigning a Name to a Attribute (line 50):
        # Getting the type of 'None' (line 50)
        None_58996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 24), 'None')
        # Getting the type of 'self' (line 50)
        self_58997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'self')
        # Setting the type of the member 'f77flags' of a type (line 50)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 50, 8), self_58997, 'f77flags', None_58996)
        
        # Assigning a Name to a Attribute (line 51):
        # Getting the type of 'None' (line 51)
        None_58998 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 24), 'None')
        # Getting the type of 'self' (line 51)
        self_58999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'self')
        # Setting the type of the member 'f90flags' of a type (line 51)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 8), self_58999, 'f90flags', None_58998)
        
        # Assigning a Name to a Attribute (line 52):
        # Getting the type of 'None' (line 52)
        None_59000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 19), 'None')
        # Getting the type of 'self' (line 52)
        self_59001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 8), 'self')
        # Setting the type of the member 'opt' of a type (line 52)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 8), self_59001, 'opt', None_59000)
        
        # Assigning a Name to a Attribute (line 53):
        # Getting the type of 'None' (line 53)
        None_59002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 20), 'None')
        # Getting the type of 'self' (line 53)
        self_59003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 8), 'self')
        # Setting the type of the member 'arch' of a type (line 53)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 53, 8), self_59003, 'arch', None_59002)
        
        # Assigning a Name to a Attribute (line 54):
        # Getting the type of 'None' (line 54)
        None_59004 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 21), 'None')
        # Getting the type of 'self' (line 54)
        self_59005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'self')
        # Setting the type of the member 'debug' of a type (line 54)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 8), self_59005, 'debug', None_59004)
        
        # Assigning a Name to a Attribute (line 55):
        # Getting the type of 'None' (line 55)
        None_59006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 21), 'None')
        # Getting the type of 'self' (line 55)
        self_59007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 8), 'self')
        # Setting the type of the member 'noopt' of a type (line 55)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 8), self_59007, 'noopt', None_59006)
        
        # Assigning a Name to a Attribute (line 56):
        # Getting the type of 'None' (line 56)
        None_59008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 22), 'None')
        # Getting the type of 'self' (line 56)
        self_59009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 8), 'self')
        # Setting the type of the member 'noarch' of a type (line 56)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 8), self_59009, 'noarch', None_59008)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_59010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59010)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_59010


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
        config_fc.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        config_fc.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config_fc.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        config_fc.finalize_options.__dict__.__setitem__('stypy_function_name', 'config_fc.finalize_options')
        config_fc.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        config_fc.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        config_fc.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config_fc.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        config_fc.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        config_fc.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config_fc.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config_fc.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to info(...): (line 59)
        # Processing the call arguments (line 59)
        str_59013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 17), 'str', 'unifing config_fc, config, build_clib, build_ext, build commands --fcompiler options')
        # Processing the call keyword arguments (line 59)
        kwargs_59014 = {}
        # Getting the type of 'log' (line 59)
        log_59011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 59)
        info_59012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 8), log_59011, 'info')
        # Calling info(args, kwargs) (line 59)
        info_call_result_59015 = invoke(stypy.reporting.localization.Localization(__file__, 59, 8), info_59012, *[str_59013], **kwargs_59014)
        
        
        # Assigning a Call to a Name (line 60):
        
        # Call to get_finalized_command(...): (line 60)
        # Processing the call arguments (line 60)
        str_59018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 48), 'str', 'build_clib')
        # Processing the call keyword arguments (line 60)
        kwargs_59019 = {}
        # Getting the type of 'self' (line 60)
        self_59016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 21), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 60)
        get_finalized_command_59017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 21), self_59016, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 60)
        get_finalized_command_call_result_59020 = invoke(stypy.reporting.localization.Localization(__file__, 60, 21), get_finalized_command_59017, *[str_59018], **kwargs_59019)
        
        # Assigning a type to the variable 'build_clib' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'build_clib', get_finalized_command_call_result_59020)
        
        # Assigning a Call to a Name (line 61):
        
        # Call to get_finalized_command(...): (line 61)
        # Processing the call arguments (line 61)
        str_59023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, 47), 'str', 'build_ext')
        # Processing the call keyword arguments (line 61)
        kwargs_59024 = {}
        # Getting the type of 'self' (line 61)
        self_59021 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 61)
        get_finalized_command_59022 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 20), self_59021, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 61)
        get_finalized_command_call_result_59025 = invoke(stypy.reporting.localization.Localization(__file__, 61, 20), get_finalized_command_59022, *[str_59023], **kwargs_59024)
        
        # Assigning a type to the variable 'build_ext' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'build_ext', get_finalized_command_call_result_59025)
        
        # Assigning a Call to a Name (line 62):
        
        # Call to get_finalized_command(...): (line 62)
        # Processing the call arguments (line 62)
        str_59028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 44), 'str', 'config')
        # Processing the call keyword arguments (line 62)
        kwargs_59029 = {}
        # Getting the type of 'self' (line 62)
        self_59026 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 17), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 62)
        get_finalized_command_59027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 17), self_59026, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 62)
        get_finalized_command_call_result_59030 = invoke(stypy.reporting.localization.Localization(__file__, 62, 17), get_finalized_command_59027, *[str_59028], **kwargs_59029)
        
        # Assigning a type to the variable 'config' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 8), 'config', get_finalized_command_call_result_59030)
        
        # Assigning a Call to a Name (line 63):
        
        # Call to get_finalized_command(...): (line 63)
        # Processing the call arguments (line 63)
        str_59033 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 43), 'str', 'build')
        # Processing the call keyword arguments (line 63)
        kwargs_59034 = {}
        # Getting the type of 'self' (line 63)
        self_59031 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 16), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 63)
        get_finalized_command_59032 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 16), self_59031, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 63)
        get_finalized_command_call_result_59035 = invoke(stypy.reporting.localization.Localization(__file__, 63, 16), get_finalized_command_59032, *[str_59033], **kwargs_59034)
        
        # Assigning a type to the variable 'build' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'build', get_finalized_command_call_result_59035)
        
        # Assigning a List to a Name (line 64):
        
        # Obtaining an instance of the builtin type 'list' (line 64)
        list_59036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 64)
        # Adding element type (line 64)
        # Getting the type of 'self' (line 64)
        self_59037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 20), 'self')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_59036, self_59037)
        # Adding element type (line 64)
        # Getting the type of 'config' (line 64)
        config_59038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 26), 'config')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_59036, config_59038)
        # Adding element type (line 64)
        # Getting the type of 'build_clib' (line 64)
        build_clib_59039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 34), 'build_clib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_59036, build_clib_59039)
        # Adding element type (line 64)
        # Getting the type of 'build_ext' (line 64)
        build_ext_59040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 46), 'build_ext')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_59036, build_ext_59040)
        # Adding element type (line 64)
        # Getting the type of 'build' (line 64)
        build_59041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 57), 'build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 64, 19), list_59036, build_59041)
        
        # Assigning a type to the variable 'cmd_list' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'cmd_list', list_59036)
        
        
        # Obtaining an instance of the builtin type 'list' (line 65)
        list_59042 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 65)
        # Adding element type (line 65)
        str_59043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 65, 18), 'str', 'fcompiler')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 65, 17), list_59042, str_59043)
        
        # Testing the type of a for loop iterable (line 65)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 65, 8), list_59042)
        # Getting the type of the for loop variable (line 65)
        for_loop_var_59044 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 65, 8), list_59042)
        # Assigning a type to the variable 'a' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 8), 'a', for_loop_var_59044)
        # SSA begins for a for statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 66):
        
        # Obtaining an instance of the builtin type 'list' (line 66)
        list_59045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 66, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 66)
        
        # Assigning a type to the variable 'l' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 12), 'l', list_59045)
        
        # Getting the type of 'cmd_list' (line 67)
        cmd_list_59046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'cmd_list')
        # Testing the type of a for loop iterable (line 67)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 67, 12), cmd_list_59046)
        # Getting the type of the for loop variable (line 67)
        for_loop_var_59047 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 67, 12), cmd_list_59046)
        # Assigning a type to the variable 'c' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'c', for_loop_var_59047)
        # SSA begins for a for statement (line 67)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 68):
        
        # Call to getattr(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'c' (line 68)
        c_59049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 28), 'c', False)
        # Getting the type of 'a' (line 68)
        a_59050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 31), 'a', False)
        # Processing the call keyword arguments (line 68)
        kwargs_59051 = {}
        # Getting the type of 'getattr' (line 68)
        getattr_59048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 68)
        getattr_call_result_59052 = invoke(stypy.reporting.localization.Localization(__file__, 68, 20), getattr_59048, *[c_59049, a_59050], **kwargs_59051)
        
        # Assigning a type to the variable 'v' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 16), 'v', getattr_call_result_59052)
        
        # Type idiom detected: calculating its left and rigth part (line 69)
        # Getting the type of 'v' (line 69)
        v_59053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 16), 'v')
        # Getting the type of 'None' (line 69)
        None_59054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 28), 'None')
        
        (may_be_59055, more_types_in_union_59056) = may_not_be_none(v_59053, None_59054)

        if may_be_59055:

            if more_types_in_union_59056:
                # Runtime conditional SSA (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 70)
            # Getting the type of 'str' (line 70)
            str_59057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 41), 'str')
            # Getting the type of 'v' (line 70)
            v_59058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 38), 'v')
            
            (may_be_59059, more_types_in_union_59060) = may_not_be_subtype(str_59057, v_59058)

            if may_be_59059:

                if more_types_in_union_59060:
                    # Runtime conditional SSA (line 70)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'v' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 20), 'v', remove_subtype_from_union(v_59058, str))
                
                # Assigning a Attribute to a Name (line 70):
                # Getting the type of 'v' (line 70)
                v_59061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 51), 'v')
                # Obtaining the member 'compiler_type' of a type (line 70)
                compiler_type_59062 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 51), v_59061, 'compiler_type')
                # Assigning a type to the variable 'v' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 47), 'v', compiler_type_59062)

                if more_types_in_union_59060:
                    # SSA join for if statement (line 70)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Getting the type of 'v' (line 71)
            v_59063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 23), 'v')
            # Getting the type of 'l' (line 71)
            l_59064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 32), 'l')
            # Applying the binary operator 'notin' (line 71)
            result_contains_59065 = python_operator(stypy.reporting.localization.Localization(__file__, 71, 23), 'notin', v_59063, l_59064)
            
            # Testing the type of an if condition (line 71)
            if_condition_59066 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 71, 20), result_contains_59065)
            # Assigning a type to the variable 'if_condition_59066' (line 71)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'if_condition_59066', if_condition_59066)
            # SSA begins for if statement (line 71)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 71)
            # Processing the call arguments (line 71)
            # Getting the type of 'v' (line 71)
            v_59069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 44), 'v', False)
            # Processing the call keyword arguments (line 71)
            kwargs_59070 = {}
            # Getting the type of 'l' (line 71)
            l_59067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 35), 'l', False)
            # Obtaining the member 'append' of a type (line 71)
            append_59068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 35), l_59067, 'append')
            # Calling append(args, kwargs) (line 71)
            append_call_result_59071 = invoke(stypy.reporting.localization.Localization(__file__, 71, 35), append_59068, *[v_59069], **kwargs_59070)
            
            # SSA join for if statement (line 71)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_59056:
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'l' (line 72)
        l_59072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 19), 'l')
        # Applying the 'not' unary operator (line 72)
        result_not__59073 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 15), 'not', l_59072)
        
        # Testing the type of an if condition (line 72)
        if_condition_59074 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 12), result_not__59073)
        # Assigning a type to the variable 'if_condition_59074' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 12), 'if_condition_59074', if_condition_59074)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 72):
        # Getting the type of 'None' (line 72)
        None_59075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 27), 'None')
        # Assigning a type to the variable 'v1' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 22), 'v1', None_59075)
        # SSA branch for the else part of an if statement (line 72)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 73):
        
        # Obtaining the type of the subscript
        int_59076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 73, 25), 'int')
        # Getting the type of 'l' (line 73)
        l_59077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 23), 'l')
        # Obtaining the member '__getitem__' of a type (line 73)
        getitem___59078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 23), l_59077, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 73)
        subscript_call_result_59079 = invoke(stypy.reporting.localization.Localization(__file__, 73, 23), getitem___59078, int_59076)
        
        # Assigning a type to the variable 'v1' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 18), 'v1', subscript_call_result_59079)
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 74)
        # Processing the call arguments (line 74)
        # Getting the type of 'l' (line 74)
        l_59081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 19), 'l', False)
        # Processing the call keyword arguments (line 74)
        kwargs_59082 = {}
        # Getting the type of 'len' (line 74)
        len_59080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 15), 'len', False)
        # Calling len(args, kwargs) (line 74)
        len_call_result_59083 = invoke(stypy.reporting.localization.Localization(__file__, 74, 15), len_59080, *[l_59081], **kwargs_59082)
        
        int_59084 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 22), 'int')
        # Applying the binary operator '>' (line 74)
        result_gt_59085 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 15), '>', len_call_result_59083, int_59084)
        
        # Testing the type of an if condition (line 74)
        if_condition_59086 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 12), result_gt_59085)
        # Assigning a type to the variable 'if_condition_59086' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 12), 'if_condition_59086', if_condition_59086)
        # SSA begins for if statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 75)
        # Processing the call arguments (line 75)
        str_59089 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 75, 25), 'str', '  commands have different --%s options: %s, using first in list as default')
        
        # Obtaining an instance of the builtin type 'tuple' (line 76)
        tuple_59090 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 76)
        # Adding element type (line 76)
        # Getting the type of 'a' (line 76)
        a_59091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 63), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 63), tuple_59090, a_59091)
        # Adding element type (line 76)
        # Getting the type of 'l' (line 76)
        l_59092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 66), 'l', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 76, 63), tuple_59090, l_59092)
        
        # Applying the binary operator '%' (line 75)
        result_mod_59093 = python_operator(stypy.reporting.localization.Localization(__file__, 75, 25), '%', str_59089, tuple_59090)
        
        # Processing the call keyword arguments (line 75)
        kwargs_59094 = {}
        # Getting the type of 'log' (line 75)
        log_59087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 75)
        warn_59088 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), log_59087, 'warn')
        # Calling warn(args, kwargs) (line 75)
        warn_call_result_59095 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), warn_59088, *[result_mod_59093], **kwargs_59094)
        
        # SSA join for if statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'v1' (line 77)
        v1_59096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'v1')
        # Testing the type of an if condition (line 77)
        if_condition_59097 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 12), v1_59096)
        # Assigning a type to the variable 'if_condition_59097' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'if_condition_59097', if_condition_59097)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'cmd_list' (line 78)
        cmd_list_59098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 25), 'cmd_list')
        # Testing the type of a for loop iterable (line 78)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 78, 16), cmd_list_59098)
        # Getting the type of the for loop variable (line 78)
        for_loop_var_59099 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 78, 16), cmd_list_59098)
        # Assigning a type to the variable 'c' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'c', for_loop_var_59099)
        # SSA begins for a for statement (line 78)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 79)
        
        # Call to getattr(...): (line 79)
        # Processing the call arguments (line 79)
        # Getting the type of 'c' (line 79)
        c_59101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 31), 'c', False)
        # Getting the type of 'a' (line 79)
        a_59102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 34), 'a', False)
        # Processing the call keyword arguments (line 79)
        kwargs_59103 = {}
        # Getting the type of 'getattr' (line 79)
        getattr_59100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 23), 'getattr', False)
        # Calling getattr(args, kwargs) (line 79)
        getattr_call_result_59104 = invoke(stypy.reporting.localization.Localization(__file__, 79, 23), getattr_59100, *[c_59101, a_59102], **kwargs_59103)
        
        # Getting the type of 'None' (line 79)
        None_59105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 40), 'None')
        
        (may_be_59106, more_types_in_union_59107) = may_be_none(getattr_call_result_59104, None_59105)

        if may_be_59106:

            if more_types_in_union_59107:
                # Runtime conditional SSA (line 79)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setattr(...): (line 79)
            # Processing the call arguments (line 79)
            # Getting the type of 'c' (line 79)
            c_59109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 54), 'c', False)
            # Getting the type of 'a' (line 79)
            a_59110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 57), 'a', False)
            # Getting the type of 'v1' (line 79)
            v1_59111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 60), 'v1', False)
            # Processing the call keyword arguments (line 79)
            kwargs_59112 = {}
            # Getting the type of 'setattr' (line 79)
            setattr_59108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 46), 'setattr', False)
            # Calling setattr(args, kwargs) (line 79)
            setattr_call_result_59113 = invoke(stypy.reporting.localization.Localization(__file__, 79, 46), setattr_59108, *[c_59109, a_59110, v1_59111], **kwargs_59112)
            

            if more_types_in_union_59107:
                # SSA join for if statement (line 79)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_59114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59114)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_59114


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 81, 4, False)
        # Assigning a type to the variable 'self' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config_fc.run.__dict__.__setitem__('stypy_localization', localization)
        config_fc.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config_fc.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        config_fc.run.__dict__.__setitem__('stypy_function_name', 'config_fc.run')
        config_fc.run.__dict__.__setitem__('stypy_param_names_list', [])
        config_fc.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        config_fc.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config_fc.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        config_fc.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        config_fc.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config_fc.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config_fc.run', [], None, None, defaults, varargs, kwargs)

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

        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 81)
        stypy_return_type_59115 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59115)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_59115


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 17, 0, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config_fc.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'config_fc' (line 17)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'config_fc', config_fc)

# Assigning a Str to a Name (line 24):
str_59116 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 18), 'str', 'specify Fortran 77/Fortran 90 compiler information')
# Getting the type of 'config_fc'
config_fc_59117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'config_fc')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), config_fc_59117, 'description', str_59116)

# Assigning a List to a Name (line 26):

# Obtaining an instance of the builtin type 'list' (line 26)
list_59118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 27)
tuple_59119 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 27)
# Adding element type (line 27)
str_59120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'str', 'fcompiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_59119, str_59120)
# Adding element type (line 27)
# Getting the type of 'None' (line 27)
None_59121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_59119, None_59121)
# Adding element type (line 27)
str_59122 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 29), 'str', 'specify Fortran compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 9), tuple_59119, str_59122)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59119)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_59123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)
str_59124 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'str', 'f77exec=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_59123, str_59124)
# Adding element type (line 28)
# Getting the type of 'None' (line 28)
None_59125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_59123, None_59125)
# Adding element type (line 28)
str_59126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 27), 'str', 'specify F77 compiler command')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_59123, str_59126)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59123)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 29)
tuple_59127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 29)
# Adding element type (line 29)
str_59128 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'str', 'f90exec=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_59127, str_59128)
# Adding element type (line 29)
# Getting the type of 'None' (line 29)
None_59129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 21), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_59127, None_59129)
# Adding element type (line 29)
str_59130 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 27), 'str', 'specify F90 compiler command')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 9), tuple_59127, str_59130)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59127)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 30)
tuple_59131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 30)
# Adding element type (line 30)
str_59132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 9), 'str', 'f77flags=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_59131, str_59132)
# Adding element type (line 30)
# Getting the type of 'None' (line 30)
None_59133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_59131, None_59133)
# Adding element type (line 30)
str_59134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 28), 'str', 'specify F77 compiler flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 9), tuple_59131, str_59134)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59131)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 31)
tuple_59135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 31)
# Adding element type (line 31)
str_59136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 9), 'str', 'f90flags=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_59135, str_59136)
# Adding element type (line 31)
# Getting the type of 'None' (line 31)
None_59137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_59135, None_59137)
# Adding element type (line 31)
str_59138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'str', 'specify F90 compiler flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 9), tuple_59135, str_59138)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59135)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 32)
tuple_59139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 32)
# Adding element type (line 32)
str_59140 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 9), 'str', 'opt=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_59139, str_59140)
# Adding element type (line 32)
# Getting the type of 'None' (line 32)
None_59141 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 17), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_59139, None_59141)
# Adding element type (line 32)
str_59142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'str', 'specify optimization flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 9), tuple_59139, str_59142)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59139)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 33)
tuple_59143 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 33)
# Adding element type (line 33)
str_59144 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 9), 'str', 'arch=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_59143, str_59144)
# Adding element type (line 33)
# Getting the type of 'None' (line 33)
None_59145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_59143, None_59145)
# Adding element type (line 33)
str_59146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 24), 'str', 'specify architecture specific optimization flags')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 9), tuple_59143, str_59146)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59143)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 34)
tuple_59147 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 34)
# Adding element type (line 34)
str_59148 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 9), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), tuple_59147, str_59148)
# Adding element type (line 34)
str_59149 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 18), 'str', 'g')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), tuple_59147, str_59149)
# Adding element type (line 34)
str_59150 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 23), 'str', 'compile with debugging information')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 9), tuple_59147, str_59150)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59147)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 35)
tuple_59151 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 35)
# Adding element type (line 35)
str_59152 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 9), 'str', 'noopt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_59151, str_59152)
# Adding element type (line 35)
# Getting the type of 'None' (line 35)
None_59153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 18), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_59151, None_59153)
# Adding element type (line 35)
str_59154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 24), 'str', 'compile without optimization')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 35, 9), tuple_59151, str_59154)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59151)
# Adding element type (line 26)

# Obtaining an instance of the builtin type 'tuple' (line 36)
tuple_59155 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 36)
# Adding element type (line 36)
str_59156 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 9), 'str', 'noarch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_59155, str_59156)
# Adding element type (line 36)
# Getting the type of 'None' (line 36)
None_59157 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_59155, None_59157)
# Adding element type (line 36)
str_59158 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 25), 'str', 'compile without arch-dependent optimization')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 9), tuple_59155, str_59158)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 19), list_59118, tuple_59155)

# Getting the type of 'config_fc'
config_fc_59159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'config_fc')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), config_fc_59159, 'user_options', list_59118)

# Assigning a List to a Name (line 39):

# Obtaining an instance of the builtin type 'list' (line 39)
list_59160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 39)
# Adding element type (line 39)

# Obtaining an instance of the builtin type 'tuple' (line 40)
tuple_59161 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 40)
# Adding element type (line 40)
str_59162 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 9), 'str', 'help-fcompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_59161, str_59162)
# Adding element type (line 40)
# Getting the type of 'None' (line 40)
None_59163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_59161, None_59163)
# Adding element type (line 40)
str_59164 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 33), 'str', 'list available Fortran compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_59161, str_59164)
# Adding element type (line 40)
# Getting the type of 'show_fortran_compilers' (line 41)
show_fortran_compilers_59165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 9), 'show_fortran_compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 9), tuple_59161, show_fortran_compilers_59165)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 19), list_59160, tuple_59161)

# Getting the type of 'config_fc'
config_fc_59166 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'config_fc')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), config_fc_59166, 'help_options', list_59160)

# Assigning a List to a Name (line 44):

# Obtaining an instance of the builtin type 'list' (line 44)
list_59167 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 44)
# Adding element type (line 44)
str_59168 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 23), 'str', 'debug')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), list_59167, str_59168)
# Adding element type (line 44)
str_59169 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 32), 'str', 'noopt')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), list_59167, str_59169)
# Adding element type (line 44)
str_59170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 41), 'str', 'noarch')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 22), list_59167, str_59170)

# Getting the type of 'config_fc'
config_fc_59171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'config_fc')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), config_fc_59171, 'boolean_options', list_59167)
# Declaration of the 'config_cc' class
# Getting the type of 'Command' (line 85)
Command_59172 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 16), 'Command')

class config_cc(Command_59172, ):
    str_59173 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, (-1)), 'str', ' Distutils command to hold user specified options\n    to C/C++ compilers.\n    ')

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 96, 4, False)
        # Assigning a type to the variable 'self' (line 97)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 97, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config_cc.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        config_cc.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config_cc.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        config_cc.initialize_options.__dict__.__setitem__('stypy_function_name', 'config_cc.initialize_options')
        config_cc.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        config_cc.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        config_cc.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config_cc.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        config_cc.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        config_cc.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config_cc.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config_cc.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 97):
        # Getting the type of 'None' (line 97)
        None_59174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 24), 'None')
        # Getting the type of 'self' (line 97)
        self_59175 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 97, 8), 'self')
        # Setting the type of the member 'compiler' of a type (line 97)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 97, 8), self_59175, 'compiler', None_59174)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 96)
        stypy_return_type_59176 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59176)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_59176


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 99, 4, False)
        # Assigning a type to the variable 'self' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config_cc.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        config_cc.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config_cc.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        config_cc.finalize_options.__dict__.__setitem__('stypy_function_name', 'config_cc.finalize_options')
        config_cc.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        config_cc.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        config_cc.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config_cc.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        config_cc.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        config_cc.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config_cc.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config_cc.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to info(...): (line 100)
        # Processing the call arguments (line 100)
        str_59179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 17), 'str', 'unifing config_cc, config, build_clib, build_ext, build commands --compiler options')
        # Processing the call keyword arguments (line 100)
        kwargs_59180 = {}
        # Getting the type of 'log' (line 100)
        log_59177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 100)
        info_59178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 8), log_59177, 'info')
        # Calling info(args, kwargs) (line 100)
        info_call_result_59181 = invoke(stypy.reporting.localization.Localization(__file__, 100, 8), info_59178, *[str_59179], **kwargs_59180)
        
        
        # Assigning a Call to a Name (line 101):
        
        # Call to get_finalized_command(...): (line 101)
        # Processing the call arguments (line 101)
        str_59184 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 48), 'str', 'build_clib')
        # Processing the call keyword arguments (line 101)
        kwargs_59185 = {}
        # Getting the type of 'self' (line 101)
        self_59182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 21), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 101)
        get_finalized_command_59183 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 21), self_59182, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 101)
        get_finalized_command_call_result_59186 = invoke(stypy.reporting.localization.Localization(__file__, 101, 21), get_finalized_command_59183, *[str_59184], **kwargs_59185)
        
        # Assigning a type to the variable 'build_clib' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'build_clib', get_finalized_command_call_result_59186)
        
        # Assigning a Call to a Name (line 102):
        
        # Call to get_finalized_command(...): (line 102)
        # Processing the call arguments (line 102)
        str_59189 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 47), 'str', 'build_ext')
        # Processing the call keyword arguments (line 102)
        kwargs_59190 = {}
        # Getting the type of 'self' (line 102)
        self_59187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 102)
        get_finalized_command_59188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 20), self_59187, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 102)
        get_finalized_command_call_result_59191 = invoke(stypy.reporting.localization.Localization(__file__, 102, 20), get_finalized_command_59188, *[str_59189], **kwargs_59190)
        
        # Assigning a type to the variable 'build_ext' (line 102)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 8), 'build_ext', get_finalized_command_call_result_59191)
        
        # Assigning a Call to a Name (line 103):
        
        # Call to get_finalized_command(...): (line 103)
        # Processing the call arguments (line 103)
        str_59194 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 44), 'str', 'config')
        # Processing the call keyword arguments (line 103)
        kwargs_59195 = {}
        # Getting the type of 'self' (line 103)
        self_59192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 17), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 103)
        get_finalized_command_59193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 17), self_59192, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 103)
        get_finalized_command_call_result_59196 = invoke(stypy.reporting.localization.Localization(__file__, 103, 17), get_finalized_command_59193, *[str_59194], **kwargs_59195)
        
        # Assigning a type to the variable 'config' (line 103)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'config', get_finalized_command_call_result_59196)
        
        # Assigning a Call to a Name (line 104):
        
        # Call to get_finalized_command(...): (line 104)
        # Processing the call arguments (line 104)
        str_59199 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 43), 'str', 'build')
        # Processing the call keyword arguments (line 104)
        kwargs_59200 = {}
        # Getting the type of 'self' (line 104)
        self_59197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 16), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 104)
        get_finalized_command_59198 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 16), self_59197, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 104)
        get_finalized_command_call_result_59201 = invoke(stypy.reporting.localization.Localization(__file__, 104, 16), get_finalized_command_59198, *[str_59199], **kwargs_59200)
        
        # Assigning a type to the variable 'build' (line 104)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 8), 'build', get_finalized_command_call_result_59201)
        
        # Assigning a List to a Name (line 105):
        
        # Obtaining an instance of the builtin type 'list' (line 105)
        list_59202 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 105)
        # Adding element type (line 105)
        # Getting the type of 'self' (line 105)
        self_59203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 20), 'self')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_59202, self_59203)
        # Adding element type (line 105)
        # Getting the type of 'config' (line 105)
        config_59204 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'config')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_59202, config_59204)
        # Adding element type (line 105)
        # Getting the type of 'build_clib' (line 105)
        build_clib_59205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 34), 'build_clib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_59202, build_clib_59205)
        # Adding element type (line 105)
        # Getting the type of 'build_ext' (line 105)
        build_ext_59206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 46), 'build_ext')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_59202, build_ext_59206)
        # Adding element type (line 105)
        # Getting the type of 'build' (line 105)
        build_59207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 57), 'build')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 105, 19), list_59202, build_59207)
        
        # Assigning a type to the variable 'cmd_list' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 8), 'cmd_list', list_59202)
        
        
        # Obtaining an instance of the builtin type 'list' (line 106)
        list_59208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 17), 'list')
        # Adding type elements to the builtin type 'list' instance (line 106)
        # Adding element type (line 106)
        str_59209 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 106, 18), 'str', 'compiler')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 106, 17), list_59208, str_59209)
        
        # Testing the type of a for loop iterable (line 106)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 106, 8), list_59208)
        # Getting the type of the for loop variable (line 106)
        for_loop_var_59210 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 106, 8), list_59208)
        # Assigning a type to the variable 'a' (line 106)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'a', for_loop_var_59210)
        # SSA begins for a for statement (line 106)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a List to a Name (line 107):
        
        # Obtaining an instance of the builtin type 'list' (line 107)
        list_59211 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 107, 16), 'list')
        # Adding type elements to the builtin type 'list' instance (line 107)
        
        # Assigning a type to the variable 'l' (line 107)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'l', list_59211)
        
        # Getting the type of 'cmd_list' (line 108)
        cmd_list_59212 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 21), 'cmd_list')
        # Testing the type of a for loop iterable (line 108)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 108, 12), cmd_list_59212)
        # Getting the type of the for loop variable (line 108)
        for_loop_var_59213 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 108, 12), cmd_list_59212)
        # Assigning a type to the variable 'c' (line 108)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'c', for_loop_var_59213)
        # SSA begins for a for statement (line 108)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 109):
        
        # Call to getattr(...): (line 109)
        # Processing the call arguments (line 109)
        # Getting the type of 'c' (line 109)
        c_59215 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 28), 'c', False)
        # Getting the type of 'a' (line 109)
        a_59216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 31), 'a', False)
        # Processing the call keyword arguments (line 109)
        kwargs_59217 = {}
        # Getting the type of 'getattr' (line 109)
        getattr_59214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 20), 'getattr', False)
        # Calling getattr(args, kwargs) (line 109)
        getattr_call_result_59218 = invoke(stypy.reporting.localization.Localization(__file__, 109, 20), getattr_59214, *[c_59215, a_59216], **kwargs_59217)
        
        # Assigning a type to the variable 'v' (line 109)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'v', getattr_call_result_59218)
        
        # Type idiom detected: calculating its left and rigth part (line 110)
        # Getting the type of 'v' (line 110)
        v_59219 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 16), 'v')
        # Getting the type of 'None' (line 110)
        None_59220 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 28), 'None')
        
        (may_be_59221, more_types_in_union_59222) = may_not_be_none(v_59219, None_59220)

        if may_be_59221:

            if more_types_in_union_59222:
                # Runtime conditional SSA (line 110)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Type idiom detected: calculating its left and rigth part (line 111)
            # Getting the type of 'str' (line 111)
            str_59223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 41), 'str')
            # Getting the type of 'v' (line 111)
            v_59224 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'v')
            
            (may_be_59225, more_types_in_union_59226) = may_not_be_subtype(str_59223, v_59224)

            if may_be_59225:

                if more_types_in_union_59226:
                    # Runtime conditional SSA (line 111)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Assigning a type to the variable 'v' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 20), 'v', remove_subtype_from_union(v_59224, str))
                
                # Assigning a Attribute to a Name (line 111):
                # Getting the type of 'v' (line 111)
                v_59227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 51), 'v')
                # Obtaining the member 'compiler_type' of a type (line 111)
                compiler_type_59228 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 51), v_59227, 'compiler_type')
                # Assigning a type to the variable 'v' (line 111)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 47), 'v', compiler_type_59228)

                if more_types_in_union_59226:
                    # SSA join for if statement (line 111)
                    module_type_store = module_type_store.join_ssa_context()


            
            
            
            # Getting the type of 'v' (line 112)
            v_59229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 23), 'v')
            # Getting the type of 'l' (line 112)
            l_59230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 32), 'l')
            # Applying the binary operator 'notin' (line 112)
            result_contains_59231 = python_operator(stypy.reporting.localization.Localization(__file__, 112, 23), 'notin', v_59229, l_59230)
            
            # Testing the type of an if condition (line 112)
            if_condition_59232 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 20), result_contains_59231)
            # Assigning a type to the variable 'if_condition_59232' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'if_condition_59232', if_condition_59232)
            # SSA begins for if statement (line 112)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to append(...): (line 112)
            # Processing the call arguments (line 112)
            # Getting the type of 'v' (line 112)
            v_59235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 44), 'v', False)
            # Processing the call keyword arguments (line 112)
            kwargs_59236 = {}
            # Getting the type of 'l' (line 112)
            l_59233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'l', False)
            # Obtaining the member 'append' of a type (line 112)
            append_59234 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 35), l_59233, 'append')
            # Calling append(args, kwargs) (line 112)
            append_call_result_59237 = invoke(stypy.reporting.localization.Localization(__file__, 112, 35), append_59234, *[v_59235], **kwargs_59236)
            
            # SSA join for if statement (line 112)
            module_type_store = module_type_store.join_ssa_context()
            

            if more_types_in_union_59222:
                # SSA join for if statement (line 110)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'l' (line 113)
        l_59238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 19), 'l')
        # Applying the 'not' unary operator (line 113)
        result_not__59239 = python_operator(stypy.reporting.localization.Localization(__file__, 113, 15), 'not', l_59238)
        
        # Testing the type of an if condition (line 113)
        if_condition_59240 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 113, 12), result_not__59239)
        # Assigning a type to the variable 'if_condition_59240' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 12), 'if_condition_59240', if_condition_59240)
        # SSA begins for if statement (line 113)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 113):
        # Getting the type of 'None' (line 113)
        None_59241 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 27), 'None')
        # Assigning a type to the variable 'v1' (line 113)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 22), 'v1', None_59241)
        # SSA branch for the else part of an if statement (line 113)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Subscript to a Name (line 114):
        
        # Obtaining the type of the subscript
        int_59242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 114, 25), 'int')
        # Getting the type of 'l' (line 114)
        l_59243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 23), 'l')
        # Obtaining the member '__getitem__' of a type (line 114)
        getitem___59244 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 23), l_59243, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 114)
        subscript_call_result_59245 = invoke(stypy.reporting.localization.Localization(__file__, 114, 23), getitem___59244, int_59242)
        
        # Assigning a type to the variable 'v1' (line 114)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 18), 'v1', subscript_call_result_59245)
        # SSA join for if statement (line 113)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        
        # Call to len(...): (line 115)
        # Processing the call arguments (line 115)
        # Getting the type of 'l' (line 115)
        l_59247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 19), 'l', False)
        # Processing the call keyword arguments (line 115)
        kwargs_59248 = {}
        # Getting the type of 'len' (line 115)
        len_59246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 15), 'len', False)
        # Calling len(args, kwargs) (line 115)
        len_call_result_59249 = invoke(stypy.reporting.localization.Localization(__file__, 115, 15), len_59246, *[l_59247], **kwargs_59248)
        
        int_59250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 115, 22), 'int')
        # Applying the binary operator '>' (line 115)
        result_gt_59251 = python_operator(stypy.reporting.localization.Localization(__file__, 115, 15), '>', len_call_result_59249, int_59250)
        
        # Testing the type of an if condition (line 115)
        if_condition_59252 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 115, 12), result_gt_59251)
        # Assigning a type to the variable 'if_condition_59252' (line 115)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'if_condition_59252', if_condition_59252)
        # SSA begins for if statement (line 115)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to warn(...): (line 116)
        # Processing the call arguments (line 116)
        str_59255 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 25), 'str', '  commands have different --%s options: %s, using first in list as default')
        
        # Obtaining an instance of the builtin type 'tuple' (line 117)
        tuple_59256 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 117, 63), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 117)
        # Adding element type (line 117)
        # Getting the type of 'a' (line 117)
        a_59257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 63), 'a', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 63), tuple_59256, a_59257)
        # Adding element type (line 117)
        # Getting the type of 'l' (line 117)
        l_59258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 66), 'l', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 117, 63), tuple_59256, l_59258)
        
        # Applying the binary operator '%' (line 116)
        result_mod_59259 = python_operator(stypy.reporting.localization.Localization(__file__, 116, 25), '%', str_59255, tuple_59256)
        
        # Processing the call keyword arguments (line 116)
        kwargs_59260 = {}
        # Getting the type of 'log' (line 116)
        log_59253 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 16), 'log', False)
        # Obtaining the member 'warn' of a type (line 116)
        warn_59254 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 16), log_59253, 'warn')
        # Calling warn(args, kwargs) (line 116)
        warn_call_result_59261 = invoke(stypy.reporting.localization.Localization(__file__, 116, 16), warn_59254, *[result_mod_59259], **kwargs_59260)
        
        # SSA join for if statement (line 115)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'v1' (line 118)
        v1_59262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 118, 15), 'v1')
        # Testing the type of an if condition (line 118)
        if_condition_59263 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 118, 12), v1_59262)
        # Assigning a type to the variable 'if_condition_59263' (line 118)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 118, 12), 'if_condition_59263', if_condition_59263)
        # SSA begins for if statement (line 118)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'cmd_list' (line 119)
        cmd_list_59264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'cmd_list')
        # Testing the type of a for loop iterable (line 119)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 119, 16), cmd_list_59264)
        # Getting the type of the for loop variable (line 119)
        for_loop_var_59265 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 119, 16), cmd_list_59264)
        # Assigning a type to the variable 'c' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'c', for_loop_var_59265)
        # SSA begins for a for statement (line 119)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Type idiom detected: calculating its left and rigth part (line 120)
        
        # Call to getattr(...): (line 120)
        # Processing the call arguments (line 120)
        # Getting the type of 'c' (line 120)
        c_59267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 31), 'c', False)
        # Getting the type of 'a' (line 120)
        a_59268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 34), 'a', False)
        # Processing the call keyword arguments (line 120)
        kwargs_59269 = {}
        # Getting the type of 'getattr' (line 120)
        getattr_59266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 23), 'getattr', False)
        # Calling getattr(args, kwargs) (line 120)
        getattr_call_result_59270 = invoke(stypy.reporting.localization.Localization(__file__, 120, 23), getattr_59266, *[c_59267, a_59268], **kwargs_59269)
        
        # Getting the type of 'None' (line 120)
        None_59271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 40), 'None')
        
        (may_be_59272, more_types_in_union_59273) = may_be_none(getattr_call_result_59270, None_59271)

        if may_be_59272:

            if more_types_in_union_59273:
                # Runtime conditional SSA (line 120)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Call to setattr(...): (line 120)
            # Processing the call arguments (line 120)
            # Getting the type of 'c' (line 120)
            c_59275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 54), 'c', False)
            # Getting the type of 'a' (line 120)
            a_59276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 57), 'a', False)
            # Getting the type of 'v1' (line 120)
            v1_59277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 60), 'v1', False)
            # Processing the call keyword arguments (line 120)
            kwargs_59278 = {}
            # Getting the type of 'setattr' (line 120)
            setattr_59274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 46), 'setattr', False)
            # Calling setattr(args, kwargs) (line 120)
            setattr_call_result_59279 = invoke(stypy.reporting.localization.Localization(__file__, 120, 46), setattr_59274, *[c_59275, a_59276, v1_59277], **kwargs_59278)
            

            if more_types_in_union_59273:
                # SSA join for if statement (line 120)
                module_type_store = module_type_store.join_ssa_context()


        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 118)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Assigning a type to the variable 'stypy_return_type' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 99)
        stypy_return_type_59280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_59280


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 123, 4, False)
        # Assigning a type to the variable 'self' (line 124)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        config_cc.run.__dict__.__setitem__('stypy_localization', localization)
        config_cc.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        config_cc.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        config_cc.run.__dict__.__setitem__('stypy_function_name', 'config_cc.run')
        config_cc.run.__dict__.__setitem__('stypy_param_names_list', [])
        config_cc.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        config_cc.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        config_cc.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        config_cc.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        config_cc.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        config_cc.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config_cc.run', [], None, None, defaults, varargs, kwargs)

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

        # Assigning a type to the variable 'stypy_return_type' (line 125)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 125, 8), 'stypy_return_type', types.NoneType)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 123)
        stypy_return_type_59281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59281)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_59281


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 85, 0, False)
        # Assigning a type to the variable 'self' (line 86)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'config_cc.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'config_cc' (line 85)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 0), 'config_cc', config_cc)

# Assigning a Str to a Name (line 90):
str_59282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 90, 18), 'str', 'specify C/C++ compiler information')
# Getting the type of 'config_cc'
config_cc_59283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'config_cc')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), config_cc_59283, 'description', str_59282)

# Assigning a List to a Name (line 92):

# Obtaining an instance of the builtin type 'list' (line 92)
list_59284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 92)
# Adding element type (line 92)

# Obtaining an instance of the builtin type 'tuple' (line 93)
tuple_59285 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 93)
# Adding element type (line 93)
str_59286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 9), 'str', 'compiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 9), tuple_59285, str_59286)
# Adding element type (line 93)
# Getting the type of 'None' (line 93)
None_59287 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 22), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 9), tuple_59285, None_59287)
# Adding element type (line 93)
str_59288 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 93, 28), 'str', 'specify C/C++ compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 93, 9), tuple_59285, str_59288)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 92, 19), list_59284, tuple_59285)

# Getting the type of 'config_cc'
config_cc_59289 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'config_cc')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), config_cc_59289, 'user_options', list_59284)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
