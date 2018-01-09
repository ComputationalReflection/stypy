
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.clean
2: 
3: Implements the Distutils 'clean' command.'''
4: 
5: # contributed by Bastian Kleineidam <calvin@cs.uni-sb.de>, added 2000-03-18
6: 
7: __revision__ = "$Id$"
8: 
9: import os
10: from distutils.core import Command
11: from distutils.dir_util import remove_tree
12: from distutils import log
13: 
14: class clean(Command):
15: 
16:     description = "clean up temporary files from 'build' command"
17:     user_options = [
18:         ('build-base=', 'b',
19:          "base build directory (default: 'build.build-base')"),
20:         ('build-lib=', None,
21:          "build directory for all modules (default: 'build.build-lib')"),
22:         ('build-temp=', 't',
23:          "temporary build directory (default: 'build.build-temp')"),
24:         ('build-scripts=', None,
25:          "build directory for scripts (default: 'build.build-scripts')"),
26:         ('bdist-base=', None,
27:          "temporary directory for built distributions"),
28:         ('all', 'a',
29:          "remove all build output, not just temporary by-products")
30:     ]
31: 
32:     boolean_options = ['all']
33: 
34:     def initialize_options(self):
35:         self.build_base = None
36:         self.build_lib = None
37:         self.build_temp = None
38:         self.build_scripts = None
39:         self.bdist_base = None
40:         self.all = None
41: 
42:     def finalize_options(self):
43:         self.set_undefined_options('build',
44:                                    ('build_base', 'build_base'),
45:                                    ('build_lib', 'build_lib'),
46:                                    ('build_scripts', 'build_scripts'),
47:                                    ('build_temp', 'build_temp'))
48:         self.set_undefined_options('bdist',
49:                                    ('bdist_base', 'bdist_base'))
50: 
51:     def run(self):
52:         # remove the build/temp.<plat> directory (unless it's already
53:         # gone)
54:         if os.path.exists(self.build_temp):
55:             remove_tree(self.build_temp, dry_run=self.dry_run)
56:         else:
57:             log.debug("'%s' does not exist -- can't clean it",
58:                       self.build_temp)
59: 
60:         if self.all:
61:             # remove build directories
62:             for directory in (self.build_lib,
63:                               self.bdist_base,
64:                               self.build_scripts):
65:                 if os.path.exists(directory):
66:                     remove_tree(directory, dry_run=self.dry_run)
67:                 else:
68:                     log.warn("'%s' does not exist -- can't clean it",
69:                              directory)
70: 
71:         # just for the heck of it, try to remove the base build directory:
72:         # we might have emptied it right now, but if not we don't care
73:         if not self.dry_run:
74:             try:
75:                 os.rmdir(self.build_base)
76:                 log.info("removing '%s'", self.build_base)
77:             except OSError:
78:                 pass
79: 
80: # class clean
81: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_21296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', "distutils.command.clean\n\nImplements the Distutils 'clean' command.")

# Assigning a Str to a Name (line 7):
str_21297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), '__revision__', str_21297)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# 'import os' statement (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.core import Command' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_21298 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.core')

if (type(import_21298) is not StypyTypeError):

    if (import_21298 != 'pyd_module'):
        __import__(import_21298)
        sys_modules_21299 = sys.modules[import_21298]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.core', sys_modules_21299.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_21299, sys_modules_21299.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.core', import_21298)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 11, 0))

# 'from distutils.dir_util import remove_tree' statement (line 11)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_21300 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dir_util')

if (type(import_21300) is not StypyTypeError):

    if (import_21300 != 'pyd_module'):
        __import__(import_21300)
        sys_modules_21301 = sys.modules[import_21300]
        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dir_util', sys_modules_21301.module_type_store, module_type_store, ['remove_tree'])
        nest_module(stypy.reporting.localization.Localization(__file__, 11, 0), __file__, sys_modules_21301, sys_modules_21301.module_type_store, module_type_store)
    else:
        from distutils.dir_util import remove_tree

        import_from_module(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dir_util', None, module_type_store, ['remove_tree'], [remove_tree])

else:
    # Assigning a type to the variable 'distutils.dir_util' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'distutils.dir_util', import_21300)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 12, 0))

# 'from distutils import log' statement (line 12)
try:
    from distutils import log

except:
    log = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 12, 0), 'distutils', None, module_type_store, ['log'], [log])

# Declaration of the 'clean' class
# Getting the type of 'Command' (line 14)
Command_21302 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 12), 'Command')

class clean(Command_21302, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        clean.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        clean.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        clean.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        clean.initialize_options.__dict__.__setitem__('stypy_function_name', 'clean.initialize_options')
        clean.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        clean.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        clean.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        clean.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        clean.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        clean.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        clean.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'clean.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 35):
        # Getting the type of 'None' (line 35)
        None_21303 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 26), 'None')
        # Getting the type of 'self' (line 35)
        self_21304 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'self')
        # Setting the type of the member 'build_base' of a type (line 35)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 8), self_21304, 'build_base', None_21303)
        
        # Assigning a Name to a Attribute (line 36):
        # Getting the type of 'None' (line 36)
        None_21305 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 25), 'None')
        # Getting the type of 'self' (line 36)
        self_21306 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'self')
        # Setting the type of the member 'build_lib' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 8), self_21306, 'build_lib', None_21305)
        
        # Assigning a Name to a Attribute (line 37):
        # Getting the type of 'None' (line 37)
        None_21307 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 26), 'None')
        # Getting the type of 'self' (line 37)
        self_21308 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'self')
        # Setting the type of the member 'build_temp' of a type (line 37)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), self_21308, 'build_temp', None_21307)
        
        # Assigning a Name to a Attribute (line 38):
        # Getting the type of 'None' (line 38)
        None_21309 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 29), 'None')
        # Getting the type of 'self' (line 38)
        self_21310 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'self')
        # Setting the type of the member 'build_scripts' of a type (line 38)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 8), self_21310, 'build_scripts', None_21309)
        
        # Assigning a Name to a Attribute (line 39):
        # Getting the type of 'None' (line 39)
        None_21311 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 26), 'None')
        # Getting the type of 'self' (line 39)
        self_21312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'self')
        # Setting the type of the member 'bdist_base' of a type (line 39)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 8), self_21312, 'bdist_base', None_21311)
        
        # Assigning a Name to a Attribute (line 40):
        # Getting the type of 'None' (line 40)
        None_21313 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 19), 'None')
        # Getting the type of 'self' (line 40)
        self_21314 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self')
        # Setting the type of the member 'all' of a type (line 40)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_21314, 'all', None_21313)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_21315 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21315)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_21315


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 42, 4, False)
        # Assigning a type to the variable 'self' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        clean.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        clean.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        clean.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        clean.finalize_options.__dict__.__setitem__('stypy_function_name', 'clean.finalize_options')
        clean.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        clean.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        clean.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        clean.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        clean.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        clean.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        clean.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'clean.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 43)
        # Processing the call arguments (line 43)
        str_21318 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 35), 'str', 'build')
        
        # Obtaining an instance of the builtin type 'tuple' (line 44)
        tuple_21319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 44)
        # Adding element type (line 44)
        str_21320 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 36), 'str', 'build_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 36), tuple_21319, str_21320)
        # Adding element type (line 44)
        str_21321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 50), 'str', 'build_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 44, 36), tuple_21319, str_21321)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 45)
        tuple_21322 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 45)
        # Adding element type (line 45)
        str_21323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 36), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 36), tuple_21322, str_21323)
        # Adding element type (line 45)
        str_21324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 49), 'str', 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 45, 36), tuple_21322, str_21324)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 46)
        tuple_21325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 46)
        # Adding element type (line 46)
        str_21326 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 36), 'str', 'build_scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 36), tuple_21325, str_21326)
        # Adding element type (line 46)
        str_21327 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 53), 'str', 'build_scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 46, 36), tuple_21325, str_21327)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 47)
        tuple_21328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 47)
        # Adding element type (line 47)
        str_21329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 36), 'str', 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 36), tuple_21328, str_21329)
        # Adding element type (line 47)
        str_21330 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 50), 'str', 'build_temp')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 47, 36), tuple_21328, str_21330)
        
        # Processing the call keyword arguments (line 43)
        kwargs_21331 = {}
        # Getting the type of 'self' (line 43)
        self_21316 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 43)
        set_undefined_options_21317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), self_21316, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 43)
        set_undefined_options_call_result_21332 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), set_undefined_options_21317, *[str_21318, tuple_21319, tuple_21322, tuple_21325, tuple_21328], **kwargs_21331)
        
        
        # Call to set_undefined_options(...): (line 48)
        # Processing the call arguments (line 48)
        str_21335 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, 35), 'str', 'bdist')
        
        # Obtaining an instance of the builtin type 'tuple' (line 49)
        tuple_21336 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 49)
        # Adding element type (line 49)
        str_21337 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 36), 'str', 'bdist_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 36), tuple_21336, str_21337)
        # Adding element type (line 49)
        str_21338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 49, 50), 'str', 'bdist_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 49, 36), tuple_21336, str_21338)
        
        # Processing the call keyword arguments (line 48)
        kwargs_21339 = {}
        # Getting the type of 'self' (line 48)
        self_21333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 48)
        set_undefined_options_21334 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 8), self_21333, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 48)
        set_undefined_options_call_result_21340 = invoke(stypy.reporting.localization.Localization(__file__, 48, 8), set_undefined_options_21334, *[str_21335, tuple_21336], **kwargs_21339)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 42)
        stypy_return_type_21341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21341)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_21341


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 51, 4, False)
        # Assigning a type to the variable 'self' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        clean.run.__dict__.__setitem__('stypy_localization', localization)
        clean.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        clean.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        clean.run.__dict__.__setitem__('stypy_function_name', 'clean.run')
        clean.run.__dict__.__setitem__('stypy_param_names_list', [])
        clean.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        clean.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        clean.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        clean.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        clean.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        clean.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'clean.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Call to exists(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_21345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 26), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 54)
        build_temp_21346 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 26), self_21345, 'build_temp')
        # Processing the call keyword arguments (line 54)
        kwargs_21347 = {}
        # Getting the type of 'os' (line 54)
        os_21342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 54)
        path_21343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), os_21342, 'path')
        # Obtaining the member 'exists' of a type (line 54)
        exists_21344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 11), path_21343, 'exists')
        # Calling exists(args, kwargs) (line 54)
        exists_call_result_21348 = invoke(stypy.reporting.localization.Localization(__file__, 54, 11), exists_21344, *[build_temp_21346], **kwargs_21347)
        
        # Testing the type of an if condition (line 54)
        if_condition_21349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 8), exists_call_result_21348)
        # Assigning a type to the variable 'if_condition_21349' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 8), 'if_condition_21349', if_condition_21349)
        # SSA begins for if statement (line 54)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove_tree(...): (line 55)
        # Processing the call arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_21351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 55)
        build_temp_21352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), self_21351, 'build_temp')
        # Processing the call keyword arguments (line 55)
        # Getting the type of 'self' (line 55)
        self_21353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 49), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 55)
        dry_run_21354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 49), self_21353, 'dry_run')
        keyword_21355 = dry_run_21354
        kwargs_21356 = {'dry_run': keyword_21355}
        # Getting the type of 'remove_tree' (line 55)
        remove_tree_21350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 12), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 55)
        remove_tree_call_result_21357 = invoke(stypy.reporting.localization.Localization(__file__, 55, 12), remove_tree_21350, *[build_temp_21352], **kwargs_21356)
        
        # SSA branch for the else part of an if statement (line 54)
        module_type_store.open_ssa_branch('else')
        
        # Call to debug(...): (line 57)
        # Processing the call arguments (line 57)
        str_21360 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, 22), 'str', "'%s' does not exist -- can't clean it")
        # Getting the type of 'self' (line 58)
        self_21361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 22), 'self', False)
        # Obtaining the member 'build_temp' of a type (line 58)
        build_temp_21362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 22), self_21361, 'build_temp')
        # Processing the call keyword arguments (line 57)
        kwargs_21363 = {}
        # Getting the type of 'log' (line 57)
        log_21358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 12), 'log', False)
        # Obtaining the member 'debug' of a type (line 57)
        debug_21359 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 57, 12), log_21358, 'debug')
        # Calling debug(args, kwargs) (line 57)
        debug_call_result_21364 = invoke(stypy.reporting.localization.Localization(__file__, 57, 12), debug_21359, *[str_21360, build_temp_21362], **kwargs_21363)
        
        # SSA join for if statement (line 54)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 60)
        self_21365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 'self')
        # Obtaining the member 'all' of a type (line 60)
        all_21366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), self_21365, 'all')
        # Testing the type of an if condition (line 60)
        if_condition_21367 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 8), all_21366)
        # Assigning a type to the variable 'if_condition_21367' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'if_condition_21367', if_condition_21367)
        # SSA begins for if statement (line 60)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 62)
        tuple_21368 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 62)
        # Adding element type (line 62)
        # Getting the type of 'self' (line 62)
        self_21369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 30), 'self')
        # Obtaining the member 'build_lib' of a type (line 62)
        build_lib_21370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 30), self_21369, 'build_lib')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 30), tuple_21368, build_lib_21370)
        # Adding element type (line 62)
        # Getting the type of 'self' (line 63)
        self_21371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 30), 'self')
        # Obtaining the member 'bdist_base' of a type (line 63)
        bdist_base_21372 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 30), self_21371, 'bdist_base')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 30), tuple_21368, bdist_base_21372)
        # Adding element type (line 62)
        # Getting the type of 'self' (line 64)
        self_21373 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 30), 'self')
        # Obtaining the member 'build_scripts' of a type (line 64)
        build_scripts_21374 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 30), self_21373, 'build_scripts')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 30), tuple_21368, build_scripts_21374)
        
        # Testing the type of a for loop iterable (line 62)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 62, 12), tuple_21368)
        # Getting the type of the for loop variable (line 62)
        for_loop_var_21375 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 62, 12), tuple_21368)
        # Assigning a type to the variable 'directory' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'directory', for_loop_var_21375)
        # SSA begins for a for statement (line 62)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to exists(...): (line 65)
        # Processing the call arguments (line 65)
        # Getting the type of 'directory' (line 65)
        directory_21379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 34), 'directory', False)
        # Processing the call keyword arguments (line 65)
        kwargs_21380 = {}
        # Getting the type of 'os' (line 65)
        os_21376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'os', False)
        # Obtaining the member 'path' of a type (line 65)
        path_21377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), os_21376, 'path')
        # Obtaining the member 'exists' of a type (line 65)
        exists_21378 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 65, 19), path_21377, 'exists')
        # Calling exists(args, kwargs) (line 65)
        exists_call_result_21381 = invoke(stypy.reporting.localization.Localization(__file__, 65, 19), exists_21378, *[directory_21379], **kwargs_21380)
        
        # Testing the type of an if condition (line 65)
        if_condition_21382 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 65, 16), exists_call_result_21381)
        # Assigning a type to the variable 'if_condition_21382' (line 65)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 16), 'if_condition_21382', if_condition_21382)
        # SSA begins for if statement (line 65)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove_tree(...): (line 66)
        # Processing the call arguments (line 66)
        # Getting the type of 'directory' (line 66)
        directory_21384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'directory', False)
        # Processing the call keyword arguments (line 66)
        # Getting the type of 'self' (line 66)
        self_21385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 51), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 66)
        dry_run_21386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 66, 51), self_21385, 'dry_run')
        keyword_21387 = dry_run_21386
        kwargs_21388 = {'dry_run': keyword_21387}
        # Getting the type of 'remove_tree' (line 66)
        remove_tree_21383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 20), 'remove_tree', False)
        # Calling remove_tree(args, kwargs) (line 66)
        remove_tree_call_result_21389 = invoke(stypy.reporting.localization.Localization(__file__, 66, 20), remove_tree_21383, *[directory_21384], **kwargs_21388)
        
        # SSA branch for the else part of an if statement (line 65)
        module_type_store.open_ssa_branch('else')
        
        # Call to warn(...): (line 68)
        # Processing the call arguments (line 68)
        str_21392 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 29), 'str', "'%s' does not exist -- can't clean it")
        # Getting the type of 'directory' (line 69)
        directory_21393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 29), 'directory', False)
        # Processing the call keyword arguments (line 68)
        kwargs_21394 = {}
        # Getting the type of 'log' (line 68)
        log_21390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 20), 'log', False)
        # Obtaining the member 'warn' of a type (line 68)
        warn_21391 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 20), log_21390, 'warn')
        # Calling warn(args, kwargs) (line 68)
        warn_call_result_21395 = invoke(stypy.reporting.localization.Localization(__file__, 68, 20), warn_21391, *[str_21392, directory_21393], **kwargs_21394)
        
        # SSA join for if statement (line 65)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 60)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'self' (line 73)
        self_21396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'self')
        # Obtaining the member 'dry_run' of a type (line 73)
        dry_run_21397 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), self_21396, 'dry_run')
        # Applying the 'not' unary operator (line 73)
        result_not__21398 = python_operator(stypy.reporting.localization.Localization(__file__, 73, 11), 'not', dry_run_21397)
        
        # Testing the type of an if condition (line 73)
        if_condition_21399 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 8), result_not__21398)
        # Assigning a type to the variable 'if_condition_21399' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 8), 'if_condition_21399', if_condition_21399)
        # SSA begins for if statement (line 73)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 74)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Call to rmdir(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'self' (line 75)
        self_21402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 25), 'self', False)
        # Obtaining the member 'build_base' of a type (line 75)
        build_base_21403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 25), self_21402, 'build_base')
        # Processing the call keyword arguments (line 75)
        kwargs_21404 = {}
        # Getting the type of 'os' (line 75)
        os_21400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'os', False)
        # Obtaining the member 'rmdir' of a type (line 75)
        rmdir_21401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), os_21400, 'rmdir')
        # Calling rmdir(args, kwargs) (line 75)
        rmdir_call_result_21405 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), rmdir_21401, *[build_base_21403], **kwargs_21404)
        
        
        # Call to info(...): (line 76)
        # Processing the call arguments (line 76)
        str_21408 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 76, 25), 'str', "removing '%s'")
        # Getting the type of 'self' (line 76)
        self_21409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 42), 'self', False)
        # Obtaining the member 'build_base' of a type (line 76)
        build_base_21410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 42), self_21409, 'build_base')
        # Processing the call keyword arguments (line 76)
        kwargs_21411 = {}
        # Getting the type of 'log' (line 76)
        log_21406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 76)
        info_21407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 16), log_21406, 'info')
        # Calling info(args, kwargs) (line 76)
        info_call_result_21412 = invoke(stypy.reporting.localization.Localization(__file__, 76, 16), info_21407, *[str_21408, build_base_21410], **kwargs_21411)
        
        # SSA branch for the except part of a try statement (line 74)
        # SSA branch for the except 'OSError' branch of a try statement (line 74)
        module_type_store.open_ssa_branch('except')
        pass
        # SSA join for try-except statement (line 74)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 73)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 51)
        stypy_return_type_21413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_21413)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_21413


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 14, 0, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'clean.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'clean' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'clean', clean)

# Assigning a Str to a Name (line 16):
str_21414 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 18), 'str', "clean up temporary files from 'build' command")
# Getting the type of 'clean'
clean_21415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'clean')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), clean_21415, 'description', str_21414)

# Assigning a List to a Name (line 17):

# Obtaining an instance of the builtin type 'list' (line 17)
list_21416 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_21417 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
str_21418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'str', 'build-base=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 9), tuple_21417, str_21418)
# Adding element type (line 18)
str_21419 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 24), 'str', 'b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 9), tuple_21417, str_21419)
# Adding element type (line 18)
str_21420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'str', "base build directory (default: 'build.build-base')")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 9), tuple_21417, str_21420)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), list_21416, tuple_21417)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'tuple' (line 20)
tuple_21421 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 20)
# Adding element type (line 20)
str_21422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'str', 'build-lib=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), tuple_21421, str_21422)
# Adding element type (line 20)
# Getting the type of 'None' (line 20)
None_21423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), tuple_21421, None_21423)
# Adding element type (line 20)
str_21424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 9), 'str', "build directory for all modules (default: 'build.build-lib')")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 9), tuple_21421, str_21424)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), list_21416, tuple_21421)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'tuple' (line 22)
tuple_21425 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 22)
# Adding element type (line 22)
str_21426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 9), 'str', 'build-temp=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_21425, str_21426)
# Adding element type (line 22)
str_21427 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 24), 'str', 't')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_21425, str_21427)
# Adding element type (line 22)
str_21428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 9), 'str', "temporary build directory (default: 'build.build-temp')")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 9), tuple_21425, str_21428)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), list_21416, tuple_21425)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_21429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_21430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'str', 'build-scripts=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_21429, str_21430)
# Adding element type (line 24)
# Getting the type of 'None' (line 24)
None_21431 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_21429, None_21431)
# Adding element type (line 24)
str_21432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 9), 'str', "build directory for scripts (default: 'build.build-scripts')")
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_21429, str_21432)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), list_21416, tuple_21429)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'tuple' (line 26)
tuple_21433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 26)
# Adding element type (line 26)
str_21434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 9), 'str', 'bdist-base=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_21433, str_21434)
# Adding element type (line 26)
# Getting the type of 'None' (line 26)
None_21435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_21433, None_21435)
# Adding element type (line 26)
str_21436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 9), 'str', 'temporary directory for built distributions')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 9), tuple_21433, str_21436)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), list_21416, tuple_21433)
# Adding element type (line 17)

# Obtaining an instance of the builtin type 'tuple' (line 28)
tuple_21437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 28)
# Adding element type (line 28)
str_21438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 9), 'str', 'all')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_21437, str_21438)
# Adding element type (line 28)
str_21439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'str', 'a')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_21437, str_21439)
# Adding element type (line 28)
str_21440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 9), 'str', 'remove all build output, not just temporary by-products')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 9), tuple_21437, str_21440)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 19), list_21416, tuple_21437)

# Getting the type of 'clean'
clean_21441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'clean')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), clean_21441, 'user_options', list_21416)

# Assigning a List to a Name (line 32):

# Obtaining an instance of the builtin type 'list' (line 32)
list_21442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 32)
# Adding element type (line 32)
str_21443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 23), 'str', 'all')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 22), list_21442, str_21443)

# Getting the type of 'clean'
clean_21444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'clean')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), clean_21444, 'boolean_options', list_21442)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
