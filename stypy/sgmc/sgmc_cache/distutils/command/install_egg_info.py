
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.install_egg_info
2: 
3: Implements the Distutils 'install_egg_info' command, for installing
4: a package's PKG-INFO metadata.'''
5: 
6: 
7: from distutils.cmd import Command
8: from distutils import log, dir_util
9: import os, sys, re
10: 
11: class install_egg_info(Command):
12:     '''Install an .egg-info file for the package'''
13: 
14:     description = "Install package's PKG-INFO metadata as an .egg-info file"
15:     user_options = [
16:         ('install-dir=', 'd', "directory to install to"),
17:     ]
18: 
19:     def initialize_options(self):
20:         self.install_dir = None
21: 
22:     def finalize_options(self):
23:         self.set_undefined_options('install_lib',('install_dir','install_dir'))
24:         basename = "%s-%s-py%s.egg-info" % (
25:             to_filename(safe_name(self.distribution.get_name())),
26:             to_filename(safe_version(self.distribution.get_version())),
27:             sys.version[:3]
28:         )
29:         self.target = os.path.join(self.install_dir, basename)
30:         self.outputs = [self.target]
31: 
32:     def run(self):
33:         target = self.target
34:         if os.path.isdir(target) and not os.path.islink(target):
35:             dir_util.remove_tree(target, dry_run=self.dry_run)
36:         elif os.path.exists(target):
37:             self.execute(os.unlink,(self.target,),"Removing "+target)
38:         elif not os.path.isdir(self.install_dir):
39:             self.execute(os.makedirs, (self.install_dir,),
40:                          "Creating "+self.install_dir)
41:         log.info("Writing %s", target)
42:         if not self.dry_run:
43:             f = open(target, 'w')
44:             self.distribution.metadata.write_pkg_file(f)
45:             f.close()
46: 
47:     def get_outputs(self):
48:         return self.outputs
49: 
50: 
51: # The following routines are taken from setuptools' pkg_resources module and
52: # can be replaced by importing them from pkg_resources once it is included
53: # in the stdlib.
54: 
55: def safe_name(name):
56:     '''Convert an arbitrary string to a standard distribution name
57: 
58:     Any runs of non-alphanumeric/. characters are replaced with a single '-'.
59:     '''
60:     return re.sub('[^A-Za-z0-9.]+', '-', name)
61: 
62: 
63: def safe_version(version):
64:     '''Convert an arbitrary string to a standard version string
65: 
66:     Spaces become dots, and all other non-alphanumeric characters become
67:     dashes, with runs of multiple dashes condensed to a single dash.
68:     '''
69:     version = version.replace(' ','.')
70:     return re.sub('[^A-Za-z0-9.]+', '-', version)
71: 
72: 
73: def to_filename(name):
74:     '''Convert a project or version name to its filename-escaped form
75: 
76:     Any '-' characters are currently replaced with '_'.
77:     '''
78:     return name.replace('-','_')
79: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_23854 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.install_egg_info\n\nImplements the Distutils 'install_egg_info' command, for installing\na package's PKG-INFO metadata.")
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from distutils.cmd import Command' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_23855 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.cmd')

if (type(import_23855) is not StypyTypeError):

    if (import_23855 != 'pyd_module'):
        __import__(import_23855)
        sys_modules_23856 = sys.modules[import_23855]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.cmd', sys_modules_23856.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_23856, sys_modules_23856.module_type_store, module_type_store)
    else:
        from distutils.cmd import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.cmd', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.cmd' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'distutils.cmd', import_23855)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils import log, dir_util' statement (line 8)
try:
    from distutils import log, dir_util

except:
    log = UndefinedType
    dir_util = UndefinedType
import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils', None, module_type_store, ['log', 'dir_util'], [log, dir_util])

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 9, 0))

# Multiple import statement. import os (1/3) (line 9)
import os

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'os', os, module_type_store)
# Multiple import statement. import sys (2/3) (line 9)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 'sys', sys, module_type_store)
# Multiple import statement. import re (3/3) (line 9)
import re

import_module(stypy.reporting.localization.Localization(__file__, 9, 0), 're', re, module_type_store)

# Declaration of the 'install_egg_info' class
# Getting the type of 'Command' (line 11)
Command_23857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 23), 'Command')

class install_egg_info(Command_23857, ):
    str_23858 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 4), 'str', 'Install an .egg-info file for the package')

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 19, 4, False)
        # Assigning a type to the variable 'self' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_function_name', 'install_egg_info.initialize_options')
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_egg_info.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_egg_info.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 20):
        # Getting the type of 'None' (line 20)
        None_23859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 27), 'None')
        # Getting the type of 'self' (line 20)
        self_23860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'self')
        # Setting the type of the member 'install_dir' of a type (line 20)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 20, 8), self_23860, 'install_dir', None_23859)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 19)
        stypy_return_type_23861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23861)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_23861


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_function_name', 'install_egg_info.finalize_options')
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_egg_info.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_egg_info.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 23)
        # Processing the call arguments (line 23)
        str_23864 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 35), 'str', 'install_lib')
        
        # Obtaining an instance of the builtin type 'tuple' (line 23)
        tuple_23865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 50), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 23)
        # Adding element type (line 23)
        str_23866 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 50), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 50), tuple_23865, str_23866)
        # Adding element type (line 23)
        str_23867 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 64), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 50), tuple_23865, str_23867)
        
        # Processing the call keyword arguments (line 23)
        kwargs_23868 = {}
        # Getting the type of 'self' (line 23)
        self_23862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 23)
        set_undefined_options_23863 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_23862, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 23)
        set_undefined_options_call_result_23869 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), set_undefined_options_23863, *[str_23864, tuple_23865], **kwargs_23868)
        
        
        # Assigning a BinOp to a Name (line 24):
        str_23870 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 19), 'str', '%s-%s-py%s.egg-info')
        
        # Obtaining an instance of the builtin type 'tuple' (line 25)
        tuple_23871 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 12), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 25)
        # Adding element type (line 25)
        
        # Call to to_filename(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Call to safe_name(...): (line 25)
        # Processing the call arguments (line 25)
        
        # Call to get_name(...): (line 25)
        # Processing the call keyword arguments (line 25)
        kwargs_23877 = {}
        # Getting the type of 'self' (line 25)
        self_23874 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 34), 'self', False)
        # Obtaining the member 'distribution' of a type (line 25)
        distribution_23875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 34), self_23874, 'distribution')
        # Obtaining the member 'get_name' of a type (line 25)
        get_name_23876 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 34), distribution_23875, 'get_name')
        # Calling get_name(args, kwargs) (line 25)
        get_name_call_result_23878 = invoke(stypy.reporting.localization.Localization(__file__, 25, 34), get_name_23876, *[], **kwargs_23877)
        
        # Processing the call keyword arguments (line 25)
        kwargs_23879 = {}
        # Getting the type of 'safe_name' (line 25)
        safe_name_23873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 24), 'safe_name', False)
        # Calling safe_name(args, kwargs) (line 25)
        safe_name_call_result_23880 = invoke(stypy.reporting.localization.Localization(__file__, 25, 24), safe_name_23873, *[get_name_call_result_23878], **kwargs_23879)
        
        # Processing the call keyword arguments (line 25)
        kwargs_23881 = {}
        # Getting the type of 'to_filename' (line 25)
        to_filename_23872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'to_filename', False)
        # Calling to_filename(args, kwargs) (line 25)
        to_filename_call_result_23882 = invoke(stypy.reporting.localization.Localization(__file__, 25, 12), to_filename_23872, *[safe_name_call_result_23880], **kwargs_23881)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), tuple_23871, to_filename_call_result_23882)
        # Adding element type (line 25)
        
        # Call to to_filename(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to safe_version(...): (line 26)
        # Processing the call arguments (line 26)
        
        # Call to get_version(...): (line 26)
        # Processing the call keyword arguments (line 26)
        kwargs_23888 = {}
        # Getting the type of 'self' (line 26)
        self_23885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 37), 'self', False)
        # Obtaining the member 'distribution' of a type (line 26)
        distribution_23886 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 37), self_23885, 'distribution')
        # Obtaining the member 'get_version' of a type (line 26)
        get_version_23887 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 37), distribution_23886, 'get_version')
        # Calling get_version(args, kwargs) (line 26)
        get_version_call_result_23889 = invoke(stypy.reporting.localization.Localization(__file__, 26, 37), get_version_23887, *[], **kwargs_23888)
        
        # Processing the call keyword arguments (line 26)
        kwargs_23890 = {}
        # Getting the type of 'safe_version' (line 26)
        safe_version_23884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 24), 'safe_version', False)
        # Calling safe_version(args, kwargs) (line 26)
        safe_version_call_result_23891 = invoke(stypy.reporting.localization.Localization(__file__, 26, 24), safe_version_23884, *[get_version_call_result_23889], **kwargs_23890)
        
        # Processing the call keyword arguments (line 26)
        kwargs_23892 = {}
        # Getting the type of 'to_filename' (line 26)
        to_filename_23883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'to_filename', False)
        # Calling to_filename(args, kwargs) (line 26)
        to_filename_call_result_23893 = invoke(stypy.reporting.localization.Localization(__file__, 26, 12), to_filename_23883, *[safe_version_call_result_23891], **kwargs_23892)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), tuple_23871, to_filename_call_result_23893)
        # Adding element type (line 25)
        
        # Obtaining the type of the subscript
        int_23894 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'int')
        slice_23895 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 27, 12), None, int_23894, None)
        # Getting the type of 'sys' (line 27)
        sys_23896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'sys')
        # Obtaining the member 'version' of a type (line 27)
        version_23897 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), sys_23896, 'version')
        # Obtaining the member '__getitem__' of a type (line 27)
        getitem___23898 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 12), version_23897, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 27)
        subscript_call_result_23899 = invoke(stypy.reporting.localization.Localization(__file__, 27, 12), getitem___23898, slice_23895)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 12), tuple_23871, subscript_call_result_23899)
        
        # Applying the binary operator '%' (line 24)
        result_mod_23900 = python_operator(stypy.reporting.localization.Localization(__file__, 24, 19), '%', str_23870, tuple_23871)
        
        # Assigning a type to the variable 'basename' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'basename', result_mod_23900)
        
        # Assigning a Call to a Attribute (line 29):
        
        # Call to join(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'self' (line 29)
        self_23904 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 35), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 29)
        install_dir_23905 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 35), self_23904, 'install_dir')
        # Getting the type of 'basename' (line 29)
        basename_23906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 53), 'basename', False)
        # Processing the call keyword arguments (line 29)
        kwargs_23907 = {}
        # Getting the type of 'os' (line 29)
        os_23901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 22), 'os', False)
        # Obtaining the member 'path' of a type (line 29)
        path_23902 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 22), os_23901, 'path')
        # Obtaining the member 'join' of a type (line 29)
        join_23903 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 22), path_23902, 'join')
        # Calling join(args, kwargs) (line 29)
        join_call_result_23908 = invoke(stypy.reporting.localization.Localization(__file__, 29, 22), join_23903, *[install_dir_23905, basename_23906], **kwargs_23907)
        
        # Getting the type of 'self' (line 29)
        self_23909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'self')
        # Setting the type of the member 'target' of a type (line 29)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), self_23909, 'target', join_call_result_23908)
        
        # Assigning a List to a Attribute (line 30):
        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_23910 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        # Getting the type of 'self' (line 30)
        self_23911 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 24), 'self')
        # Obtaining the member 'target' of a type (line 30)
        target_23912 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 24), self_23911, 'target')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 23), list_23910, target_23912)
        
        # Getting the type of 'self' (line 30)
        self_23913 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'outputs' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_23913, 'outputs', list_23910)
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_23914 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_23914)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_23914


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_egg_info.run.__dict__.__setitem__('stypy_localization', localization)
        install_egg_info.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_egg_info.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_egg_info.run.__dict__.__setitem__('stypy_function_name', 'install_egg_info.run')
        install_egg_info.run.__dict__.__setitem__('stypy_param_names_list', [])
        install_egg_info.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_egg_info.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_egg_info.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_egg_info.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_egg_info.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_egg_info.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_egg_info.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 33):
        # Getting the type of 'self' (line 33)
        self_23915 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'self')
        # Obtaining the member 'target' of a type (line 33)
        target_23916 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), self_23915, 'target')
        # Assigning a type to the variable 'target' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'target', target_23916)
        
        
        # Evaluating a boolean operation
        
        # Call to isdir(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'target' (line 34)
        target_23920 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 25), 'target', False)
        # Processing the call keyword arguments (line 34)
        kwargs_23921 = {}
        # Getting the type of 'os' (line 34)
        os_23917 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_23918 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), os_23917, 'path')
        # Obtaining the member 'isdir' of a type (line 34)
        isdir_23919 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), path_23918, 'isdir')
        # Calling isdir(args, kwargs) (line 34)
        isdir_call_result_23922 = invoke(stypy.reporting.localization.Localization(__file__, 34, 11), isdir_23919, *[target_23920], **kwargs_23921)
        
        
        
        # Call to islink(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'target' (line 34)
        target_23926 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 56), 'target', False)
        # Processing the call keyword arguments (line 34)
        kwargs_23927 = {}
        # Getting the type of 'os' (line 34)
        os_23923 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 41), 'os', False)
        # Obtaining the member 'path' of a type (line 34)
        path_23924 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 41), os_23923, 'path')
        # Obtaining the member 'islink' of a type (line 34)
        islink_23925 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 41), path_23924, 'islink')
        # Calling islink(args, kwargs) (line 34)
        islink_call_result_23928 = invoke(stypy.reporting.localization.Localization(__file__, 34, 41), islink_23925, *[target_23926], **kwargs_23927)
        
        # Applying the 'not' unary operator (line 34)
        result_not__23929 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 37), 'not', islink_call_result_23928)
        
        # Applying the binary operator 'and' (line 34)
        result_and_keyword_23930 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 11), 'and', isdir_call_result_23922, result_not__23929)
        
        # Testing the type of an if condition (line 34)
        if_condition_23931 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), result_and_keyword_23930)
        # Assigning a type to the variable 'if_condition_23931' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_23931', if_condition_23931)
        # SSA begins for if statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to remove_tree(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'target' (line 35)
        target_23934 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'target', False)
        # Processing the call keyword arguments (line 35)
        # Getting the type of 'self' (line 35)
        self_23935 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 49), 'self', False)
        # Obtaining the member 'dry_run' of a type (line 35)
        dry_run_23936 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 49), self_23935, 'dry_run')
        keyword_23937 = dry_run_23936
        kwargs_23938 = {'dry_run': keyword_23937}
        # Getting the type of 'dir_util' (line 35)
        dir_util_23932 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'dir_util', False)
        # Obtaining the member 'remove_tree' of a type (line 35)
        remove_tree_23933 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), dir_util_23932, 'remove_tree')
        # Calling remove_tree(args, kwargs) (line 35)
        remove_tree_call_result_23939 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), remove_tree_23933, *[target_23934], **kwargs_23938)
        
        # SSA branch for the else part of an if statement (line 34)
        module_type_store.open_ssa_branch('else')
        
        
        # Call to exists(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'target' (line 36)
        target_23943 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 28), 'target', False)
        # Processing the call keyword arguments (line 36)
        kwargs_23944 = {}
        # Getting the type of 'os' (line 36)
        os_23940 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'os', False)
        # Obtaining the member 'path' of a type (line 36)
        path_23941 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), os_23940, 'path')
        # Obtaining the member 'exists' of a type (line 36)
        exists_23942 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 13), path_23941, 'exists')
        # Calling exists(args, kwargs) (line 36)
        exists_call_result_23945 = invoke(stypy.reporting.localization.Localization(__file__, 36, 13), exists_23942, *[target_23943], **kwargs_23944)
        
        # Testing the type of an if condition (line 36)
        if_condition_23946 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 36, 13), exists_call_result_23945)
        # Assigning a type to the variable 'if_condition_23946' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 13), 'if_condition_23946', if_condition_23946)
        # SSA begins for if statement (line 36)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to execute(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'os' (line 37)
        os_23949 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 25), 'os', False)
        # Obtaining the member 'unlink' of a type (line 37)
        unlink_23950 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 25), os_23949, 'unlink')
        
        # Obtaining an instance of the builtin type 'tuple' (line 37)
        tuple_23951 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 37)
        # Adding element type (line 37)
        # Getting the type of 'self' (line 37)
        self_23952 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 36), 'self', False)
        # Obtaining the member 'target' of a type (line 37)
        target_23953 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 36), self_23952, 'target')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 37, 36), tuple_23951, target_23953)
        
        str_23954 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 50), 'str', 'Removing ')
        # Getting the type of 'target' (line 37)
        target_23955 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 62), 'target', False)
        # Applying the binary operator '+' (line 37)
        result_add_23956 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 50), '+', str_23954, target_23955)
        
        # Processing the call keyword arguments (line 37)
        kwargs_23957 = {}
        # Getting the type of 'self' (line 37)
        self_23947 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 12), 'self', False)
        # Obtaining the member 'execute' of a type (line 37)
        execute_23948 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 12), self_23947, 'execute')
        # Calling execute(args, kwargs) (line 37)
        execute_call_result_23958 = invoke(stypy.reporting.localization.Localization(__file__, 37, 12), execute_23948, *[unlink_23950, tuple_23951, result_add_23956], **kwargs_23957)
        
        # SSA branch for the else part of an if statement (line 36)
        module_type_store.open_ssa_branch('else')
        
        
        
        # Call to isdir(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'self' (line 38)
        self_23962 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 31), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 38)
        install_dir_23963 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 31), self_23962, 'install_dir')
        # Processing the call keyword arguments (line 38)
        kwargs_23964 = {}
        # Getting the type of 'os' (line 38)
        os_23959 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 17), 'os', False)
        # Obtaining the member 'path' of a type (line 38)
        path_23960 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), os_23959, 'path')
        # Obtaining the member 'isdir' of a type (line 38)
        isdir_23961 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 17), path_23960, 'isdir')
        # Calling isdir(args, kwargs) (line 38)
        isdir_call_result_23965 = invoke(stypy.reporting.localization.Localization(__file__, 38, 17), isdir_23961, *[install_dir_23963], **kwargs_23964)
        
        # Applying the 'not' unary operator (line 38)
        result_not__23966 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 13), 'not', isdir_call_result_23965)
        
        # Testing the type of an if condition (line 38)
        if_condition_23967 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 13), result_not__23966)
        # Assigning a type to the variable 'if_condition_23967' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 13), 'if_condition_23967', if_condition_23967)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to execute(...): (line 39)
        # Processing the call arguments (line 39)
        # Getting the type of 'os' (line 39)
        os_23970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 25), 'os', False)
        # Obtaining the member 'makedirs' of a type (line 39)
        makedirs_23971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 25), os_23970, 'makedirs')
        
        # Obtaining an instance of the builtin type 'tuple' (line 39)
        tuple_23972 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 39), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 39)
        # Adding element type (line 39)
        # Getting the type of 'self' (line 39)
        self_23973 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 39), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 39)
        install_dir_23974 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 39), self_23973, 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 39, 39), tuple_23972, install_dir_23974)
        
        str_23975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 25), 'str', 'Creating ')
        # Getting the type of 'self' (line 40)
        self_23976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 37), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 40)
        install_dir_23977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 37), self_23976, 'install_dir')
        # Applying the binary operator '+' (line 40)
        result_add_23978 = python_operator(stypy.reporting.localization.Localization(__file__, 40, 25), '+', str_23975, install_dir_23977)
        
        # Processing the call keyword arguments (line 39)
        kwargs_23979 = {}
        # Getting the type of 'self' (line 39)
        self_23968 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'self', False)
        # Obtaining the member 'execute' of a type (line 39)
        execute_23969 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 12), self_23968, 'execute')
        # Calling execute(args, kwargs) (line 39)
        execute_call_result_23980 = invoke(stypy.reporting.localization.Localization(__file__, 39, 12), execute_23969, *[makedirs_23971, tuple_23972, result_add_23978], **kwargs_23979)
        
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 36)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 34)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to info(...): (line 41)
        # Processing the call arguments (line 41)
        str_23983 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 17), 'str', 'Writing %s')
        # Getting the type of 'target' (line 41)
        target_23984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 31), 'target', False)
        # Processing the call keyword arguments (line 41)
        kwargs_23985 = {}
        # Getting the type of 'log' (line 41)
        log_23981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'log', False)
        # Obtaining the member 'info' of a type (line 41)
        info_23982 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), log_23981, 'info')
        # Calling info(args, kwargs) (line 41)
        info_call_result_23986 = invoke(stypy.reporting.localization.Localization(__file__, 41, 8), info_23982, *[str_23983, target_23984], **kwargs_23985)
        
        
        
        # Getting the type of 'self' (line 42)
        self_23987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'self')
        # Obtaining the member 'dry_run' of a type (line 42)
        dry_run_23988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 15), self_23987, 'dry_run')
        # Applying the 'not' unary operator (line 42)
        result_not__23989 = python_operator(stypy.reporting.localization.Localization(__file__, 42, 11), 'not', dry_run_23988)
        
        # Testing the type of an if condition (line 42)
        if_condition_23990 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 42, 8), result_not__23989)
        # Assigning a type to the variable 'if_condition_23990' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'if_condition_23990', if_condition_23990)
        # SSA begins for if statement (line 42)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 43):
        
        # Call to open(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'target' (line 43)
        target_23992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 21), 'target', False)
        str_23993 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 29), 'str', 'w')
        # Processing the call keyword arguments (line 43)
        kwargs_23994 = {}
        # Getting the type of 'open' (line 43)
        open_23991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 16), 'open', False)
        # Calling open(args, kwargs) (line 43)
        open_call_result_23995 = invoke(stypy.reporting.localization.Localization(__file__, 43, 16), open_23991, *[target_23992, str_23993], **kwargs_23994)
        
        # Assigning a type to the variable 'f' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'f', open_call_result_23995)
        
        # Call to write_pkg_file(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'f' (line 44)
        f_24000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 54), 'f', False)
        # Processing the call keyword arguments (line 44)
        kwargs_24001 = {}
        # Getting the type of 'self' (line 44)
        self_23996 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 12), 'self', False)
        # Obtaining the member 'distribution' of a type (line 44)
        distribution_23997 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), self_23996, 'distribution')
        # Obtaining the member 'metadata' of a type (line 44)
        metadata_23998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), distribution_23997, 'metadata')
        # Obtaining the member 'write_pkg_file' of a type (line 44)
        write_pkg_file_23999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 12), metadata_23998, 'write_pkg_file')
        # Calling write_pkg_file(args, kwargs) (line 44)
        write_pkg_file_call_result_24002 = invoke(stypy.reporting.localization.Localization(__file__, 44, 12), write_pkg_file_23999, *[f_24000], **kwargs_24001)
        
        
        # Call to close(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_24005 = {}
        # Getting the type of 'f' (line 45)
        f_24003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 45)
        close_24004 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 12), f_24003, 'close')
        # Calling close(args, kwargs) (line 45)
        close_call_result_24006 = invoke(stypy.reporting.localization.Localization(__file__, 45, 12), close_24004, *[], **kwargs_24005)
        
        # SSA join for if statement (line 42)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_24007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24007)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_24007


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_function_name', 'install_egg_info.get_outputs')
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_egg_info.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_egg_info.get_outputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_outputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_outputs(...)' code ##################

        # Getting the type of 'self' (line 48)
        self_24008 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'self')
        # Obtaining the member 'outputs' of a type (line 48)
        outputs_24009 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 15), self_24008, 'outputs')
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', outputs_24009)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_24010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24010)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_24010


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 0, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_egg_info.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_egg_info' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'install_egg_info', install_egg_info)

# Assigning a Str to a Name (line 14):
str_24011 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'str', "Install package's PKG-INFO metadata as an .egg-info file")
# Getting the type of 'install_egg_info'
install_egg_info_24012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_egg_info')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_egg_info_24012, 'description', str_24011)

# Assigning a List to a Name (line 15):

# Obtaining an instance of the builtin type 'list' (line 15)
list_24013 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_24014 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
str_24015 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 9), 'str', 'install-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 9), tuple_24014, str_24015)
# Adding element type (line 16)
str_24016 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 9), tuple_24014, str_24016)
# Adding element type (line 16)
str_24017 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 30), 'str', 'directory to install to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 9), tuple_24014, str_24017)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 19), list_24013, tuple_24014)

# Getting the type of 'install_egg_info'
install_egg_info_24018 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_egg_info')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_egg_info_24018, 'user_options', list_24013)

@norecursion
def safe_name(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'safe_name'
    module_type_store = module_type_store.open_function_context('safe_name', 55, 0, False)
    
    # Passed parameters checking function
    safe_name.stypy_localization = localization
    safe_name.stypy_type_of_self = None
    safe_name.stypy_type_store = module_type_store
    safe_name.stypy_function_name = 'safe_name'
    safe_name.stypy_param_names_list = ['name']
    safe_name.stypy_varargs_param_name = None
    safe_name.stypy_kwargs_param_name = None
    safe_name.stypy_call_defaults = defaults
    safe_name.stypy_call_varargs = varargs
    safe_name.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'safe_name', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'safe_name', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'safe_name(...)' code ##################

    str_24019 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', "Convert an arbitrary string to a standard distribution name\n\n    Any runs of non-alphanumeric/. characters are replaced with a single '-'.\n    ")
    
    # Call to sub(...): (line 60)
    # Processing the call arguments (line 60)
    str_24022 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 18), 'str', '[^A-Za-z0-9.]+')
    str_24023 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 36), 'str', '-')
    # Getting the type of 'name' (line 60)
    name_24024 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 41), 'name', False)
    # Processing the call keyword arguments (line 60)
    kwargs_24025 = {}
    # Getting the type of 're' (line 60)
    re_24020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 11), 're', False)
    # Obtaining the member 'sub' of a type (line 60)
    sub_24021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 11), re_24020, 'sub')
    # Calling sub(args, kwargs) (line 60)
    sub_call_result_24026 = invoke(stypy.reporting.localization.Localization(__file__, 60, 11), sub_24021, *[str_24022, str_24023, name_24024], **kwargs_24025)
    
    # Assigning a type to the variable 'stypy_return_type' (line 60)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'stypy_return_type', sub_call_result_24026)
    
    # ################# End of 'safe_name(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'safe_name' in the type store
    # Getting the type of 'stypy_return_type' (line 55)
    stypy_return_type_24027 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24027)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'safe_name'
    return stypy_return_type_24027

# Assigning a type to the variable 'safe_name' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'safe_name', safe_name)

@norecursion
def safe_version(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'safe_version'
    module_type_store = module_type_store.open_function_context('safe_version', 63, 0, False)
    
    # Passed parameters checking function
    safe_version.stypy_localization = localization
    safe_version.stypy_type_of_self = None
    safe_version.stypy_type_store = module_type_store
    safe_version.stypy_function_name = 'safe_version'
    safe_version.stypy_param_names_list = ['version']
    safe_version.stypy_varargs_param_name = None
    safe_version.stypy_kwargs_param_name = None
    safe_version.stypy_call_defaults = defaults
    safe_version.stypy_call_varargs = varargs
    safe_version.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'safe_version', ['version'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'safe_version', localization, ['version'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'safe_version(...)' code ##################

    str_24028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', 'Convert an arbitrary string to a standard version string\n\n    Spaces become dots, and all other non-alphanumeric characters become\n    dashes, with runs of multiple dashes condensed to a single dash.\n    ')
    
    # Assigning a Call to a Name (line 69):
    
    # Call to replace(...): (line 69)
    # Processing the call arguments (line 69)
    str_24031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 30), 'str', ' ')
    str_24032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 34), 'str', '.')
    # Processing the call keyword arguments (line 69)
    kwargs_24033 = {}
    # Getting the type of 'version' (line 69)
    version_24029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 14), 'version', False)
    # Obtaining the member 'replace' of a type (line 69)
    replace_24030 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 14), version_24029, 'replace')
    # Calling replace(args, kwargs) (line 69)
    replace_call_result_24034 = invoke(stypy.reporting.localization.Localization(__file__, 69, 14), replace_24030, *[str_24031, str_24032], **kwargs_24033)
    
    # Assigning a type to the variable 'version' (line 69)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 4), 'version', replace_call_result_24034)
    
    # Call to sub(...): (line 70)
    # Processing the call arguments (line 70)
    str_24037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 18), 'str', '[^A-Za-z0-9.]+')
    str_24038 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 70, 36), 'str', '-')
    # Getting the type of 'version' (line 70)
    version_24039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 41), 'version', False)
    # Processing the call keyword arguments (line 70)
    kwargs_24040 = {}
    # Getting the type of 're' (line 70)
    re_24035 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 11), 're', False)
    # Obtaining the member 'sub' of a type (line 70)
    sub_24036 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 11), re_24035, 'sub')
    # Calling sub(args, kwargs) (line 70)
    sub_call_result_24041 = invoke(stypy.reporting.localization.Localization(__file__, 70, 11), sub_24036, *[str_24037, str_24038, version_24039], **kwargs_24040)
    
    # Assigning a type to the variable 'stypy_return_type' (line 70)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type', sub_call_result_24041)
    
    # ################# End of 'safe_version(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'safe_version' in the type store
    # Getting the type of 'stypy_return_type' (line 63)
    stypy_return_type_24042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24042)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'safe_version'
    return stypy_return_type_24042

# Assigning a type to the variable 'safe_version' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'safe_version', safe_version)

@norecursion
def to_filename(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function 'to_filename'
    module_type_store = module_type_store.open_function_context('to_filename', 73, 0, False)
    
    # Passed parameters checking function
    to_filename.stypy_localization = localization
    to_filename.stypy_type_of_self = None
    to_filename.stypy_type_store = module_type_store
    to_filename.stypy_function_name = 'to_filename'
    to_filename.stypy_param_names_list = ['name']
    to_filename.stypy_varargs_param_name = None
    to_filename.stypy_kwargs_param_name = None
    to_filename.stypy_call_defaults = defaults
    to_filename.stypy_call_varargs = varargs
    to_filename.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, 'to_filename', ['name'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Initialize method data
    init_call_information(module_type_store, 'to_filename', localization, ['name'], arguments)
    
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of 'to_filename(...)' code ##################

    str_24043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', "Convert a project or version name to its filename-escaped form\n\n    Any '-' characters are currently replaced with '_'.\n    ")
    
    # Call to replace(...): (line 78)
    # Processing the call arguments (line 78)
    str_24046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 24), 'str', '-')
    str_24047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 78, 28), 'str', '_')
    # Processing the call keyword arguments (line 78)
    kwargs_24048 = {}
    # Getting the type of 'name' (line 78)
    name_24044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 11), 'name', False)
    # Obtaining the member 'replace' of a type (line 78)
    replace_24045 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 11), name_24044, 'replace')
    # Calling replace(args, kwargs) (line 78)
    replace_call_result_24049 = invoke(stypy.reporting.localization.Localization(__file__, 78, 11), replace_24045, *[str_24046, str_24047], **kwargs_24048)
    
    # Assigning a type to the variable 'stypy_return_type' (line 78)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 4), 'stypy_return_type', replace_call_result_24049)
    
    # ################# End of 'to_filename(...)' code ##################

    # Teardown call information
    teardown_call_information(localization, arguments)
    
    # Storing the return type of function 'to_filename' in the type store
    # Getting the type of 'stypy_return_type' (line 73)
    stypy_return_type_24050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_24050)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function 'to_filename'
    return stypy_return_type_24050

# Assigning a type to the variable 'to_filename' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'to_filename', to_filename)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
