
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import sys
4: if 'setuptools' in sys.modules:
5:     import setuptools.command.install as old_install_mod
6:     have_setuptools = True
7: else:
8:     import distutils.command.install as old_install_mod
9:     have_setuptools = False
10: from distutils.file_util import write_file
11: 
12: old_install = old_install_mod.install
13: 
14: class install(old_install):
15: 
16:     # Always run install_clib - the command is cheap, so no need to bypass it;
17:     # but it's not run by setuptools -- so it's run again in install_data
18:     sub_commands = old_install.sub_commands + [
19:         ('install_clib', lambda x: True)
20:     ]
21: 
22:     def finalize_options (self):
23:         old_install.finalize_options(self)
24:         self.install_lib = self.install_libbase
25: 
26:     def setuptools_run(self):
27:         ''' The setuptools version of the .run() method.
28: 
29:         We must pull in the entire code so we can override the level used in the
30:         _getframe() call since we wrap this call by one more level.
31:         '''
32:         from distutils.command.install import install as distutils_install
33: 
34:         # Explicit request for old-style install?  Just do it
35:         if self.old_and_unmanageable or self.single_version_externally_managed:
36:             return distutils_install.run(self)
37: 
38:         # Attempt to detect whether we were called from setup() or by another
39:         # command.  If we were called by setup(), our caller will be the
40:         # 'run_command' method in 'distutils.dist', and *its* caller will be
41:         # the 'run_commands' method.  If we were called any other way, our
42:         # immediate caller *might* be 'run_command', but it won't have been
43:         # called by 'run_commands'.  This is slightly kludgy, but seems to
44:         # work.
45:         #
46:         caller = sys._getframe(3)
47:         caller_module = caller.f_globals.get('__name__', '')
48:         caller_name = caller.f_code.co_name
49: 
50:         if caller_module != 'distutils.dist' or caller_name!='run_commands':
51:             # We weren't called from the command line or setup(), so we
52:             # should run in backward-compatibility mode to support bdist_*
53:             # commands.
54:             distutils_install.run(self)
55:         else:
56:             self.do_egg_install()
57: 
58:     def run(self):
59:         if not have_setuptools:
60:             r = old_install.run(self)
61:         else:
62:             r = self.setuptools_run()
63:         if self.record:
64:             # bdist_rpm fails when INSTALLED_FILES contains
65:             # paths with spaces. Such paths must be enclosed
66:             # with double-quotes.
67:             f = open(self.record, 'r')
68:             lines = []
69:             need_rewrite = False
70:             for l in f:
71:                 l = l.rstrip()
72:                 if ' ' in l:
73:                     need_rewrite = True
74:                     l = '"%s"' % (l)
75:                 lines.append(l)
76:             f.close()
77:             if need_rewrite:
78:                 self.execute(write_file,
79:                              (self.record, lines),
80:                              "re-writing list of installed files to '%s'" %
81:                              self.record)
82:         return r
83: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import sys' statement (line 3)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'sys', sys, module_type_store)



str_59340 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, 3), 'str', 'setuptools')
# Getting the type of 'sys' (line 4)
sys_59341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 4, 19), 'sys')
# Obtaining the member 'modules' of a type (line 4)
modules_59342 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 4, 19), sys_59341, 'modules')
# Applying the binary operator 'in' (line 4)
result_contains_59343 = python_operator(stypy.reporting.localization.Localization(__file__, 4, 3), 'in', str_59340, modules_59342)

# Testing the type of an if condition (line 4)
if_condition_59344 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 4, 0), result_contains_59343)
# Assigning a type to the variable 'if_condition_59344' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'if_condition_59344', if_condition_59344)
# SSA begins for if statement (line 4)
module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 4))

# 'import setuptools.command.install' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59345 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'setuptools.command.install')

if (type(import_59345) is not StypyTypeError):

    if (import_59345 != 'pyd_module'):
        __import__(import_59345)
        sys_modules_59346 = sys.modules[import_59345]
        import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'old_install_mod', sys_modules_59346.module_type_store, module_type_store)
    else:
        import setuptools.command.install as old_install_mod

        import_module(stypy.reporting.localization.Localization(__file__, 5, 4), 'old_install_mod', setuptools.command.install, module_type_store)

else:
    # Assigning a type to the variable 'setuptools.command.install' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 4), 'setuptools.command.install', import_59345)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')


# Assigning a Name to a Name (line 6):
# Getting the type of 'True' (line 6)
True_59347 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 22), 'True')
# Assigning a type to the variable 'have_setuptools' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 4), 'have_setuptools', True_59347)
# SSA branch for the else part of an if statement (line 4)
module_type_store.open_ssa_branch('else')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 4))

# 'import distutils.command.install' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59348 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.command.install')

if (type(import_59348) is not StypyTypeError):

    if (import_59348 != 'pyd_module'):
        __import__(import_59348)
        sys_modules_59349 = sys.modules[import_59348]
        import_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'old_install_mod', sys_modules_59349.module_type_store, module_type_store)
    else:
        import distutils.command.install as old_install_mod

        import_module(stypy.reporting.localization.Localization(__file__, 8, 4), 'old_install_mod', distutils.command.install, module_type_store)

else:
    # Assigning a type to the variable 'distutils.command.install' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'distutils.command.install', import_59348)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')


# Assigning a Name to a Name (line 9):
# Getting the type of 'False' (line 9)
False_59350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 22), 'False')
# Assigning a type to the variable 'have_setuptools' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'have_setuptools', False_59350)
# SSA join for if statement (line 4)
module_type_store = module_type_store.join_ssa_context()

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 10, 0))

# 'from distutils.file_util import write_file' statement (line 10)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_59351 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.file_util')

if (type(import_59351) is not StypyTypeError):

    if (import_59351 != 'pyd_module'):
        __import__(import_59351)
        sys_modules_59352 = sys.modules[import_59351]
        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.file_util', sys_modules_59352.module_type_store, module_type_store, ['write_file'])
        nest_module(stypy.reporting.localization.Localization(__file__, 10, 0), __file__, sys_modules_59352, sys_modules_59352.module_type_store, module_type_store)
    else:
        from distutils.file_util import write_file

        import_from_module(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.file_util', None, module_type_store, ['write_file'], [write_file])

else:
    # Assigning a type to the variable 'distutils.file_util' (line 10)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'distutils.file_util', import_59351)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')


# Assigning a Attribute to a Name (line 12):
# Getting the type of 'old_install_mod' (line 12)
old_install_mod_59353 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 14), 'old_install_mod')
# Obtaining the member 'install' of a type (line 12)
install_59354 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 14), old_install_mod_59353, 'install')
# Assigning a type to the variable 'old_install' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'old_install', install_59354)
# Declaration of the 'install' class
# Getting the type of 'old_install' (line 14)
old_install_59355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 14), 'old_install')

class install(old_install_59355, ):

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
        install.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.finalize_options.__dict__.__setitem__('stypy_function_name', 'install.finalize_options')
        install.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to finalize_options(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'self' (line 23)
        self_59358 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 37), 'self', False)
        # Processing the call keyword arguments (line 23)
        kwargs_59359 = {}
        # Getting the type of 'old_install' (line 23)
        old_install_59356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'old_install', False)
        # Obtaining the member 'finalize_options' of a type (line 23)
        finalize_options_59357 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), old_install_59356, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 23)
        finalize_options_call_result_59360 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), finalize_options_59357, *[self_59358], **kwargs_59359)
        
        
        # Assigning a Attribute to a Attribute (line 24):
        # Getting the type of 'self' (line 24)
        self_59361 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'self')
        # Obtaining the member 'install_libbase' of a type (line 24)
        install_libbase_59362 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 27), self_59361, 'install_libbase')
        # Getting the type of 'self' (line 24)
        self_59363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self')
        # Setting the type of the member 'install_lib' of a type (line 24)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_59363, 'install_lib', install_libbase_59362)
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_59364 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59364)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_59364


    @norecursion
    def setuptools_run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'setuptools_run'
        module_type_store = module_type_store.open_function_context('setuptools_run', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.setuptools_run.__dict__.__setitem__('stypy_localization', localization)
        install.setuptools_run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.setuptools_run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.setuptools_run.__dict__.__setitem__('stypy_function_name', 'install.setuptools_run')
        install.setuptools_run.__dict__.__setitem__('stypy_param_names_list', [])
        install.setuptools_run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.setuptools_run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.setuptools_run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.setuptools_run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.setuptools_run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.setuptools_run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.setuptools_run', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'setuptools_run', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'setuptools_run(...)' code ##################

        str_59365 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', ' The setuptools version of the .run() method.\n\n        We must pull in the entire code so we can override the level used in the\n        _getframe() call since we wrap this call by one more level.\n        ')
        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 8))
        
        # 'from distutils.command.install import distutils_install' statement (line 32)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_59366 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 32, 8), 'distutils.command.install')

        if (type(import_59366) is not StypyTypeError):

            if (import_59366 != 'pyd_module'):
                __import__(import_59366)
                sys_modules_59367 = sys.modules[import_59366]
                import_from_module(stypy.reporting.localization.Localization(__file__, 32, 8), 'distutils.command.install', sys_modules_59367.module_type_store, module_type_store, ['install'])
                nest_module(stypy.reporting.localization.Localization(__file__, 32, 8), __file__, sys_modules_59367, sys_modules_59367.module_type_store, module_type_store)
            else:
                from distutils.command.install import install as distutils_install

                import_from_module(stypy.reporting.localization.Localization(__file__, 32, 8), 'distutils.command.install', None, module_type_store, ['install'], [distutils_install])

        else:
            # Assigning a type to the variable 'distutils.command.install' (line 32)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'distutils.command.install', import_59366)

        # Adding an alias
        module_type_store.add_alias('distutils_install', 'install')
        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 35)
        self_59368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 11), 'self')
        # Obtaining the member 'old_and_unmanageable' of a type (line 35)
        old_and_unmanageable_59369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 11), self_59368, 'old_and_unmanageable')
        # Getting the type of 'self' (line 35)
        self_59370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 40), 'self')
        # Obtaining the member 'single_version_externally_managed' of a type (line 35)
        single_version_externally_managed_59371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 40), self_59370, 'single_version_externally_managed')
        # Applying the binary operator 'or' (line 35)
        result_or_keyword_59372 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 11), 'or', old_and_unmanageable_59369, single_version_externally_managed_59371)
        
        # Testing the type of an if condition (line 35)
        if_condition_59373 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 8), result_or_keyword_59372)
        # Assigning a type to the variable 'if_condition_59373' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'if_condition_59373', if_condition_59373)
        # SSA begins for if statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_59376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 41), 'self', False)
        # Processing the call keyword arguments (line 36)
        kwargs_59377 = {}
        # Getting the type of 'distutils_install' (line 36)
        distutils_install_59374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 19), 'distutils_install', False)
        # Obtaining the member 'run' of a type (line 36)
        run_59375 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 19), distutils_install_59374, 'run')
        # Calling run(args, kwargs) (line 36)
        run_call_result_59378 = invoke(stypy.reporting.localization.Localization(__file__, 36, 19), run_59375, *[self_59376], **kwargs_59377)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'stypy_return_type', run_call_result_59378)
        # SSA join for if statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Name (line 46):
        
        # Call to _getframe(...): (line 46)
        # Processing the call arguments (line 46)
        int_59381 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 31), 'int')
        # Processing the call keyword arguments (line 46)
        kwargs_59382 = {}
        # Getting the type of 'sys' (line 46)
        sys_59379 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 17), 'sys', False)
        # Obtaining the member '_getframe' of a type (line 46)
        _getframe_59380 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 17), sys_59379, '_getframe')
        # Calling _getframe(args, kwargs) (line 46)
        _getframe_call_result_59383 = invoke(stypy.reporting.localization.Localization(__file__, 46, 17), _getframe_59380, *[int_59381], **kwargs_59382)
        
        # Assigning a type to the variable 'caller' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'caller', _getframe_call_result_59383)
        
        # Assigning a Call to a Name (line 47):
        
        # Call to get(...): (line 47)
        # Processing the call arguments (line 47)
        str_59387 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 45), 'str', '__name__')
        str_59388 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 57), 'str', '')
        # Processing the call keyword arguments (line 47)
        kwargs_59389 = {}
        # Getting the type of 'caller' (line 47)
        caller_59384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 24), 'caller', False)
        # Obtaining the member 'f_globals' of a type (line 47)
        f_globals_59385 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), caller_59384, 'f_globals')
        # Obtaining the member 'get' of a type (line 47)
        get_59386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 24), f_globals_59385, 'get')
        # Calling get(args, kwargs) (line 47)
        get_call_result_59390 = invoke(stypy.reporting.localization.Localization(__file__, 47, 24), get_59386, *[str_59387, str_59388], **kwargs_59389)
        
        # Assigning a type to the variable 'caller_module' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'caller_module', get_call_result_59390)
        
        # Assigning a Attribute to a Name (line 48):
        # Getting the type of 'caller' (line 48)
        caller_59391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 22), 'caller')
        # Obtaining the member 'f_code' of a type (line 48)
        f_code_59392 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), caller_59391, 'f_code')
        # Obtaining the member 'co_name' of a type (line 48)
        co_name_59393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 22), f_code_59392, 'co_name')
        # Assigning a type to the variable 'caller_name' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'caller_name', co_name_59393)
        
        
        # Evaluating a boolean operation
        
        # Getting the type of 'caller_module' (line 50)
        caller_module_59394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 11), 'caller_module')
        str_59395 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 28), 'str', 'distutils.dist')
        # Applying the binary operator '!=' (line 50)
        result_ne_59396 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), '!=', caller_module_59394, str_59395)
        
        
        # Getting the type of 'caller_name' (line 50)
        caller_name_59397 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 48), 'caller_name')
        str_59398 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, 61), 'str', 'run_commands')
        # Applying the binary operator '!=' (line 50)
        result_ne_59399 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 48), '!=', caller_name_59397, str_59398)
        
        # Applying the binary operator 'or' (line 50)
        result_or_keyword_59400 = python_operator(stypy.reporting.localization.Localization(__file__, 50, 11), 'or', result_ne_59396, result_ne_59399)
        
        # Testing the type of an if condition (line 50)
        if_condition_59401 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 50, 8), result_or_keyword_59400)
        # Assigning a type to the variable 'if_condition_59401' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'if_condition_59401', if_condition_59401)
        # SSA begins for if statement (line 50)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to run(...): (line 54)
        # Processing the call arguments (line 54)
        # Getting the type of 'self' (line 54)
        self_59404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 34), 'self', False)
        # Processing the call keyword arguments (line 54)
        kwargs_59405 = {}
        # Getting the type of 'distutils_install' (line 54)
        distutils_install_59402 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 12), 'distutils_install', False)
        # Obtaining the member 'run' of a type (line 54)
        run_59403 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 12), distutils_install_59402, 'run')
        # Calling run(args, kwargs) (line 54)
        run_call_result_59406 = invoke(stypy.reporting.localization.Localization(__file__, 54, 12), run_59403, *[self_59404], **kwargs_59405)
        
        # SSA branch for the else part of an if statement (line 50)
        module_type_store.open_ssa_branch('else')
        
        # Call to do_egg_install(...): (line 56)
        # Processing the call keyword arguments (line 56)
        kwargs_59409 = {}
        # Getting the type of 'self' (line 56)
        self_59407 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 12), 'self', False)
        # Obtaining the member 'do_egg_install' of a type (line 56)
        do_egg_install_59408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 12), self_59407, 'do_egg_install')
        # Calling do_egg_install(args, kwargs) (line 56)
        do_egg_install_call_result_59410 = invoke(stypy.reporting.localization.Localization(__file__, 56, 12), do_egg_install_59408, *[], **kwargs_59409)
        
        # SSA join for if statement (line 50)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'setuptools_run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'setuptools_run' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_59411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59411)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'setuptools_run'
        return stypy_return_type_59411


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 58, 4, False)
        # Assigning a type to the variable 'self' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install.run.__dict__.__setitem__('stypy_localization', localization)
        install.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install.run.__dict__.__setitem__('stypy_function_name', 'install.run')
        install.run.__dict__.__setitem__('stypy_param_names_list', [])
        install.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'have_setuptools' (line 59)
        have_setuptools_59412 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 15), 'have_setuptools')
        # Applying the 'not' unary operator (line 59)
        result_not__59413 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 11), 'not', have_setuptools_59412)
        
        # Testing the type of an if condition (line 59)
        if_condition_59414 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 59, 8), result_not__59413)
        # Assigning a type to the variable 'if_condition_59414' (line 59)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 8), 'if_condition_59414', if_condition_59414)
        # SSA begins for if statement (line 59)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 60):
        
        # Call to run(...): (line 60)
        # Processing the call arguments (line 60)
        # Getting the type of 'self' (line 60)
        self_59417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 32), 'self', False)
        # Processing the call keyword arguments (line 60)
        kwargs_59418 = {}
        # Getting the type of 'old_install' (line 60)
        old_install_59415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 16), 'old_install', False)
        # Obtaining the member 'run' of a type (line 60)
        run_59416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 16), old_install_59415, 'run')
        # Calling run(args, kwargs) (line 60)
        run_call_result_59419 = invoke(stypy.reporting.localization.Localization(__file__, 60, 16), run_59416, *[self_59417], **kwargs_59418)
        
        # Assigning a type to the variable 'r' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 12), 'r', run_call_result_59419)
        # SSA branch for the else part of an if statement (line 59)
        module_type_store.open_ssa_branch('else')
        
        # Assigning a Call to a Name (line 62):
        
        # Call to setuptools_run(...): (line 62)
        # Processing the call keyword arguments (line 62)
        kwargs_59422 = {}
        # Getting the type of 'self' (line 62)
        self_59420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 16), 'self', False)
        # Obtaining the member 'setuptools_run' of a type (line 62)
        setuptools_run_59421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 16), self_59420, 'setuptools_run')
        # Calling setuptools_run(args, kwargs) (line 62)
        setuptools_run_call_result_59423 = invoke(stypy.reporting.localization.Localization(__file__, 62, 16), setuptools_run_59421, *[], **kwargs_59422)
        
        # Assigning a type to the variable 'r' (line 62)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'r', setuptools_run_call_result_59423)
        # SSA join for if statement (line 59)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Getting the type of 'self' (line 63)
        self_59424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 11), 'self')
        # Obtaining the member 'record' of a type (line 63)
        record_59425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 11), self_59424, 'record')
        # Testing the type of an if condition (line 63)
        if_condition_59426 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 63, 8), record_59425)
        # Assigning a type to the variable 'if_condition_59426' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'if_condition_59426', if_condition_59426)
        # SSA begins for if statement (line 63)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Name (line 67):
        
        # Call to open(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'self' (line 67)
        self_59428 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 21), 'self', False)
        # Obtaining the member 'record' of a type (line 67)
        record_59429 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 21), self_59428, 'record')
        str_59430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 34), 'str', 'r')
        # Processing the call keyword arguments (line 67)
        kwargs_59431 = {}
        # Getting the type of 'open' (line 67)
        open_59427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 16), 'open', False)
        # Calling open(args, kwargs) (line 67)
        open_call_result_59432 = invoke(stypy.reporting.localization.Localization(__file__, 67, 16), open_59427, *[record_59429, str_59430], **kwargs_59431)
        
        # Assigning a type to the variable 'f' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 12), 'f', open_call_result_59432)
        
        # Assigning a List to a Name (line 68):
        
        # Obtaining an instance of the builtin type 'list' (line 68)
        list_59433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 68)
        
        # Assigning a type to the variable 'lines' (line 68)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 12), 'lines', list_59433)
        
        # Assigning a Name to a Name (line 69):
        # Getting the type of 'False' (line 69)
        False_59434 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 27), 'False')
        # Assigning a type to the variable 'need_rewrite' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'need_rewrite', False_59434)
        
        # Getting the type of 'f' (line 70)
        f_59435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'f')
        # Testing the type of a for loop iterable (line 70)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 70, 12), f_59435)
        # Getting the type of the for loop variable (line 70)
        for_loop_var_59436 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 70, 12), f_59435)
        # Assigning a type to the variable 'l' (line 70)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 12), 'l', for_loop_var_59436)
        # SSA begins for a for statement (line 70)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 71):
        
        # Call to rstrip(...): (line 71)
        # Processing the call keyword arguments (line 71)
        kwargs_59439 = {}
        # Getting the type of 'l' (line 71)
        l_59437 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'l', False)
        # Obtaining the member 'rstrip' of a type (line 71)
        rstrip_59438 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 71, 20), l_59437, 'rstrip')
        # Calling rstrip(args, kwargs) (line 71)
        rstrip_call_result_59440 = invoke(stypy.reporting.localization.Localization(__file__, 71, 20), rstrip_59438, *[], **kwargs_59439)
        
        # Assigning a type to the variable 'l' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 16), 'l', rstrip_call_result_59440)
        
        
        str_59441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 72, 19), 'str', ' ')
        # Getting the type of 'l' (line 72)
        l_59442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 26), 'l')
        # Applying the binary operator 'in' (line 72)
        result_contains_59443 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 19), 'in', str_59441, l_59442)
        
        # Testing the type of an if condition (line 72)
        if_condition_59444 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 16), result_contains_59443)
        # Assigning a type to the variable 'if_condition_59444' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 16), 'if_condition_59444', if_condition_59444)
        # SSA begins for if statement (line 72)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Name to a Name (line 73):
        # Getting the type of 'True' (line 73)
        True_59445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 35), 'True')
        # Assigning a type to the variable 'need_rewrite' (line 73)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 20), 'need_rewrite', True_59445)
        
        # Assigning a BinOp to a Name (line 74):
        str_59446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 24), 'str', '"%s"')
        # Getting the type of 'l' (line 74)
        l_59447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'l')
        # Applying the binary operator '%' (line 74)
        result_mod_59448 = python_operator(stypy.reporting.localization.Localization(__file__, 74, 24), '%', str_59446, l_59447)
        
        # Assigning a type to the variable 'l' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'l', result_mod_59448)
        # SSA join for if statement (line 72)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to append(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'l' (line 75)
        l_59451 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 29), 'l', False)
        # Processing the call keyword arguments (line 75)
        kwargs_59452 = {}
        # Getting the type of 'lines' (line 75)
        lines_59449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 16), 'lines', False)
        # Obtaining the member 'append' of a type (line 75)
        append_59450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 16), lines_59449, 'append')
        # Calling append(args, kwargs) (line 75)
        append_call_result_59453 = invoke(stypy.reporting.localization.Localization(__file__, 75, 16), append_59450, *[l_59451], **kwargs_59452)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to close(...): (line 76)
        # Processing the call keyword arguments (line 76)
        kwargs_59456 = {}
        # Getting the type of 'f' (line 76)
        f_59454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 12), 'f', False)
        # Obtaining the member 'close' of a type (line 76)
        close_59455 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 76, 12), f_59454, 'close')
        # Calling close(args, kwargs) (line 76)
        close_call_result_59457 = invoke(stypy.reporting.localization.Localization(__file__, 76, 12), close_59455, *[], **kwargs_59456)
        
        
        # Getting the type of 'need_rewrite' (line 77)
        need_rewrite_59458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 15), 'need_rewrite')
        # Testing the type of an if condition (line 77)
        if_condition_59459 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 12), need_rewrite_59458)
        # Assigning a type to the variable 'if_condition_59459' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 12), 'if_condition_59459', if_condition_59459)
        # SSA begins for if statement (line 77)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to execute(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'write_file' (line 78)
        write_file_59462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 29), 'write_file', False)
        
        # Obtaining an instance of the builtin type 'tuple' (line 79)
        tuple_59463 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 79, 30), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 79)
        # Adding element type (line 79)
        # Getting the type of 'self' (line 79)
        self_59464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 30), 'self', False)
        # Obtaining the member 'record' of a type (line 79)
        record_59465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 30), self_59464, 'record')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 30), tuple_59463, record_59465)
        # Adding element type (line 79)
        # Getting the type of 'lines' (line 79)
        lines_59466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 43), 'lines', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 79, 30), tuple_59463, lines_59466)
        
        str_59467 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 80, 29), 'str', "re-writing list of installed files to '%s'")
        # Getting the type of 'self' (line 81)
        self_59468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 29), 'self', False)
        # Obtaining the member 'record' of a type (line 81)
        record_59469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 29), self_59468, 'record')
        # Applying the binary operator '%' (line 80)
        result_mod_59470 = python_operator(stypy.reporting.localization.Localization(__file__, 80, 29), '%', str_59467, record_59469)
        
        # Processing the call keyword arguments (line 78)
        kwargs_59471 = {}
        # Getting the type of 'self' (line 78)
        self_59460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 16), 'self', False)
        # Obtaining the member 'execute' of a type (line 78)
        execute_59461 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 16), self_59460, 'execute')
        # Calling execute(args, kwargs) (line 78)
        execute_call_result_59472 = invoke(stypy.reporting.localization.Localization(__file__, 78, 16), execute_59461, *[write_file_59462, tuple_59463, result_mod_59470], **kwargs_59471)
        
        # SSA join for if statement (line 77)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 63)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'r' (line 82)
        r_59473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 15), 'r')
        # Assigning a type to the variable 'stypy_return_type' (line 82)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 8), 'stypy_return_type', r_59473)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 58)
        stypy_return_type_59474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_59474)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_59474


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install' (line 14)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'install', install)

# Assigning a BinOp to a Name (line 18):
# Getting the type of 'old_install' (line 18)
old_install_59475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 19), 'old_install')
# Obtaining the member 'sub_commands' of a type (line 18)
sub_commands_59476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 19), old_install_59475, 'sub_commands')

# Obtaining an instance of the builtin type 'list' (line 18)
list_59477 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 46), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_59478 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
str_59479 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'str', 'install_clib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 9), tuple_59478, str_59479)
# Adding element type (line 19)

@norecursion
def _stypy_temp_lambda_22(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_22'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_22', 19, 25, True)
    # Passed parameters checking function
    _stypy_temp_lambda_22.stypy_localization = localization
    _stypy_temp_lambda_22.stypy_type_of_self = None
    _stypy_temp_lambda_22.stypy_type_store = module_type_store
    _stypy_temp_lambda_22.stypy_function_name = '_stypy_temp_lambda_22'
    _stypy_temp_lambda_22.stypy_param_names_list = ['x']
    _stypy_temp_lambda_22.stypy_varargs_param_name = None
    _stypy_temp_lambda_22.stypy_kwargs_param_name = None
    _stypy_temp_lambda_22.stypy_call_defaults = defaults
    _stypy_temp_lambda_22.stypy_call_varargs = varargs
    _stypy_temp_lambda_22.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_22', ['x'], None, None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_22', ['x'], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'True' (line 19)
    True_59480 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 35), 'True')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 19)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'stypy_return_type', True_59480)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_22' in the type store
    # Getting the type of 'stypy_return_type' (line 19)
    stypy_return_type_59481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_59481)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_22'
    return stypy_return_type_59481

# Assigning a type to the variable '_stypy_temp_lambda_22' (line 19)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), '_stypy_temp_lambda_22', _stypy_temp_lambda_22)
# Getting the type of '_stypy_temp_lambda_22' (line 19)
_stypy_temp_lambda_22_59482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 25), '_stypy_temp_lambda_22')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 9), tuple_59478, _stypy_temp_lambda_22_59482)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 46), list_59477, tuple_59478)

# Applying the binary operator '+' (line 18)
result_add_59483 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 19), '+', sub_commands_59476, list_59477)

# Getting the type of 'install'
install_59484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install')
# Setting the type of the member 'sub_commands' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_59484, 'sub_commands', result_add_59483)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
