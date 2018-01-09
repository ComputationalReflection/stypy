
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: ''' Modified version of build_scripts that handles building scripts from functions.
2: 
3: '''
4: from __future__ import division, absolute_import, print_function
5: 
6: from distutils.command.build_scripts import build_scripts as old_build_scripts
7: from numpy.distutils import log
8: from numpy.distutils.misc_util import is_string
9: 
10: class build_scripts(old_build_scripts):
11: 
12:     def generate_scripts(self, scripts):
13:         new_scripts = []
14:         func_scripts = []
15:         for script in scripts:
16:             if is_string(script):
17:                 new_scripts.append(script)
18:             else:
19:                 func_scripts.append(script)
20:         if not func_scripts:
21:             return new_scripts
22: 
23:         build_dir = self.build_dir
24:         self.mkpath(build_dir)
25:         for func in func_scripts:
26:             script = func(build_dir)
27:             if not script:
28:                 continue
29:             if is_string(script):
30:                 log.info("  adding '%s' to scripts" % (script,))
31:                 new_scripts.append(script)
32:             else:
33:                 [log.info("  adding '%s' to scripts" % (s,)) for s in script]
34:                 new_scripts.extend(list(script))
35:         return new_scripts
36: 
37:     def run (self):
38:         if not self.scripts:
39:             return
40: 
41:         self.scripts = self.generate_scripts(self.scripts)
42:         # Now make sure that the distribution object has this list of scripts.
43:         # setuptools' develop command requires that this be a list of filenames,
44:         # not functions.
45:         self.distribution.scripts = self.scripts
46: 
47:         return old_build_scripts.run(self)
48: 
49:     def get_source_files(self):
50:         from numpy.distutils.misc_util import get_script_files
51:         return get_script_files(self.scripts)
52: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_55170 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 3, (-1)), 'str', ' Modified version of build_scripts that handles building scripts from functions.\n\n')
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.command.build_scripts import old_build_scripts' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55171 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.build_scripts')

if (type(import_55171) is not StypyTypeError):

    if (import_55171 != 'pyd_module'):
        __import__(import_55171)
        sys_modules_55172 = sys.modules[import_55171]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.build_scripts', sys_modules_55172.module_type_store, module_type_store, ['build_scripts'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_55172, sys_modules_55172.module_type_store, module_type_store)
    else:
        from distutils.command.build_scripts import build_scripts as old_build_scripts

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.build_scripts', None, module_type_store, ['build_scripts'], [old_build_scripts])

else:
    # Assigning a type to the variable 'distutils.command.build_scripts' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.command.build_scripts', import_55171)

# Adding an alias
module_type_store.add_alias('old_build_scripts', 'build_scripts')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.distutils import log' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55173 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils')

if (type(import_55173) is not StypyTypeError):

    if (import_55173 != 'pyd_module'):
        __import__(import_55173)
        sys_modules_55174 = sys.modules[import_55173]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils', sys_modules_55174.module_type_store, module_type_store, ['log'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_55174, sys_modules_55174.module_type_store, module_type_store)
    else:
        from numpy.distutils import log

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils', None, module_type_store, ['log'], [log])

else:
    # Assigning a type to the variable 'numpy.distutils' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils', import_55173)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from numpy.distutils.misc_util import is_string' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55175 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.misc_util')

if (type(import_55175) is not StypyTypeError):

    if (import_55175 != 'pyd_module'):
        __import__(import_55175)
        sys_modules_55176 = sys.modules[import_55175]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.misc_util', sys_modules_55176.module_type_store, module_type_store, ['is_string'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_55176, sys_modules_55176.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import is_string

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.misc_util', None, module_type_store, ['is_string'], [is_string])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'numpy.distutils.misc_util', import_55175)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'build_scripts' class
# Getting the type of 'old_build_scripts' (line 10)
old_build_scripts_55177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 20), 'old_build_scripts')

class build_scripts(old_build_scripts_55177, ):

    @norecursion
    def generate_scripts(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'generate_scripts'
        module_type_store = module_type_store.open_function_context('generate_scripts', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_localization', localization)
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_function_name', 'build_scripts.generate_scripts')
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_param_names_list', ['scripts'])
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_scripts.generate_scripts.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.generate_scripts', ['scripts'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'generate_scripts', localization, ['scripts'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'generate_scripts(...)' code ##################

        
        # Assigning a List to a Name (line 13):
        
        # Obtaining an instance of the builtin type 'list' (line 13)
        list_55178 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'list')
        # Adding type elements to the builtin type 'list' instance (line 13)
        
        # Assigning a type to the variable 'new_scripts' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'new_scripts', list_55178)
        
        # Assigning a List to a Name (line 14):
        
        # Obtaining an instance of the builtin type 'list' (line 14)
        list_55179 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 23), 'list')
        # Adding type elements to the builtin type 'list' instance (line 14)
        
        # Assigning a type to the variable 'func_scripts' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'func_scripts', list_55179)
        
        # Getting the type of 'scripts' (line 15)
        scripts_55180 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'scripts')
        # Testing the type of a for loop iterable (line 15)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 15, 8), scripts_55180)
        # Getting the type of the for loop variable (line 15)
        for_loop_var_55181 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 15, 8), scripts_55180)
        # Assigning a type to the variable 'script' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'script', for_loop_var_55181)
        # SSA begins for a for statement (line 15)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        
        # Call to is_string(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'script' (line 16)
        script_55183 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 25), 'script', False)
        # Processing the call keyword arguments (line 16)
        kwargs_55184 = {}
        # Getting the type of 'is_string' (line 16)
        is_string_55182 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 15), 'is_string', False)
        # Calling is_string(args, kwargs) (line 16)
        is_string_call_result_55185 = invoke(stypy.reporting.localization.Localization(__file__, 16, 15), is_string_55182, *[script_55183], **kwargs_55184)
        
        # Testing the type of an if condition (line 16)
        if_condition_55186 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 16, 12), is_string_call_result_55185)
        # Assigning a type to the variable 'if_condition_55186' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 12), 'if_condition_55186', if_condition_55186)
        # SSA begins for if statement (line 16)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'script' (line 17)
        script_55189 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 35), 'script', False)
        # Processing the call keyword arguments (line 17)
        kwargs_55190 = {}
        # Getting the type of 'new_scripts' (line 17)
        new_scripts_55187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 16), 'new_scripts', False)
        # Obtaining the member 'append' of a type (line 17)
        append_55188 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 16), new_scripts_55187, 'append')
        # Calling append(args, kwargs) (line 17)
        append_call_result_55191 = invoke(stypy.reporting.localization.Localization(__file__, 17, 16), append_55188, *[script_55189], **kwargs_55190)
        
        # SSA branch for the else part of an if statement (line 16)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'script' (line 19)
        script_55194 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 36), 'script', False)
        # Processing the call keyword arguments (line 19)
        kwargs_55195 = {}
        # Getting the type of 'func_scripts' (line 19)
        func_scripts_55192 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 16), 'func_scripts', False)
        # Obtaining the member 'append' of a type (line 19)
        append_55193 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 16), func_scripts_55192, 'append')
        # Calling append(args, kwargs) (line 19)
        append_call_result_55196 = invoke(stypy.reporting.localization.Localization(__file__, 19, 16), append_55193, *[script_55194], **kwargs_55195)
        
        # SSA join for if statement (line 16)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Getting the type of 'func_scripts' (line 20)
        func_scripts_55197 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 15), 'func_scripts')
        # Applying the 'not' unary operator (line 20)
        result_not__55198 = python_operator(stypy.reporting.localization.Localization(__file__, 20, 11), 'not', func_scripts_55197)
        
        # Testing the type of an if condition (line 20)
        if_condition_55199 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 20, 8), result_not__55198)
        # Assigning a type to the variable 'if_condition_55199' (line 20)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'if_condition_55199', if_condition_55199)
        # SSA begins for if statement (line 20)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Getting the type of 'new_scripts' (line 21)
        new_scripts_55200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 19), 'new_scripts')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'stypy_return_type', new_scripts_55200)
        # SSA join for if statement (line 20)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 23):
        # Getting the type of 'self' (line 23)
        self_55201 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 20), 'self')
        # Obtaining the member 'build_dir' of a type (line 23)
        build_dir_55202 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 20), self_55201, 'build_dir')
        # Assigning a type to the variable 'build_dir' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'build_dir', build_dir_55202)
        
        # Call to mkpath(...): (line 24)
        # Processing the call arguments (line 24)
        # Getting the type of 'build_dir' (line 24)
        build_dir_55205 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 20), 'build_dir', False)
        # Processing the call keyword arguments (line 24)
        kwargs_55206 = {}
        # Getting the type of 'self' (line 24)
        self_55203 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 24)
        mkpath_55204 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 8), self_55203, 'mkpath')
        # Calling mkpath(args, kwargs) (line 24)
        mkpath_call_result_55207 = invoke(stypy.reporting.localization.Localization(__file__, 24, 8), mkpath_55204, *[build_dir_55205], **kwargs_55206)
        
        
        # Getting the type of 'func_scripts' (line 25)
        func_scripts_55208 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 20), 'func_scripts')
        # Testing the type of a for loop iterable (line 25)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 25, 8), func_scripts_55208)
        # Getting the type of the for loop variable (line 25)
        for_loop_var_55209 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 25, 8), func_scripts_55208)
        # Assigning a type to the variable 'func' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'func', for_loop_var_55209)
        # SSA begins for a for statement (line 25)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Name (line 26):
        
        # Call to func(...): (line 26)
        # Processing the call arguments (line 26)
        # Getting the type of 'build_dir' (line 26)
        build_dir_55211 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 26), 'build_dir', False)
        # Processing the call keyword arguments (line 26)
        kwargs_55212 = {}
        # Getting the type of 'func' (line 26)
        func_55210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 21), 'func', False)
        # Calling func(args, kwargs) (line 26)
        func_call_result_55213 = invoke(stypy.reporting.localization.Localization(__file__, 26, 21), func_55210, *[build_dir_55211], **kwargs_55212)
        
        # Assigning a type to the variable 'script' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'script', func_call_result_55213)
        
        
        # Getting the type of 'script' (line 27)
        script_55214 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 19), 'script')
        # Applying the 'not' unary operator (line 27)
        result_not__55215 = python_operator(stypy.reporting.localization.Localization(__file__, 27, 15), 'not', script_55214)
        
        # Testing the type of an if condition (line 27)
        if_condition_55216 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 12), result_not__55215)
        # Assigning a type to the variable 'if_condition_55216' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 12), 'if_condition_55216', if_condition_55216)
        # SSA begins for if statement (line 27)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # SSA join for if statement (line 27)
        module_type_store = module_type_store.join_ssa_context()
        
        
        
        # Call to is_string(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'script' (line 29)
        script_55218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'script', False)
        # Processing the call keyword arguments (line 29)
        kwargs_55219 = {}
        # Getting the type of 'is_string' (line 29)
        is_string_55217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'is_string', False)
        # Calling is_string(args, kwargs) (line 29)
        is_string_call_result_55220 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), is_string_55217, *[script_55218], **kwargs_55219)
        
        # Testing the type of an if condition (line 29)
        if_condition_55221 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 29, 12), is_string_call_result_55220)
        # Assigning a type to the variable 'if_condition_55221' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'if_condition_55221', if_condition_55221)
        # SSA begins for if statement (line 29)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to info(...): (line 30)
        # Processing the call arguments (line 30)
        str_55224 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 25), 'str', "  adding '%s' to scripts")
        
        # Obtaining an instance of the builtin type 'tuple' (line 30)
        tuple_55225 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 55), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 30)
        # Adding element type (line 30)
        # Getting the type of 'script' (line 30)
        script_55226 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 55), 'script', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 55), tuple_55225, script_55226)
        
        # Applying the binary operator '%' (line 30)
        result_mod_55227 = python_operator(stypy.reporting.localization.Localization(__file__, 30, 25), '%', str_55224, tuple_55225)
        
        # Processing the call keyword arguments (line 30)
        kwargs_55228 = {}
        # Getting the type of 'log' (line 30)
        log_55222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 16), 'log', False)
        # Obtaining the member 'info' of a type (line 30)
        info_55223 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 16), log_55222, 'info')
        # Calling info(args, kwargs) (line 30)
        info_call_result_55229 = invoke(stypy.reporting.localization.Localization(__file__, 30, 16), info_55223, *[result_mod_55227], **kwargs_55228)
        
        
        # Call to append(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'script' (line 31)
        script_55232 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 35), 'script', False)
        # Processing the call keyword arguments (line 31)
        kwargs_55233 = {}
        # Getting the type of 'new_scripts' (line 31)
        new_scripts_55230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'new_scripts', False)
        # Obtaining the member 'append' of a type (line 31)
        append_55231 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 16), new_scripts_55230, 'append')
        # Calling append(args, kwargs) (line 31)
        append_call_result_55234 = invoke(stypy.reporting.localization.Localization(__file__, 31, 16), append_55231, *[script_55232], **kwargs_55233)
        
        # SSA branch for the else part of an if statement (line 29)
        module_type_store.open_ssa_branch('else')
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'script' (line 33)
        script_55243 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 70), 'script')
        comprehension_55244 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 17), script_55243)
        # Assigning a type to the variable 's' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 's', comprehension_55244)
        
        # Call to info(...): (line 33)
        # Processing the call arguments (line 33)
        str_55237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 26), 'str', "  adding '%s' to scripts")
        
        # Obtaining an instance of the builtin type 'tuple' (line 33)
        tuple_55238 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 56), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 33)
        # Adding element type (line 33)
        # Getting the type of 's' (line 33)
        s_55239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 56), 's', False)
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 56), tuple_55238, s_55239)
        
        # Applying the binary operator '%' (line 33)
        result_mod_55240 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 26), '%', str_55237, tuple_55238)
        
        # Processing the call keyword arguments (line 33)
        kwargs_55241 = {}
        # Getting the type of 'log' (line 33)
        log_55235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 17), 'log', False)
        # Obtaining the member 'info' of a type (line 33)
        info_55236 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 17), log_55235, 'info')
        # Calling info(args, kwargs) (line 33)
        info_call_result_55242 = invoke(stypy.reporting.localization.Localization(__file__, 33, 17), info_55236, *[result_mod_55240], **kwargs_55241)
        
        list_55245 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 17), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 33, 17), list_55245, info_call_result_55242)
        
        # Call to extend(...): (line 34)
        # Processing the call arguments (line 34)
        
        # Call to list(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'script' (line 34)
        script_55249 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 40), 'script', False)
        # Processing the call keyword arguments (line 34)
        kwargs_55250 = {}
        # Getting the type of 'list' (line 34)
        list_55248 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'list', False)
        # Calling list(args, kwargs) (line 34)
        list_call_result_55251 = invoke(stypy.reporting.localization.Localization(__file__, 34, 35), list_55248, *[script_55249], **kwargs_55250)
        
        # Processing the call keyword arguments (line 34)
        kwargs_55252 = {}
        # Getting the type of 'new_scripts' (line 34)
        new_scripts_55246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'new_scripts', False)
        # Obtaining the member 'extend' of a type (line 34)
        extend_55247 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), new_scripts_55246, 'extend')
        # Calling extend(args, kwargs) (line 34)
        extend_call_result_55253 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), extend_55247, *[list_call_result_55251], **kwargs_55252)
        
        # SSA join for if statement (line 29)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'new_scripts' (line 35)
        new_scripts_55254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'new_scripts')
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', new_scripts_55254)
        
        # ################# End of 'generate_scripts(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'generate_scripts' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_55255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55255)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'generate_scripts'
        return stypy_return_type_55255


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_scripts.run.__dict__.__setitem__('stypy_localization', localization)
        build_scripts.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_scripts.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_scripts.run.__dict__.__setitem__('stypy_function_name', 'build_scripts.run')
        build_scripts.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_scripts.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_scripts.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_scripts.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_scripts.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_scripts.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_scripts.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.run', [], None, None, defaults, varargs, kwargs)

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

        
        
        # Getting the type of 'self' (line 38)
        self_55256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'self')
        # Obtaining the member 'scripts' of a type (line 38)
        scripts_55257 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), self_55256, 'scripts')
        # Applying the 'not' unary operator (line 38)
        result_not__55258 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 11), 'not', scripts_55257)
        
        # Testing the type of an if condition (line 38)
        if_condition_55259 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 38, 8), result_not__55258)
        # Assigning a type to the variable 'if_condition_55259' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'if_condition_55259', if_condition_55259)
        # SSA begins for if statement (line 38)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 38)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Call to a Attribute (line 41):
        
        # Call to generate_scripts(...): (line 41)
        # Processing the call arguments (line 41)
        # Getting the type of 'self' (line 41)
        self_55262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 45), 'self', False)
        # Obtaining the member 'scripts' of a type (line 41)
        scripts_55263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 45), self_55262, 'scripts')
        # Processing the call keyword arguments (line 41)
        kwargs_55264 = {}
        # Getting the type of 'self' (line 41)
        self_55260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 23), 'self', False)
        # Obtaining the member 'generate_scripts' of a type (line 41)
        generate_scripts_55261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 23), self_55260, 'generate_scripts')
        # Calling generate_scripts(args, kwargs) (line 41)
        generate_scripts_call_result_55265 = invoke(stypy.reporting.localization.Localization(__file__, 41, 23), generate_scripts_55261, *[scripts_55263], **kwargs_55264)
        
        # Getting the type of 'self' (line 41)
        self_55266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'self')
        # Setting the type of the member 'scripts' of a type (line 41)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 8), self_55266, 'scripts', generate_scripts_call_result_55265)
        
        # Assigning a Attribute to a Attribute (line 45):
        # Getting the type of 'self' (line 45)
        self_55267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 36), 'self')
        # Obtaining the member 'scripts' of a type (line 45)
        scripts_55268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 36), self_55267, 'scripts')
        # Getting the type of 'self' (line 45)
        self_55269 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Obtaining the member 'distribution' of a type (line 45)
        distribution_55270 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_55269, 'distribution')
        # Setting the type of the member 'scripts' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), distribution_55270, 'scripts', scripts_55268)
        
        # Call to run(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_55273 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 37), 'self', False)
        # Processing the call keyword arguments (line 47)
        kwargs_55274 = {}
        # Getting the type of 'old_build_scripts' (line 47)
        old_build_scripts_55271 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 15), 'old_build_scripts', False)
        # Obtaining the member 'run' of a type (line 47)
        run_55272 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 15), old_build_scripts_55271, 'run')
        # Calling run(args, kwargs) (line 47)
        run_call_result_55275 = invoke(stypy.reporting.localization.Localization(__file__, 47, 15), run_55272, *[self_55273], **kwargs_55274)
        
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', run_call_result_55275)
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_55276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_55276


    @norecursion
    def get_source_files(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_source_files'
        module_type_store = module_type_store.open_function_context('get_source_files', 49, 4, False)
        # Assigning a type to the variable 'self' (line 50)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_scripts.get_source_files.__dict__.__setitem__('stypy_localization', localization)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_function_name', 'build_scripts.get_source_files')
        build_scripts.get_source_files.__dict__.__setitem__('stypy_param_names_list', [])
        build_scripts.get_source_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_scripts.get_source_files.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.get_source_files', [], None, None, defaults, varargs, kwargs)

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

        stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 8))
        
        # 'from numpy.distutils.misc_util import get_script_files' statement (line 50)
        update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
        import_55277 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 50, 8), 'numpy.distutils.misc_util')

        if (type(import_55277) is not StypyTypeError):

            if (import_55277 != 'pyd_module'):
                __import__(import_55277)
                sys_modules_55278 = sys.modules[import_55277]
                import_from_module(stypy.reporting.localization.Localization(__file__, 50, 8), 'numpy.distutils.misc_util', sys_modules_55278.module_type_store, module_type_store, ['get_script_files'])
                nest_module(stypy.reporting.localization.Localization(__file__, 50, 8), __file__, sys_modules_55278, sys_modules_55278.module_type_store, module_type_store)
            else:
                from numpy.distutils.misc_util import get_script_files

                import_from_module(stypy.reporting.localization.Localization(__file__, 50, 8), 'numpy.distutils.misc_util', None, module_type_store, ['get_script_files'], [get_script_files])

        else:
            # Assigning a type to the variable 'numpy.distutils.misc_util' (line 50)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 8), 'numpy.distutils.misc_util', import_55277)

        remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')
        
        
        # Call to get_script_files(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_55280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 32), 'self', False)
        # Obtaining the member 'scripts' of a type (line 51)
        scripts_55281 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 32), self_55280, 'scripts')
        # Processing the call keyword arguments (line 51)
        kwargs_55282 = {}
        # Getting the type of 'get_script_files' (line 51)
        get_script_files_55279 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'get_script_files', False)
        # Calling get_script_files(args, kwargs) (line 51)
        get_script_files_call_result_55283 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), get_script_files_55279, *[scripts_55281], **kwargs_55282)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', get_script_files_call_result_55283)
        
        # ################# End of 'get_source_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_source_files' in the type store
        # Getting the type of 'stypy_return_type' (line 49)
        stypy_return_type_55284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55284)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_source_files'
        return stypy_return_type_55284


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 10, 0, False)
        # Assigning a type to the variable 'self' (line 11)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_scripts.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_scripts' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'build_scripts', build_scripts)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
