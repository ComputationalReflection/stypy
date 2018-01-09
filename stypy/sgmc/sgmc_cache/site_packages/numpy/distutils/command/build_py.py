
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from distutils.command.build_py import build_py as old_build_py
4: from numpy.distutils.misc_util import is_string
5: 
6: class build_py(old_build_py):
7: 
8:     def run(self):
9:         build_src = self.get_finalized_command('build_src')
10:         if build_src.py_modules_dict and self.packages is None:
11:             self.packages = list(build_src.py_modules_dict.keys ())
12:         old_build_py.run(self)
13: 
14:     def find_package_modules(self, package, package_dir):
15:         modules = old_build_py.find_package_modules(self, package, package_dir)
16: 
17:         # Find build_src generated *.py files.
18:         build_src = self.get_finalized_command('build_src')
19:         modules += build_src.py_modules_dict.get(package, [])
20: 
21:         return modules
22: 
23:     def find_modules(self):
24:         old_py_modules = self.py_modules[:]
25:         new_py_modules = [_m for _m in self.py_modules if is_string(_m)]
26:         self.py_modules[:] = new_py_modules
27:         modules = old_build_py.find_modules(self)
28:         self.py_modules[:] = old_py_modules
29: 
30:         return modules
31: 
32:     # XXX: Fix find_source_files for item in py_modules such that item is 3-tuple
33:     # and item[2] is source file.
34: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from distutils.command.build_py import old_build_py' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55085 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.command.build_py')

if (type(import_55085) is not StypyTypeError):

    if (import_55085 != 'pyd_module'):
        __import__(import_55085)
        sys_modules_55086 = sys.modules[import_55085]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.command.build_py', sys_modules_55086.module_type_store, module_type_store, ['build_py'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_55086, sys_modules_55086.module_type_store, module_type_store)
    else:
        from distutils.command.build_py import build_py as old_build_py

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.command.build_py', None, module_type_store, ['build_py'], [old_build_py])

else:
    # Assigning a type to the variable 'distutils.command.build_py' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.command.build_py', import_55085)

# Adding an alias
module_type_store.add_alias('old_build_py', 'build_py')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.distutils.misc_util import is_string' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_55087 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.misc_util')

if (type(import_55087) is not StypyTypeError):

    if (import_55087 != 'pyd_module'):
        __import__(import_55087)
        sys_modules_55088 = sys.modules[import_55087]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.misc_util', sys_modules_55088.module_type_store, module_type_store, ['is_string'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_55088, sys_modules_55088.module_type_store, module_type_store)
    else:
        from numpy.distutils.misc_util import is_string

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.misc_util', None, module_type_store, ['is_string'], [is_string])

else:
    # Assigning a type to the variable 'numpy.distutils.misc_util' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.misc_util', import_55087)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'build_py' class
# Getting the type of 'old_build_py' (line 6)
old_build_py_55089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 6, 15), 'old_build_py')

class build_py(old_build_py_55089, ):

    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 8, 4, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.run.__dict__.__setitem__('stypy_localization', localization)
        build_py.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.run.__dict__.__setitem__('stypy_function_name', 'build_py.run')
        build_py.run.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Call to a Name (line 9):
        
        # Call to get_finalized_command(...): (line 9)
        # Processing the call arguments (line 9)
        str_55092 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 47), 'str', 'build_src')
        # Processing the call keyword arguments (line 9)
        kwargs_55093 = {}
        # Getting the type of 'self' (line 9)
        self_55090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 9)
        get_finalized_command_55091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 9, 20), self_55090, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 9)
        get_finalized_command_call_result_55094 = invoke(stypy.reporting.localization.Localization(__file__, 9, 20), get_finalized_command_55091, *[str_55092], **kwargs_55093)
        
        # Assigning a type to the variable 'build_src' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 8), 'build_src', get_finalized_command_call_result_55094)
        
        
        # Evaluating a boolean operation
        # Getting the type of 'build_src' (line 10)
        build_src_55095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 11), 'build_src')
        # Obtaining the member 'py_modules_dict' of a type (line 10)
        py_modules_dict_55096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 11), build_src_55095, 'py_modules_dict')
        
        # Getting the type of 'self' (line 10)
        self_55097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 41), 'self')
        # Obtaining the member 'packages' of a type (line 10)
        packages_55098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 41), self_55097, 'packages')
        # Getting the type of 'None' (line 10)
        None_55099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 58), 'None')
        # Applying the binary operator 'is' (line 10)
        result_is__55100 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 41), 'is', packages_55098, None_55099)
        
        # Applying the binary operator 'and' (line 10)
        result_and_keyword_55101 = python_operator(stypy.reporting.localization.Localization(__file__, 10, 11), 'and', py_modules_dict_55096, result_is__55100)
        
        # Testing the type of an if condition (line 10)
        if_condition_55102 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 10, 8), result_and_keyword_55101)
        # Assigning a type to the variable 'if_condition_55102' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'if_condition_55102', if_condition_55102)
        # SSA begins for if statement (line 10)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Assigning a Call to a Attribute (line 11):
        
        # Call to list(...): (line 11)
        # Processing the call arguments (line 11)
        
        # Call to keys(...): (line 11)
        # Processing the call keyword arguments (line 11)
        kwargs_55107 = {}
        # Getting the type of 'build_src' (line 11)
        build_src_55104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 33), 'build_src', False)
        # Obtaining the member 'py_modules_dict' of a type (line 11)
        py_modules_dict_55105 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 33), build_src_55104, 'py_modules_dict')
        # Obtaining the member 'keys' of a type (line 11)
        keys_55106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 33), py_modules_dict_55105, 'keys')
        # Calling keys(args, kwargs) (line 11)
        keys_call_result_55108 = invoke(stypy.reporting.localization.Localization(__file__, 11, 33), keys_55106, *[], **kwargs_55107)
        
        # Processing the call keyword arguments (line 11)
        kwargs_55109 = {}
        # Getting the type of 'list' (line 11)
        list_55103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 28), 'list', False)
        # Calling list(args, kwargs) (line 11)
        list_call_result_55110 = invoke(stypy.reporting.localization.Localization(__file__, 11, 28), list_55103, *[keys_call_result_55108], **kwargs_55109)
        
        # Getting the type of 'self' (line 11)
        self_55111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 12), 'self')
        # Setting the type of the member 'packages' of a type (line 11)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 11, 12), self_55111, 'packages', list_call_result_55110)
        # SSA join for if statement (line 10)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to run(...): (line 12)
        # Processing the call arguments (line 12)
        # Getting the type of 'self' (line 12)
        self_55114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 25), 'self', False)
        # Processing the call keyword arguments (line 12)
        kwargs_55115 = {}
        # Getting the type of 'old_build_py' (line 12)
        old_build_py_55112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'old_build_py', False)
        # Obtaining the member 'run' of a type (line 12)
        run_55113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), old_build_py_55112, 'run')
        # Calling run(args, kwargs) (line 12)
        run_call_result_55116 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), run_55113, *[self_55114], **kwargs_55115)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 8)
        stypy_return_type_55117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55117)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_55117


    @norecursion
    def find_package_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_package_modules'
        module_type_store = module_type_store.open_function_context('find_package_modules', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.find_package_modules.__dict__.__setitem__('stypy_localization', localization)
        build_py.find_package_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.find_package_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.find_package_modules.__dict__.__setitem__('stypy_function_name', 'build_py.find_package_modules')
        build_py.find_package_modules.__dict__.__setitem__('stypy_param_names_list', ['package', 'package_dir'])
        build_py.find_package_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.find_package_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.find_package_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.find_package_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.find_package_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.find_package_modules.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.find_package_modules', ['package', 'package_dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_package_modules', localization, ['package', 'package_dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_package_modules(...)' code ##################

        
        # Assigning a Call to a Name (line 15):
        
        # Call to find_package_modules(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'self' (line 15)
        self_55120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 52), 'self', False)
        # Getting the type of 'package' (line 15)
        package_55121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 58), 'package', False)
        # Getting the type of 'package_dir' (line 15)
        package_dir_55122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 67), 'package_dir', False)
        # Processing the call keyword arguments (line 15)
        kwargs_55123 = {}
        # Getting the type of 'old_build_py' (line 15)
        old_build_py_55118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 18), 'old_build_py', False)
        # Obtaining the member 'find_package_modules' of a type (line 15)
        find_package_modules_55119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 18), old_build_py_55118, 'find_package_modules')
        # Calling find_package_modules(args, kwargs) (line 15)
        find_package_modules_call_result_55124 = invoke(stypy.reporting.localization.Localization(__file__, 15, 18), find_package_modules_55119, *[self_55120, package_55121, package_dir_55122], **kwargs_55123)
        
        # Assigning a type to the variable 'modules' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'modules', find_package_modules_call_result_55124)
        
        # Assigning a Call to a Name (line 18):
        
        # Call to get_finalized_command(...): (line 18)
        # Processing the call arguments (line 18)
        str_55127 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 47), 'str', 'build_src')
        # Processing the call keyword arguments (line 18)
        kwargs_55128 = {}
        # Getting the type of 'self' (line 18)
        self_55125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 20), 'self', False)
        # Obtaining the member 'get_finalized_command' of a type (line 18)
        get_finalized_command_55126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 20), self_55125, 'get_finalized_command')
        # Calling get_finalized_command(args, kwargs) (line 18)
        get_finalized_command_call_result_55129 = invoke(stypy.reporting.localization.Localization(__file__, 18, 20), get_finalized_command_55126, *[str_55127], **kwargs_55128)
        
        # Assigning a type to the variable 'build_src' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'build_src', get_finalized_command_call_result_55129)
        
        # Getting the type of 'modules' (line 19)
        modules_55130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'modules')
        
        # Call to get(...): (line 19)
        # Processing the call arguments (line 19)
        # Getting the type of 'package' (line 19)
        package_55134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 49), 'package', False)
        
        # Obtaining an instance of the builtin type 'list' (line 19)
        list_55135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 58), 'list')
        # Adding type elements to the builtin type 'list' instance (line 19)
        
        # Processing the call keyword arguments (line 19)
        kwargs_55136 = {}
        # Getting the type of 'build_src' (line 19)
        build_src_55131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'build_src', False)
        # Obtaining the member 'py_modules_dict' of a type (line 19)
        py_modules_dict_55132 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), build_src_55131, 'py_modules_dict')
        # Obtaining the member 'get' of a type (line 19)
        get_55133 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), py_modules_dict_55132, 'get')
        # Calling get(args, kwargs) (line 19)
        get_call_result_55137 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), get_55133, *[package_55134, list_55135], **kwargs_55136)
        
        # Applying the binary operator '+=' (line 19)
        result_iadd_55138 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 8), '+=', modules_55130, get_call_result_55137)
        # Assigning a type to the variable 'modules' (line 19)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'modules', result_iadd_55138)
        
        # Getting the type of 'modules' (line 21)
        modules_55139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'modules')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', modules_55139)
        
        # ################# End of 'find_package_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_package_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_55140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55140)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_package_modules'
        return stypy_return_type_55140


    @norecursion
    def find_modules(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_modules'
        module_type_store = module_type_store.open_function_context('find_modules', 23, 4, False)
        # Assigning a type to the variable 'self' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build_py.find_modules.__dict__.__setitem__('stypy_localization', localization)
        build_py.find_modules.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build_py.find_modules.__dict__.__setitem__('stypy_type_store', module_type_store)
        build_py.find_modules.__dict__.__setitem__('stypy_function_name', 'build_py.find_modules')
        build_py.find_modules.__dict__.__setitem__('stypy_param_names_list', [])
        build_py.find_modules.__dict__.__setitem__('stypy_varargs_param_name', None)
        build_py.find_modules.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build_py.find_modules.__dict__.__setitem__('stypy_call_defaults', defaults)
        build_py.find_modules.__dict__.__setitem__('stypy_call_varargs', varargs)
        build_py.find_modules.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build_py.find_modules.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.find_modules', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_modules', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_modules(...)' code ##################

        
        # Assigning a Subscript to a Name (line 24):
        
        # Obtaining the type of the subscript
        slice_55141 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 24, 25), None, None, None)
        # Getting the type of 'self' (line 24)
        self_55142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 25), 'self')
        # Obtaining the member 'py_modules' of a type (line 24)
        py_modules_55143 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 25), self_55142, 'py_modules')
        # Obtaining the member '__getitem__' of a type (line 24)
        getitem___55144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 24, 25), py_modules_55143, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 24)
        subscript_call_result_55145 = invoke(stypy.reporting.localization.Localization(__file__, 24, 25), getitem___55144, slice_55141)
        
        # Assigning a type to the variable 'old_py_modules' (line 24)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 24, 8), 'old_py_modules', subscript_call_result_55145)
        
        # Assigning a ListComp to a Name (line 25):
        # Calculating list comprehension
        # Calculating comprehension expression
        # Getting the type of 'self' (line 25)
        self_55151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 39), 'self')
        # Obtaining the member 'py_modules' of a type (line 25)
        py_modules_55152 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 39), self_55151, 'py_modules')
        comprehension_55153 = get_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 26), py_modules_55152)
        # Assigning a type to the variable '_m' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), '_m', comprehension_55153)
        
        # Call to is_string(...): (line 25)
        # Processing the call arguments (line 25)
        # Getting the type of '_m' (line 25)
        _m_55148 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 68), '_m', False)
        # Processing the call keyword arguments (line 25)
        kwargs_55149 = {}
        # Getting the type of 'is_string' (line 25)
        is_string_55147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 58), 'is_string', False)
        # Calling is_string(args, kwargs) (line 25)
        is_string_call_result_55150 = invoke(stypy.reporting.localization.Localization(__file__, 25, 58), is_string_55147, *[_m_55148], **kwargs_55149)
        
        # Getting the type of '_m' (line 25)
        _m_55146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 26), '_m')
        list_55154 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'list')
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 26), list_55154, _m_55146)
        # Assigning a type to the variable 'new_py_modules' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'new_py_modules', list_55154)
        
        # Assigning a Name to a Subscript (line 26):
        # Getting the type of 'new_py_modules' (line 26)
        new_py_modules_55155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 29), 'new_py_modules')
        # Getting the type of 'self' (line 26)
        self_55156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Obtaining the member 'py_modules' of a type (line 26)
        py_modules_55157 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_55156, 'py_modules')
        slice_55158 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 26, 8), None, None, None)
        # Storing an element on a container (line 26)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 8), py_modules_55157, (slice_55158, new_py_modules_55155))
        
        # Assigning a Call to a Name (line 27):
        
        # Call to find_modules(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'self' (line 27)
        self_55161 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 44), 'self', False)
        # Processing the call keyword arguments (line 27)
        kwargs_55162 = {}
        # Getting the type of 'old_build_py' (line 27)
        old_build_py_55159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 18), 'old_build_py', False)
        # Obtaining the member 'find_modules' of a type (line 27)
        find_modules_55160 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 18), old_build_py_55159, 'find_modules')
        # Calling find_modules(args, kwargs) (line 27)
        find_modules_call_result_55163 = invoke(stypy.reporting.localization.Localization(__file__, 27, 18), find_modules_55160, *[self_55161], **kwargs_55162)
        
        # Assigning a type to the variable 'modules' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'modules', find_modules_call_result_55163)
        
        # Assigning a Name to a Subscript (line 28):
        # Getting the type of 'old_py_modules' (line 28)
        old_py_modules_55164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 29), 'old_py_modules')
        # Getting the type of 'self' (line 28)
        self_55165 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'self')
        # Obtaining the member 'py_modules' of a type (line 28)
        py_modules_55166 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 8), self_55165, 'py_modules')
        slice_55167 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 28, 8), None, None, None)
        # Storing an element on a container (line 28)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 8), py_modules_55166, (slice_55167, old_py_modules_55164))
        # Getting the type of 'modules' (line 30)
        modules_55168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 15), 'modules')
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', modules_55168)
        
        # ################# End of 'find_modules(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_modules' in the type store
        # Getting the type of 'stypy_return_type' (line 23)
        stypy_return_type_55169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_55169)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_modules'
        return stypy_return_type_55169


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 6, 0, False)
        # Assigning a type to the variable 'self' (line 7)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build_py.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build_py' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'build_py', build_py)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
