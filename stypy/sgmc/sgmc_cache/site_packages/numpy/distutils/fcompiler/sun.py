
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from numpy.distutils.ccompiler import simple_version_match
4: from numpy.distutils.fcompiler import FCompiler
5: 
6: compilers = ['SunFCompiler']
7: 
8: class SunFCompiler(FCompiler):
9: 
10:     compiler_type = 'sun'
11:     description = 'Sun or Forte Fortran 95 Compiler'
12:     # ex:
13:     # f90: Sun WorkShop 6 update 2 Fortran 95 6.2 Patch 111690-10 2003/08/28
14:     version_match = simple_version_match(
15:                       start=r'f9[05]: (Sun|Forte|WorkShop).*Fortran 95')
16: 
17:     executables = {
18:         'version_cmd'  : ["<F90>", "-V"],
19:         'compiler_f77' : ["f90"],
20:         'compiler_fix' : ["f90", "-fixed"],
21:         'compiler_f90' : ["f90"],
22:         'linker_so'    : ["<F90>", "-Bdynamic", "-G"],
23:         'archiver'     : ["ar", "-cr"],
24:         'ranlib'       : ["ranlib"]
25:         }
26:     module_dir_switch = '-moddir='
27:     module_include_switch = '-M'
28:     pic_flags = ['-xcode=pic32']
29: 
30:     def get_flags_f77(self):
31:         ret = ["-ftrap=%none"]
32:         if (self.get_version() or '') >= '7':
33:             ret.append("-f77")
34:         else:
35:             ret.append("-fixed")
36:         return ret
37:     def get_opt(self):
38:         return ['-fast', '-dalign']
39:     def get_arch(self):
40:         return ['-xtarget=generic']
41:     def get_libraries(self):
42:         opt = []
43:         opt.extend(['fsu', 'sunmath', 'mvec'])
44:         return opt
45: 
46:     def runtime_library_dir_option(self, dir):
47:         return '-R"%s"' % dir
48: 
49: if __name__ == '__main__':
50:     from distutils import log
51:     log.set_verbosity(2)
52:     from numpy.distutils.fcompiler import new_fcompiler
53:     compiler = new_fcompiler(compiler='sun')
54:     compiler.customize()
55:     print(compiler.get_version())
56: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.distutils.ccompiler import simple_version_match' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63229 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.ccompiler')

if (type(import_63229) is not StypyTypeError):

    if (import_63229 != 'pyd_module'):
        __import__(import_63229)
        sys_modules_63230 = sys.modules[import_63229]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.ccompiler', sys_modules_63230.module_type_store, module_type_store, ['simple_version_match'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_63230, sys_modules_63230.module_type_store, module_type_store)
    else:
        from numpy.distutils.ccompiler import simple_version_match

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.ccompiler', None, module_type_store, ['simple_version_match'], [simple_version_match])

else:
    # Assigning a type to the variable 'numpy.distutils.ccompiler' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.ccompiler', import_63229)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63231 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler')

if (type(import_63231) is not StypyTypeError):

    if (import_63231 != 'pyd_module'):
        __import__(import_63231)
        sys_modules_63232 = sys.modules[import_63231]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', sys_modules_63232.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_63232, sys_modules_63232.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', import_63231)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 6):

# Obtaining an instance of the builtin type 'list' (line 6)
list_63233 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_63234 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'str', 'SunFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_63233, str_63234)

# Assigning a type to the variable 'compilers' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'compilers', list_63233)
# Declaration of the 'SunFCompiler' class
# Getting the type of 'FCompiler' (line 8)
FCompiler_63235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'FCompiler')

class SunFCompiler(FCompiler_63235, ):

    @norecursion
    def get_flags_f77(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_f77'
        module_type_store = module_type_store.open_function_context('get_flags_f77', 30, 4, False)
        # Assigning a type to the variable 'self' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_localization', localization)
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_type_store', module_type_store)
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_function_name', 'SunFCompiler.get_flags_f77')
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_param_names_list', [])
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_varargs_param_name', None)
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_defaults', defaults)
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_varargs', varargs)
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SunFCompiler.get_flags_f77.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SunFCompiler.get_flags_f77', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_f77', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_f77(...)' code ##################

        
        # Assigning a List to a Name (line 31):
        
        # Obtaining an instance of the builtin type 'list' (line 31)
        list_63236 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 31)
        # Adding element type (line 31)
        str_63237 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 15), 'str', '-ftrap=%none')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 14), list_63236, str_63237)
        
        # Assigning a type to the variable 'ret' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'ret', list_63236)
        
        
        
        # Evaluating a boolean operation
        
        # Call to get_version(...): (line 32)
        # Processing the call keyword arguments (line 32)
        kwargs_63240 = {}
        # Getting the type of 'self' (line 32)
        self_63238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 12), 'self', False)
        # Obtaining the member 'get_version' of a type (line 32)
        get_version_63239 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 12), self_63238, 'get_version')
        # Calling get_version(args, kwargs) (line 32)
        get_version_call_result_63241 = invoke(stypy.reporting.localization.Localization(__file__, 32, 12), get_version_63239, *[], **kwargs_63240)
        
        str_63242 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 34), 'str', '')
        # Applying the binary operator 'or' (line 32)
        result_or_keyword_63243 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 12), 'or', get_version_call_result_63241, str_63242)
        
        str_63244 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 41), 'str', '7')
        # Applying the binary operator '>=' (line 32)
        result_ge_63245 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 11), '>=', result_or_keyword_63243, str_63244)
        
        # Testing the type of an if condition (line 32)
        if_condition_63246 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 32, 8), result_ge_63245)
        # Assigning a type to the variable 'if_condition_63246' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'if_condition_63246', if_condition_63246)
        # SSA begins for if statement (line 32)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Call to append(...): (line 33)
        # Processing the call arguments (line 33)
        str_63249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 23), 'str', '-f77')
        # Processing the call keyword arguments (line 33)
        kwargs_63250 = {}
        # Getting the type of 'ret' (line 33)
        ret_63247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 12), 'ret', False)
        # Obtaining the member 'append' of a type (line 33)
        append_63248 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 12), ret_63247, 'append')
        # Calling append(args, kwargs) (line 33)
        append_call_result_63251 = invoke(stypy.reporting.localization.Localization(__file__, 33, 12), append_63248, *[str_63249], **kwargs_63250)
        
        # SSA branch for the else part of an if statement (line 32)
        module_type_store.open_ssa_branch('else')
        
        # Call to append(...): (line 35)
        # Processing the call arguments (line 35)
        str_63254 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 23), 'str', '-fixed')
        # Processing the call keyword arguments (line 35)
        kwargs_63255 = {}
        # Getting the type of 'ret' (line 35)
        ret_63252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'ret', False)
        # Obtaining the member 'append' of a type (line 35)
        append_63253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 12), ret_63252, 'append')
        # Calling append(args, kwargs) (line 35)
        append_call_result_63256 = invoke(stypy.reporting.localization.Localization(__file__, 35, 12), append_63253, *[str_63254], **kwargs_63255)
        
        # SSA join for if statement (line 32)
        module_type_store = module_type_store.join_ssa_context()
        
        # Getting the type of 'ret' (line 36)
        ret_63257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'ret')
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', ret_63257)
        
        # ################# End of 'get_flags_f77(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_f77' in the type store
        # Getting the type of 'stypy_return_type' (line 30)
        stypy_return_type_63258 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63258)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_f77'
        return stypy_return_type_63258


    @norecursion
    def get_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_opt'
        module_type_store = module_type_store.open_function_context('get_opt', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_localization', localization)
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_function_name', 'SunFCompiler.get_opt')
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_param_names_list', [])
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SunFCompiler.get_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SunFCompiler.get_opt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_opt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_opt(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_63259 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        str_63260 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'str', '-fast')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), list_63259, str_63260)
        # Adding element type (line 38)
        str_63261 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 25), 'str', '-dalign')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), list_63259, str_63261)
        
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', list_63259)
        
        # ################# End of 'get_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_63262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63262)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_opt'
        return stypy_return_type_63262


    @norecursion
    def get_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_arch'
        module_type_store = module_type_store.open_function_context('get_arch', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_localization', localization)
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_function_name', 'SunFCompiler.get_arch')
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_param_names_list', [])
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SunFCompiler.get_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SunFCompiler.get_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_arch(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 40)
        list_63263 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 40)
        # Adding element type (line 40)
        str_63264 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 40, 16), 'str', '-xtarget=generic')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 40, 15), list_63263, str_63264)
        
        # Assigning a type to the variable 'stypy_return_type' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'stypy_return_type', list_63263)
        
        # ################# End of 'get_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_63265 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63265)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_arch'
        return stypy_return_type_63265


    @norecursion
    def get_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libraries'
        module_type_store = module_type_store.open_function_context('get_libraries', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_localization', localization)
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_function_name', 'SunFCompiler.get_libraries')
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SunFCompiler.get_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SunFCompiler.get_libraries', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_libraries', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_libraries(...)' code ##################

        
        # Assigning a List to a Name (line 42):
        
        # Obtaining an instance of the builtin type 'list' (line 42)
        list_63266 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 42)
        
        # Assigning a type to the variable 'opt' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'opt', list_63266)
        
        # Call to extend(...): (line 43)
        # Processing the call arguments (line 43)
        
        # Obtaining an instance of the builtin type 'list' (line 43)
        list_63269 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 19), 'list')
        # Adding type elements to the builtin type 'list' instance (line 43)
        # Adding element type (line 43)
        str_63270 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 20), 'str', 'fsu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_63269, str_63270)
        # Adding element type (line 43)
        str_63271 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 27), 'str', 'sunmath')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_63269, str_63271)
        # Adding element type (line 43)
        str_63272 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 38), 'str', 'mvec')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 43, 19), list_63269, str_63272)
        
        # Processing the call keyword arguments (line 43)
        kwargs_63273 = {}
        # Getting the type of 'opt' (line 43)
        opt_63267 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'opt', False)
        # Obtaining the member 'extend' of a type (line 43)
        extend_63268 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 8), opt_63267, 'extend')
        # Calling extend(args, kwargs) (line 43)
        extend_call_result_63274 = invoke(stypy.reporting.localization.Localization(__file__, 43, 8), extend_63268, *[list_63269], **kwargs_63273)
        
        # Getting the type of 'opt' (line 44)
        opt_63275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'stypy_return_type', opt_63275)
        
        # ################# End of 'get_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_63276 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63276)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libraries'
        return stypy_return_type_63276


    @norecursion
    def runtime_library_dir_option(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'runtime_library_dir_option'
        module_type_store = module_type_store.open_function_context('runtime_library_dir_option', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_localization', localization)
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_type_store', module_type_store)
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_function_name', 'SunFCompiler.runtime_library_dir_option')
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_param_names_list', ['dir'])
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_varargs_param_name', None)
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_kwargs_param_name', None)
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_defaults', defaults)
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_varargs', varargs)
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        SunFCompiler.runtime_library_dir_option.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SunFCompiler.runtime_library_dir_option', ['dir'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'runtime_library_dir_option', localization, ['dir'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'runtime_library_dir_option(...)' code ##################

        str_63277 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 47, 15), 'str', '-R"%s"')
        # Getting the type of 'dir' (line 47)
        dir_63278 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 26), 'dir')
        # Applying the binary operator '%' (line 47)
        result_mod_63279 = python_operator(stypy.reporting.localization.Localization(__file__, 47, 15), '%', str_63277, dir_63278)
        
        # Assigning a type to the variable 'stypy_return_type' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'stypy_return_type', result_mod_63279)
        
        # ################# End of 'runtime_library_dir_option(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'runtime_library_dir_option' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_63280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63280)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'runtime_library_dir_option'
        return stypy_return_type_63280


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 8, 0, False)
        # Assigning a type to the variable 'self' (line 9)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'SunFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'SunFCompiler' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'SunFCompiler', SunFCompiler)

# Assigning a Str to a Name (line 10):
str_63281 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'str', 'sun')
# Getting the type of 'SunFCompiler'
SunFCompiler_63282 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SunFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SunFCompiler_63282, 'compiler_type', str_63281)

# Assigning a Str to a Name (line 11):
str_63283 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 18), 'str', 'Sun or Forte Fortran 95 Compiler')
# Getting the type of 'SunFCompiler'
SunFCompiler_63284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SunFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SunFCompiler_63284, 'description', str_63283)

# Assigning a Call to a Name (line 14):

# Call to simple_version_match(...): (line 14)
# Processing the call keyword arguments (line 14)
str_63286 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 28), 'str', 'f9[05]: (Sun|Forte|WorkShop).*Fortran 95')
keyword_63287 = str_63286
kwargs_63288 = {'start': keyword_63287}
# Getting the type of 'simple_version_match' (line 14)
simple_version_match_63285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 20), 'simple_version_match', False)
# Calling simple_version_match(args, kwargs) (line 14)
simple_version_match_call_result_63289 = invoke(stypy.reporting.localization.Localization(__file__, 14, 20), simple_version_match_63285, *[], **kwargs_63288)

# Getting the type of 'SunFCompiler'
SunFCompiler_63290 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SunFCompiler')
# Setting the type of the member 'version_match' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SunFCompiler_63290, 'version_match', simple_version_match_call_result_63289)

# Assigning a Dict to a Name (line 17):

# Obtaining an instance of the builtin type 'dict' (line 17)
dict_63291 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 17)
# Adding element type (key, value) (line 17)
str_63292 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 18)
list_63293 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_63294 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_63293, str_63294)
# Adding element type (line 18)
str_63295 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 35), 'str', '-V')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_63293, str_63295)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_63291, (str_63292, list_63293))
# Adding element type (key, value) (line 17)
str_63296 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 19)
list_63297 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_63298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_63297, str_63298)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_63291, (str_63296, list_63297))
# Adding element type (key, value) (line 17)
str_63299 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 20)
list_63300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_63301 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_63300, str_63301)
# Adding element type (line 20)
str_63302 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 33), 'str', '-fixed')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_63300, str_63302)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_63291, (str_63299, list_63300))
# Adding element type (key, value) (line 17)
str_63303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 21)
list_63304 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_63305 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_63304, str_63305)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_63291, (str_63303, list_63304))
# Adding element type (key, value) (line 17)
str_63306 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 22)
list_63307 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_63308 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_63307, str_63308)
# Adding element type (line 22)
str_63309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 35), 'str', '-Bdynamic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_63307, str_63309)
# Adding element type (line 22)
str_63310 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 48), 'str', '-G')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_63307, str_63310)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_63291, (str_63306, list_63307))
# Adding element type (key, value) (line 17)
str_63311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 23)
list_63312 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
str_63313 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_63312, str_63313)
# Adding element type (line 23)
str_63314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_63312, str_63314)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_63291, (str_63311, list_63312))
# Adding element type (key, value) (line 17)
str_63315 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 24)
list_63316 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
str_63317 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 25), list_63316, str_63317)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 18), dict_63291, (str_63315, list_63316))

# Getting the type of 'SunFCompiler'
SunFCompiler_63318 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SunFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SunFCompiler_63318, 'executables', dict_63291)

# Assigning a Str to a Name (line 26):
str_63319 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 24), 'str', '-moddir=')
# Getting the type of 'SunFCompiler'
SunFCompiler_63320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SunFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SunFCompiler_63320, 'module_dir_switch', str_63319)

# Assigning a Str to a Name (line 27):
str_63321 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 28), 'str', '-M')
# Getting the type of 'SunFCompiler'
SunFCompiler_63322 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SunFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SunFCompiler_63322, 'module_include_switch', str_63321)

# Assigning a List to a Name (line 28):

# Obtaining an instance of the builtin type 'list' (line 28)
list_63323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 28)
# Adding element type (line 28)
str_63324 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 17), 'str', '-xcode=pic32')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 16), list_63323, str_63324)

# Getting the type of 'SunFCompiler'
SunFCompiler_63325 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'SunFCompiler')
# Setting the type of the member 'pic_flags' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), SunFCompiler_63325, 'pic_flags', list_63323)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 50, 4))
    
    # 'from distutils import log' statement (line 50)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 50, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 51)
    # Processing the call arguments (line 51)
    int_63328 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 22), 'int')
    # Processing the call keyword arguments (line 51)
    kwargs_63329 = {}
    # Getting the type of 'log' (line 51)
    log_63326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 51)
    set_verbosity_63327 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 4), log_63326, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 51)
    set_verbosity_call_result_63330 = invoke(stypy.reporting.localization.Localization(__file__, 51, 4), set_verbosity_63327, *[int_63328], **kwargs_63329)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 52, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 52)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_63331 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy.distutils.fcompiler')

    if (type(import_63331) is not StypyTypeError):

        if (import_63331 != 'pyd_module'):
            __import__(import_63331)
            sys_modules_63332 = sys.modules[import_63331]
            import_from_module(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy.distutils.fcompiler', sys_modules_63332.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 52, 4), __file__, sys_modules_63332, sys_modules_63332.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 52)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'numpy.distutils.fcompiler', import_63331)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 53):
    
    # Call to new_fcompiler(...): (line 53)
    # Processing the call keyword arguments (line 53)
    str_63334 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, 38), 'str', 'sun')
    keyword_63335 = str_63334
    kwargs_63336 = {'compiler': keyword_63335}
    # Getting the type of 'new_fcompiler' (line 53)
    new_fcompiler_63333 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 53)
    new_fcompiler_call_result_63337 = invoke(stypy.reporting.localization.Localization(__file__, 53, 15), new_fcompiler_63333, *[], **kwargs_63336)
    
    # Assigning a type to the variable 'compiler' (line 53)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'compiler', new_fcompiler_call_result_63337)
    
    # Call to customize(...): (line 54)
    # Processing the call keyword arguments (line 54)
    kwargs_63340 = {}
    # Getting the type of 'compiler' (line 54)
    compiler_63338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 54)
    customize_63339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 4), compiler_63338, 'customize')
    # Calling customize(args, kwargs) (line 54)
    customize_call_result_63341 = invoke(stypy.reporting.localization.Localization(__file__, 54, 4), customize_63339, *[], **kwargs_63340)
    
    
    # Call to print(...): (line 55)
    # Processing the call arguments (line 55)
    
    # Call to get_version(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_63345 = {}
    # Getting the type of 'compiler' (line 55)
    compiler_63343 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 55)
    get_version_63344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 10), compiler_63343, 'get_version')
    # Calling get_version(args, kwargs) (line 55)
    get_version_call_result_63346 = invoke(stypy.reporting.localization.Localization(__file__, 55, 10), get_version_63344, *[], **kwargs_63345)
    
    # Processing the call keyword arguments (line 55)
    kwargs_63347 = {}
    # Getting the type of 'print' (line 55)
    print_63342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'print', False)
    # Calling print(args, kwargs) (line 55)
    print_call_result_63348 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), print_63342, *[get_version_call_result_63346], **kwargs_63347)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
