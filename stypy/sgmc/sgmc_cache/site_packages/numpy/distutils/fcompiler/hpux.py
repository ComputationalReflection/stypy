
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from numpy.distutils.fcompiler import FCompiler
4: 
5: compilers = ['HPUXFCompiler']
6: 
7: class HPUXFCompiler(FCompiler):
8: 
9:     compiler_type = 'hpux'
10:     description = 'HP Fortran 90 Compiler'
11:     version_pattern =  r'HP F90 (?P<version>[^\s*,]*)'
12: 
13:     executables = {
14:         'version_cmd'  : ["f90", "+version"],
15:         'compiler_f77' : ["f90"],
16:         'compiler_fix' : ["f90"],
17:         'compiler_f90' : ["f90"],
18:         'linker_so'    : ["ld", "-b"],
19:         'archiver'     : ["ar", "-cr"],
20:         'ranlib'       : ["ranlib"]
21:         }
22:     module_dir_switch = None #XXX: fix me
23:     module_include_switch = None #XXX: fix me
24:     pic_flags = ['+Z']
25:     def get_flags(self):
26:         return self.pic_flags + ['+ppu', '+DD64']
27:     def get_flags_opt(self):
28:         return ['-O3']
29:     def get_libraries(self):
30:         return ['m']
31:     def get_library_dirs(self):
32:         opt = ['/usr/lib/hpux64']
33:         return opt
34:     def get_version(self, force=0, ok_status=[256, 0, 1]):
35:         # XXX status==256 may indicate 'unrecognized option' or
36:         #     'no input file'. So, version_cmd needs more work.
37:         return FCompiler.get_version(self, force, ok_status)
38: 
39: if __name__ == '__main__':
40:     from distutils import log
41:     log.set_verbosity(10)
42:     from numpy.distutils.fcompiler import new_fcompiler
43:     compiler = new_fcompiler(compiler='hpux')
44:     compiler.customize()
45:     print(compiler.get_version())
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_61780 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler')

if (type(import_61780) is not StypyTypeError):

    if (import_61780 != 'pyd_module'):
        __import__(import_61780)
        sys_modules_61781 = sys.modules[import_61780]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', sys_modules_61781.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_61781, sys_modules_61781.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', import_61780)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 5):

# Obtaining an instance of the builtin type 'list' (line 5)
list_61782 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
str_61783 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'str', 'HPUXFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_61782, str_61783)

# Assigning a type to the variable 'compilers' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'compilers', list_61782)
# Declaration of the 'HPUXFCompiler' class
# Getting the type of 'FCompiler' (line 7)
FCompiler_61784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'FCompiler')

class HPUXFCompiler(FCompiler_61784, ):

    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'HPUXFCompiler.get_flags')
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HPUXFCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HPUXFCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags(...)' code ##################

        # Getting the type of 'self' (line 26)
        self_61785 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 15), 'self')
        # Obtaining the member 'pic_flags' of a type (line 26)
        pic_flags_61786 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 15), self_61785, 'pic_flags')
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_61787 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        str_61788 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 33), 'str', '+ppu')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 32), list_61787, str_61788)
        # Adding element type (line 26)
        str_61789 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 41), 'str', '+DD64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 32), list_61787, str_61789)
        
        # Applying the binary operator '+' (line 26)
        result_add_61790 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 15), '+', pic_flags_61786, list_61787)
        
        # Assigning a type to the variable 'stypy_return_type' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'stypy_return_type', result_add_61790)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_61791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61791)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_61791


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'HPUXFCompiler.get_flags_opt')
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HPUXFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HPUXFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_opt', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_opt(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 28)
        list_61792 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 28)
        # Adding element type (line 28)
        str_61793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 16), 'str', '-O3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 28, 15), list_61792, str_61793)
        
        # Assigning a type to the variable 'stypy_return_type' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 8), 'stypy_return_type', list_61792)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_61794 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61794)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_61794


    @norecursion
    def get_libraries(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_libraries'
        module_type_store = module_type_store.open_function_context('get_libraries', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_localization', localization)
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_type_store', module_type_store)
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_function_name', 'HPUXFCompiler.get_libraries')
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_param_names_list', [])
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_varargs_param_name', None)
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_call_defaults', defaults)
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_call_varargs', varargs)
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HPUXFCompiler.get_libraries.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HPUXFCompiler.get_libraries', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 30)
        list_61795 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 30)
        # Adding element type (line 30)
        str_61796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 16), 'str', 'm')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 30, 15), list_61795, str_61796)
        
        # Assigning a type to the variable 'stypy_return_type' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'stypy_return_type', list_61795)
        
        # ################# End of 'get_libraries(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_libraries' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_61797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61797)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_libraries'
        return stypy_return_type_61797


    @norecursion
    def get_library_dirs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_library_dirs'
        module_type_store = module_type_store.open_function_context('get_library_dirs', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_localization', localization)
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_type_store', module_type_store)
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_function_name', 'HPUXFCompiler.get_library_dirs')
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_param_names_list', [])
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_varargs_param_name', None)
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_defaults', defaults)
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_varargs', varargs)
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HPUXFCompiler.get_library_dirs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HPUXFCompiler.get_library_dirs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_library_dirs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_library_dirs(...)' code ##################

        
        # Assigning a List to a Name (line 32):
        
        # Obtaining an instance of the builtin type 'list' (line 32)
        list_61798 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 14), 'list')
        # Adding type elements to the builtin type 'list' instance (line 32)
        # Adding element type (line 32)
        str_61799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 15), 'str', '/usr/lib/hpux64')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 14), list_61798, str_61799)
        
        # Assigning a type to the variable 'opt' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 8), 'opt', list_61798)
        # Getting the type of 'opt' (line 33)
        opt_61800 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'stypy_return_type', opt_61800)
        
        # ################# End of 'get_library_dirs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_library_dirs' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_61801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_library_dirs'
        return stypy_return_type_61801


    @norecursion
    def get_version(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_61802 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'int')
        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_61803 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 45), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        int_61804 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 46), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 45), list_61803, int_61804)
        # Adding element type (line 34)
        int_61805 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 51), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 45), list_61803, int_61805)
        # Adding element type (line 34)
        int_61806 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 54), 'int')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 45), list_61803, int_61806)
        
        defaults = [int_61802, list_61803]
        # Create a new context for function 'get_version'
        module_type_store = module_type_store.open_function_context('get_version', 34, 4, False)
        # Assigning a type to the variable 'self' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_localization', localization)
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_type_store', module_type_store)
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_function_name', 'HPUXFCompiler.get_version')
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_param_names_list', ['force', 'ok_status'])
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_varargs_param_name', None)
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_kwargs_param_name', None)
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_call_defaults', defaults)
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_call_varargs', varargs)
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        HPUXFCompiler.get_version.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HPUXFCompiler.get_version', ['force', 'ok_status'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_version', localization, ['force', 'ok_status'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_version(...)' code ##################

        
        # Call to get_version(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'self' (line 37)
        self_61809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 37), 'self', False)
        # Getting the type of 'force' (line 37)
        force_61810 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 43), 'force', False)
        # Getting the type of 'ok_status' (line 37)
        ok_status_61811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 50), 'ok_status', False)
        # Processing the call keyword arguments (line 37)
        kwargs_61812 = {}
        # Getting the type of 'FCompiler' (line 37)
        FCompiler_61807 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'FCompiler', False)
        # Obtaining the member 'get_version' of a type (line 37)
        get_version_61808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), FCompiler_61807, 'get_version')
        # Calling get_version(args, kwargs) (line 37)
        get_version_call_result_61813 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), get_version_61808, *[self_61809, force_61810, ok_status_61811], **kwargs_61812)
        
        # Assigning a type to the variable 'stypy_return_type' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'stypy_return_type', get_version_call_result_61813)
        
        # ################# End of 'get_version(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_version' in the type store
        # Getting the type of 'stypy_return_type' (line 34)
        stypy_return_type_61814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_61814)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_version'
        return stypy_return_type_61814


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 7, 0, False)
        # Assigning a type to the variable 'self' (line 8)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'HPUXFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'HPUXFCompiler' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'HPUXFCompiler', HPUXFCompiler)

# Assigning a Str to a Name (line 9):
str_61815 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'str', 'hpux')
# Getting the type of 'HPUXFCompiler'
HPUXFCompiler_61816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HPUXFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HPUXFCompiler_61816, 'compiler_type', str_61815)

# Assigning a Str to a Name (line 10):
str_61817 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'str', 'HP Fortran 90 Compiler')
# Getting the type of 'HPUXFCompiler'
HPUXFCompiler_61818 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HPUXFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HPUXFCompiler_61818, 'description', str_61817)

# Assigning a Str to a Name (line 11):
str_61819 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'str', 'HP F90 (?P<version>[^\\s*,]*)')
# Getting the type of 'HPUXFCompiler'
HPUXFCompiler_61820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HPUXFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HPUXFCompiler_61820, 'version_pattern', str_61819)

# Assigning a Dict to a Name (line 13):

# Obtaining an instance of the builtin type 'dict' (line 13)
dict_61821 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 13)
# Adding element type (key, value) (line 13)
str_61822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 14)
list_61823 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_61824 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 25), list_61823, str_61824)
# Adding element type (line 14)
str_61825 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 33), 'str', '+version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 25), list_61823, str_61825)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_61821, (str_61822, list_61823))
# Adding element type (key, value) (line 13)
str_61826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 15)
list_61827 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_61828 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_61827, str_61828)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_61821, (str_61826, list_61827))
# Adding element type (key, value) (line 13)
str_61829 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 16)
list_61830 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_61831 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_61830, str_61831)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_61821, (str_61829, list_61830))
# Adding element type (key, value) (line 13)
str_61832 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 17)
list_61833 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_61834 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_61833, str_61834)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_61821, (str_61832, list_61833))
# Adding element type (key, value) (line 13)
str_61835 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 18)
list_61836 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_61837 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'ld')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_61836, str_61837)
# Adding element type (line 18)
str_61838 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 32), 'str', '-b')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_61836, str_61838)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_61821, (str_61835, list_61836))
# Adding element type (key, value) (line 13)
str_61839 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 19)
list_61840 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_61841 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_61840, str_61841)
# Adding element type (line 19)
str_61842 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_61840, str_61842)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_61821, (str_61839, list_61840))
# Adding element type (key, value) (line 13)
str_61843 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 20)
list_61844 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_61845 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_61844, str_61845)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_61821, (str_61843, list_61844))

# Getting the type of 'HPUXFCompiler'
HPUXFCompiler_61846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HPUXFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HPUXFCompiler_61846, 'executables', dict_61821)

# Assigning a Name to a Name (line 22):
# Getting the type of 'None' (line 22)
None_61847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 24), 'None')
# Getting the type of 'HPUXFCompiler'
HPUXFCompiler_61848 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HPUXFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HPUXFCompiler_61848, 'module_dir_switch', None_61847)

# Assigning a Name to a Name (line 23):
# Getting the type of 'None' (line 23)
None_61849 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 28), 'None')
# Getting the type of 'HPUXFCompiler'
HPUXFCompiler_61850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HPUXFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HPUXFCompiler_61850, 'module_include_switch', None_61849)

# Assigning a List to a Name (line 24):

# Obtaining an instance of the builtin type 'list' (line 24)
list_61851 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
str_61852 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 17), 'str', '+Z')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 16), list_61851, str_61852)

# Getting the type of 'HPUXFCompiler'
HPUXFCompiler_61853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'HPUXFCompiler')
# Setting the type of the member 'pic_flags' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), HPUXFCompiler_61853, 'pic_flags', list_61851)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 40, 4))
    
    # 'from distutils import log' statement (line 40)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 40, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 41)
    # Processing the call arguments (line 41)
    int_61856 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 22), 'int')
    # Processing the call keyword arguments (line 41)
    kwargs_61857 = {}
    # Getting the type of 'log' (line 41)
    log_61854 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 41)
    set_verbosity_61855 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 4), log_61854, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 41)
    set_verbosity_call_result_61858 = invoke(stypy.reporting.localization.Localization(__file__, 41, 4), set_verbosity_61855, *[int_61856], **kwargs_61857)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 42, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 42)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_61859 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 42, 4), 'numpy.distutils.fcompiler')

    if (type(import_61859) is not StypyTypeError):

        if (import_61859 != 'pyd_module'):
            __import__(import_61859)
            sys_modules_61860 = sys.modules[import_61859]
            import_from_module(stypy.reporting.localization.Localization(__file__, 42, 4), 'numpy.distutils.fcompiler', sys_modules_61860.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 42, 4), __file__, sys_modules_61860, sys_modules_61860.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 42, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'numpy.distutils.fcompiler', import_61859)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 43):
    
    # Call to new_fcompiler(...): (line 43)
    # Processing the call keyword arguments (line 43)
    str_61862 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 38), 'str', 'hpux')
    keyword_61863 = str_61862
    kwargs_61864 = {'compiler': keyword_61863}
    # Getting the type of 'new_fcompiler' (line 43)
    new_fcompiler_61861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 43)
    new_fcompiler_call_result_61865 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), new_fcompiler_61861, *[], **kwargs_61864)
    
    # Assigning a type to the variable 'compiler' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'compiler', new_fcompiler_call_result_61865)
    
    # Call to customize(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_61868 = {}
    # Getting the type of 'compiler' (line 44)
    compiler_61866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 44)
    customize_61867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), compiler_61866, 'customize')
    # Calling customize(args, kwargs) (line 44)
    customize_call_result_61869 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), customize_61867, *[], **kwargs_61868)
    
    
    # Call to print(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to get_version(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_61873 = {}
    # Getting the type of 'compiler' (line 45)
    compiler_61871 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 45)
    get_version_61872 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 10), compiler_61871, 'get_version')
    # Calling get_version(args, kwargs) (line 45)
    get_version_call_result_61874 = invoke(stypy.reporting.localization.Localization(__file__, 45, 10), get_version_61872, *[], **kwargs_61873)
    
    # Processing the call keyword arguments (line 45)
    kwargs_61875 = {}
    # Getting the type of 'print' (line 45)
    print_61870 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'print', False)
    # Calling print(args, kwargs) (line 45)
    print_call_result_61876 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), print_61870, *[get_version_call_result_61874], **kwargs_61875)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
