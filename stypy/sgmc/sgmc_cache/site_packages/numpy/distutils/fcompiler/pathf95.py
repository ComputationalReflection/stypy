
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from numpy.distutils.fcompiler import FCompiler
4: 
5: compilers = ['PathScaleFCompiler']
6: 
7: class PathScaleFCompiler(FCompiler):
8: 
9:     compiler_type = 'pathf95'
10:     description = 'PathScale Fortran Compiler'
11:     version_pattern =  r'PathScale\(TM\) Compiler Suite: Version (?P<version>[\d.]+)'
12: 
13:     executables = {
14:         'version_cmd'  : ["pathf95", "-version"],
15:         'compiler_f77' : ["pathf95", "-fixedform"],
16:         'compiler_fix' : ["pathf95", "-fixedform"],
17:         'compiler_f90' : ["pathf95"],
18:         'linker_so'    : ["pathf95", "-shared"],
19:         'archiver'     : ["ar", "-cr"],
20:         'ranlib'       : ["ranlib"]
21:     }
22:     pic_flags = ['-fPIC']
23:     module_dir_switch = '-module ' # Don't remove ending space!
24:     module_include_switch = '-I'
25: 
26:     def get_flags_opt(self):
27:         return ['-O3']
28:     def get_flags_debug(self):
29:         return ['-g']
30: 
31: if __name__ == '__main__':
32:     from distutils import log
33:     log.set_verbosity(2)
34:     #compiler = PathScaleFCompiler()
35:     from numpy.distutils.fcompiler import new_fcompiler
36:     compiler = new_fcompiler(compiler='pathf95')
37:     compiler.customize()
38:     print(compiler.get_version())
39: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63026 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler')

if (type(import_63026) is not StypyTypeError):

    if (import_63026 != 'pyd_module'):
        __import__(import_63026)
        sys_modules_63027 = sys.modules[import_63026]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', sys_modules_63027.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_63027, sys_modules_63027.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', import_63026)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 5):

# Obtaining an instance of the builtin type 'list' (line 5)
list_63028 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
str_63029 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'str', 'PathScaleFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_63028, str_63029)

# Assigning a type to the variable 'compilers' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'compilers', list_63028)
# Declaration of the 'PathScaleFCompiler' class
# Getting the type of 'FCompiler' (line 7)
FCompiler_63030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 25), 'FCompiler')

class PathScaleFCompiler(FCompiler_63030, ):

    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 26, 4, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'PathScaleFCompiler.get_flags_opt')
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathScaleFCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathScaleFCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_63031 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        # Adding element type (line 27)
        str_63032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 16), 'str', '-O3')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 15), list_63031, str_63032)
        
        # Assigning a type to the variable 'stypy_return_type' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'stypy_return_type', list_63031)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 26)
        stypy_return_type_63033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63033)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_63033


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'PathScaleFCompiler.get_flags_debug')
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        PathScaleFCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathScaleFCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_debug', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_debug(...)' code ##################

        
        # Obtaining an instance of the builtin type 'list' (line 29)
        list_63034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 29)
        # Adding element type (line 29)
        str_63035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 15), list_63034, str_63035)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', list_63034)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_63036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63036)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_63036


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathScaleFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PathScaleFCompiler' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'PathScaleFCompiler', PathScaleFCompiler)

# Assigning a Str to a Name (line 9):
str_63037 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'str', 'pathf95')
# Getting the type of 'PathScaleFCompiler'
PathScaleFCompiler_63038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleFCompiler_63038, 'compiler_type', str_63037)

# Assigning a Str to a Name (line 10):
str_63039 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'str', 'PathScale Fortran Compiler')
# Getting the type of 'PathScaleFCompiler'
PathScaleFCompiler_63040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleFCompiler_63040, 'description', str_63039)

# Assigning a Str to a Name (line 11):
str_63041 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'str', 'PathScale\\(TM\\) Compiler Suite: Version (?P<version>[\\d.]+)')
# Getting the type of 'PathScaleFCompiler'
PathScaleFCompiler_63042 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleFCompiler_63042, 'version_pattern', str_63041)

# Assigning a Dict to a Name (line 13):

# Obtaining an instance of the builtin type 'dict' (line 13)
dict_63043 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 13)
# Adding element type (key, value) (line 13)
str_63044 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 14)
list_63045 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 14)
# Adding element type (line 14)
str_63046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 26), 'str', 'pathf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 25), list_63045, str_63046)
# Adding element type (line 14)
str_63047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 37), 'str', '-version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 14, 25), list_63045, str_63047)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_63043, (str_63044, list_63045))
# Adding element type (key, value) (line 13)
str_63048 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 15)
list_63049 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 15)
# Adding element type (line 15)
str_63050 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 26), 'str', 'pathf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_63049, str_63050)
# Adding element type (line 15)
str_63051 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 37), 'str', '-fixedform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 15, 25), list_63049, str_63051)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_63043, (str_63048, list_63049))
# Adding element type (key, value) (line 13)
str_63052 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 16)
list_63053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)
str_63054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 26), 'str', 'pathf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_63053, str_63054)
# Adding element type (line 16)
str_63055 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 37), 'str', '-fixedform')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 25), list_63053, str_63055)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_63043, (str_63052, list_63053))
# Adding element type (key, value) (line 13)
str_63056 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 17)
list_63057 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 17)
# Adding element type (line 17)
str_63058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 26), 'str', 'pathf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 25), list_63057, str_63058)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_63043, (str_63056, list_63057))
# Adding element type (key, value) (line 13)
str_63059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 18)
list_63060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 18)
# Adding element type (line 18)
str_63061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 26), 'str', 'pathf95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_63060, str_63061)
# Adding element type (line 18)
str_63062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 37), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 25), list_63060, str_63062)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_63043, (str_63059, list_63060))
# Adding element type (key, value) (line 13)
str_63063 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 19)
list_63064 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 19)
# Adding element type (line 19)
str_63065 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_63064, str_63065)
# Adding element type (line 19)
str_63066 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 25), list_63064, str_63066)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_63043, (str_63063, list_63064))
# Adding element type (key, value) (line 13)
str_63067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 20)
list_63068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 20)
# Adding element type (line 20)
str_63069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 25), list_63068, str_63069)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 18), dict_63043, (str_63067, list_63068))

# Getting the type of 'PathScaleFCompiler'
PathScaleFCompiler_63070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleFCompiler_63070, 'executables', dict_63043)

# Assigning a List to a Name (line 22):

# Obtaining an instance of the builtin type 'list' (line 22)
list_63071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_63072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 17), 'str', '-fPIC')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 16), list_63071, str_63072)

# Getting the type of 'PathScaleFCompiler'
PathScaleFCompiler_63073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleFCompiler')
# Setting the type of the member 'pic_flags' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleFCompiler_63073, 'pic_flags', list_63071)

# Assigning a Str to a Name (line 23):
str_63074 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 24), 'str', '-module ')
# Getting the type of 'PathScaleFCompiler'
PathScaleFCompiler_63075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleFCompiler_63075, 'module_dir_switch', str_63074)

# Assigning a Str to a Name (line 24):
str_63076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 28), 'str', '-I')
# Getting the type of 'PathScaleFCompiler'
PathScaleFCompiler_63077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleFCompiler_63077, 'module_include_switch', str_63076)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 32, 4))
    
    # 'from distutils import log' statement (line 32)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 32, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 33)
    # Processing the call arguments (line 33)
    int_63080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 22), 'int')
    # Processing the call keyword arguments (line 33)
    kwargs_63081 = {}
    # Getting the type of 'log' (line 33)
    log_63078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 33)
    set_verbosity_63079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 4), log_63078, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 33)
    set_verbosity_call_result_63082 = invoke(stypy.reporting.localization.Localization(__file__, 33, 4), set_verbosity_63079, *[int_63080], **kwargs_63081)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 35, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 35)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_63083 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 35, 4), 'numpy.distutils.fcompiler')

    if (type(import_63083) is not StypyTypeError):

        if (import_63083 != 'pyd_module'):
            __import__(import_63083)
            sys_modules_63084 = sys.modules[import_63083]
            import_from_module(stypy.reporting.localization.Localization(__file__, 35, 4), 'numpy.distutils.fcompiler', sys_modules_63084.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 35, 4), __file__, sys_modules_63084, sys_modules_63084.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 35, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'numpy.distutils.fcompiler', import_63083)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 36):
    
    # Call to new_fcompiler(...): (line 36)
    # Processing the call keyword arguments (line 36)
    str_63086 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 38), 'str', 'pathf95')
    keyword_63087 = str_63086
    kwargs_63088 = {'compiler': keyword_63087}
    # Getting the type of 'new_fcompiler' (line 36)
    new_fcompiler_63085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 36)
    new_fcompiler_call_result_63089 = invoke(stypy.reporting.localization.Localization(__file__, 36, 15), new_fcompiler_63085, *[], **kwargs_63088)
    
    # Assigning a type to the variable 'compiler' (line 36)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'compiler', new_fcompiler_call_result_63089)
    
    # Call to customize(...): (line 37)
    # Processing the call keyword arguments (line 37)
    kwargs_63092 = {}
    # Getting the type of 'compiler' (line 37)
    compiler_63090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 37)
    customize_63091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 4), compiler_63090, 'customize')
    # Calling customize(args, kwargs) (line 37)
    customize_call_result_63093 = invoke(stypy.reporting.localization.Localization(__file__, 37, 4), customize_63091, *[], **kwargs_63092)
    
    
    # Call to print(...): (line 38)
    # Processing the call arguments (line 38)
    
    # Call to get_version(...): (line 38)
    # Processing the call keyword arguments (line 38)
    kwargs_63097 = {}
    # Getting the type of 'compiler' (line 38)
    compiler_63095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 38)
    get_version_63096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 10), compiler_63095, 'get_version')
    # Calling get_version(args, kwargs) (line 38)
    get_version_call_result_63098 = invoke(stypy.reporting.localization.Localization(__file__, 38, 10), get_version_63096, *[], **kwargs_63097)
    
    # Processing the call keyword arguments (line 38)
    kwargs_63099 = {}
    # Getting the type of 'print' (line 38)
    print_63094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'print', False)
    # Calling print(args, kwargs) (line 38)
    print_call_result_63100 = invoke(stypy.reporting.localization.Localization(__file__, 38, 4), print_63094, *[get_version_call_result_63098], **kwargs_63099)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
