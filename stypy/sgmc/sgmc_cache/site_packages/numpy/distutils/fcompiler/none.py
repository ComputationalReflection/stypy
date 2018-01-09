
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from numpy.distutils.fcompiler import FCompiler
4: 
5: compilers = ['NoneFCompiler']
6: 
7: class NoneFCompiler(FCompiler):
8: 
9:     compiler_type = 'none'
10:     description = 'Fake Fortran compiler'
11: 
12:     executables = {'compiler_f77': None,
13:                    'compiler_f90': None,
14:                    'compiler_fix': None,
15:                    'linker_so': None,
16:                    'linker_exe': None,
17:                    'archiver': None,
18:                    'ranlib': None,
19:                    'version_cmd': None,
20:                    }
21: 
22:     def find_executables(self):
23:         pass
24: 
25: 
26: if __name__ == '__main__':
27:     from distutils import log
28:     log.set_verbosity(2)
29:     compiler = NoneFCompiler()
30:     compiler.customize()
31:     print(compiler.get_version())
32: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_62979 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler')

if (type(import_62979) is not StypyTypeError):

    if (import_62979 != 'pyd_module'):
        __import__(import_62979)
        sys_modules_62980 = sys.modules[import_62979]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', sys_modules_62980.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_62980, sys_modules_62980.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'numpy.distutils.fcompiler', import_62979)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 5):

# Obtaining an instance of the builtin type 'list' (line 5)
list_62981 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 5)
# Adding element type (line 5)
str_62982 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 5, 13), 'str', 'NoneFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 5, 12), list_62981, str_62982)

# Assigning a type to the variable 'compilers' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'compilers', list_62981)
# Declaration of the 'NoneFCompiler' class
# Getting the type of 'FCompiler' (line 7)
FCompiler_62983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 7, 20), 'FCompiler')

class NoneFCompiler(FCompiler_62983, ):

    @norecursion
    def find_executables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_executables'
        module_type_store = module_type_store.open_function_context('find_executables', 22, 4, False)
        # Assigning a type to the variable 'self' (line 23)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 23, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_localization', localization)
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_function_name', 'NoneFCompiler.find_executables')
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_param_names_list', [])
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NoneFCompiler.find_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoneFCompiler.find_executables', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'find_executables', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'find_executables(...)' code ##################

        pass
        
        # ################# End of 'find_executables(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'find_executables' in the type store
        # Getting the type of 'stypy_return_type' (line 22)
        stypy_return_type_62984 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_62984)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_executables'
        return stypy_return_type_62984


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NoneFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NoneFCompiler' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'NoneFCompiler', NoneFCompiler)

# Assigning a Str to a Name (line 9):
str_62985 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'str', 'none')
# Getting the type of 'NoneFCompiler'
NoneFCompiler_62986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NoneFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NoneFCompiler_62986, 'compiler_type', str_62985)

# Assigning a Str to a Name (line 10):
str_62987 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'str', 'Fake Fortran compiler')
# Getting the type of 'NoneFCompiler'
NoneFCompiler_62988 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NoneFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NoneFCompiler_62988, 'description', str_62987)

# Assigning a Dict to a Name (line 12):

# Obtaining an instance of the builtin type 'dict' (line 12)
dict_62989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 12)
# Adding element type (key, value) (line 12)
str_62990 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 19), 'str', 'compiler_f77')
# Getting the type of 'None' (line 12)
None_62991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 35), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), dict_62989, (str_62990, None_62991))
# Adding element type (key, value) (line 12)
str_62992 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 19), 'str', 'compiler_f90')
# Getting the type of 'None' (line 13)
None_62993 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 35), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), dict_62989, (str_62992, None_62993))
# Adding element type (key, value) (line 12)
str_62994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 19), 'str', 'compiler_fix')
# Getting the type of 'None' (line 14)
None_62995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 35), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), dict_62989, (str_62994, None_62995))
# Adding element type (key, value) (line 12)
str_62996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 19), 'str', 'linker_so')
# Getting the type of 'None' (line 15)
None_62997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 32), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), dict_62989, (str_62996, None_62997))
# Adding element type (key, value) (line 12)
str_62998 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'str', 'linker_exe')
# Getting the type of 'None' (line 16)
None_62999 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 33), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), dict_62989, (str_62998, None_62999))
# Adding element type (key, value) (line 12)
str_63000 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 19), 'str', 'archiver')
# Getting the type of 'None' (line 17)
None_63001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 31), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), dict_62989, (str_63000, None_63001))
# Adding element type (key, value) (line 12)
str_63002 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 19), 'str', 'ranlib')
# Getting the type of 'None' (line 18)
None_63003 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 29), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), dict_62989, (str_63002, None_63003))
# Adding element type (key, value) (line 12)
str_63004 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 19), 'str', 'version_cmd')
# Getting the type of 'None' (line 19)
None_63005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 34), 'None')
set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 18), dict_62989, (str_63004, None_63005))

# Getting the type of 'NoneFCompiler'
NoneFCompiler_63006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'NoneFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), NoneFCompiler_63006, 'executables', dict_62989)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 27, 4))
    
    # 'from distutils import log' statement (line 27)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 27, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 28)
    # Processing the call arguments (line 28)
    int_63009 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, 22), 'int')
    # Processing the call keyword arguments (line 28)
    kwargs_63010 = {}
    # Getting the type of 'log' (line 28)
    log_63007 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 28)
    set_verbosity_63008 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 28, 4), log_63007, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 28)
    set_verbosity_call_result_63011 = invoke(stypy.reporting.localization.Localization(__file__, 28, 4), set_verbosity_63008, *[int_63009], **kwargs_63010)
    
    
    # Assigning a Call to a Name (line 29):
    
    # Call to NoneFCompiler(...): (line 29)
    # Processing the call keyword arguments (line 29)
    kwargs_63013 = {}
    # Getting the type of 'NoneFCompiler' (line 29)
    NoneFCompiler_63012 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'NoneFCompiler', False)
    # Calling NoneFCompiler(args, kwargs) (line 29)
    NoneFCompiler_call_result_63014 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), NoneFCompiler_63012, *[], **kwargs_63013)
    
    # Assigning a type to the variable 'compiler' (line 29)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'compiler', NoneFCompiler_call_result_63014)
    
    # Call to customize(...): (line 30)
    # Processing the call keyword arguments (line 30)
    kwargs_63017 = {}
    # Getting the type of 'compiler' (line 30)
    compiler_63015 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 30)
    customize_63016 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 4), compiler_63015, 'customize')
    # Calling customize(args, kwargs) (line 30)
    customize_call_result_63018 = invoke(stypy.reporting.localization.Localization(__file__, 30, 4), customize_63016, *[], **kwargs_63017)
    
    
    # Call to print(...): (line 31)
    # Processing the call arguments (line 31)
    
    # Call to get_version(...): (line 31)
    # Processing the call keyword arguments (line 31)
    kwargs_63022 = {}
    # Getting the type of 'compiler' (line 31)
    compiler_63020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 31)
    get_version_63021 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 10), compiler_63020, 'get_version')
    # Calling get_version(args, kwargs) (line 31)
    get_version_call_result_63023 = invoke(stypy.reporting.localization.Localization(__file__, 31, 10), get_version_63021, *[], **kwargs_63022)
    
    # Processing the call keyword arguments (line 31)
    kwargs_63024 = {}
    # Getting the type of 'print' (line 31)
    print_63019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'print', False)
    # Calling print(args, kwargs) (line 31)
    print_call_result_63025 = invoke(stypy.reporting.localization.Localization(__file__, 31, 4), print_63019, *[get_version_call_result_63023], **kwargs_63024)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
