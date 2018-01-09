
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: # http://g95.sourceforge.net/
2: from __future__ import division, absolute_import, print_function
3: 
4: from numpy.distutils.fcompiler import FCompiler
5: 
6: compilers = ['G95FCompiler']
7: 
8: class G95FCompiler(FCompiler):
9:     compiler_type = 'g95'
10:     description = 'G95 Fortran Compiler'
11: 
12: #    version_pattern = r'G95 \((GCC (?P<gccversion>[\d.]+)|.*?) \(g95!\) (?P<version>.*)\).*'
13:     # $ g95 --version
14:     # G95 (GCC 4.0.3 (g95!) May 22 2006)
15: 
16:     version_pattern = r'G95 \((GCC (?P<gccversion>[\d.]+)|.*?) \(g95 (?P<version>.*)!\) (?P<date>.*)\).*'
17:     # $ g95 --version
18:     # G95 (GCC 4.0.3 (g95 0.90!) Aug 22 2006)
19: 
20:     executables = {
21:         'version_cmd'  : ["<F90>", "--version"],
22:         'compiler_f77' : ["g95", "-ffixed-form"],
23:         'compiler_fix' : ["g95", "-ffixed-form"],
24:         'compiler_f90' : ["g95"],
25:         'linker_so'    : ["<F90>", "-shared"],
26:         'archiver'     : ["ar", "-cr"],
27:         'ranlib'       : ["ranlib"]
28:         }
29:     pic_flags = ['-fpic']
30:     module_dir_switch = '-fmod='
31:     module_include_switch = '-I'
32: 
33:     def get_flags(self):
34:         return ['-fno-second-underscore']
35:     def get_flags_opt(self):
36:         return ['-O']
37:     def get_flags_debug(self):
38:         return ['-g']
39: 
40: if __name__ == '__main__':
41:     from distutils import log
42:     log.set_verbosity(2)
43:     compiler = G95FCompiler()
44:     compiler.customize()
45:     print(compiler.get_version())
46: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from numpy.distutils.fcompiler import FCompiler' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_60545 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler')

if (type(import_60545) is not StypyTypeError):

    if (import_60545 != 'pyd_module'):
        __import__(import_60545)
        sys_modules_60546 = sys.modules[import_60545]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', sys_modules_60546.module_type_store, module_type_store, ['FCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_60546, sys_modules_60546.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler import FCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', None, module_type_store, ['FCompiler'], [FCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'numpy.distutils.fcompiler', import_60545)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 6):

# Obtaining an instance of the builtin type 'list' (line 6)
list_60547 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 6)
# Adding element type (line 6)
str_60548 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 13), 'str', 'G95FCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 6, 12), list_60547, str_60548)

# Assigning a type to the variable 'compilers' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'compilers', list_60547)
# Declaration of the 'G95FCompiler' class
# Getting the type of 'FCompiler' (line 8)
FCompiler_60549 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'FCompiler')

class G95FCompiler(FCompiler_60549, ):

    @norecursion
    def get_flags(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags'
        module_type_store = module_type_store.open_function_context('get_flags', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_localization', localization)
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_type_store', module_type_store)
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_function_name', 'G95FCompiler.get_flags')
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_param_names_list', [])
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_varargs_param_name', None)
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_kwargs_param_name', None)
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_call_defaults', defaults)
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_call_varargs', varargs)
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        G95FCompiler.get_flags.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'G95FCompiler.get_flags', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 34)
        list_60550 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 34)
        # Adding element type (line 34)
        str_60551 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 16), 'str', '-fno-second-underscore')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 34, 15), list_60550, str_60551)
        
        # Assigning a type to the variable 'stypy_return_type' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'stypy_return_type', list_60550)
        
        # ################# End of 'get_flags(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_60552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60552)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags'
        return stypy_return_type_60552


    @norecursion
    def get_flags_opt(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_opt'
        module_type_store = module_type_store.open_function_context('get_flags_opt', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_localization', localization)
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_type_store', module_type_store)
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_function_name', 'G95FCompiler.get_flags_opt')
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_param_names_list', [])
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_varargs_param_name', None)
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_kwargs_param_name', None)
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_defaults', defaults)
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_varargs', varargs)
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        G95FCompiler.get_flags_opt.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'G95FCompiler.get_flags_opt', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 36)
        list_60553 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 36)
        # Adding element type (line 36)
        str_60554 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 16), 'str', '-O')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 36, 15), list_60553, str_60554)
        
        # Assigning a type to the variable 'stypy_return_type' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'stypy_return_type', list_60553)
        
        # ################# End of 'get_flags_opt(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_opt' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_60555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60555)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_opt'
        return stypy_return_type_60555


    @norecursion
    def get_flags_debug(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_debug'
        module_type_store = module_type_store.open_function_context('get_flags_debug', 37, 4, False)
        # Assigning a type to the variable 'self' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_localization', localization)
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_type_store', module_type_store)
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_function_name', 'G95FCompiler.get_flags_debug')
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_param_names_list', [])
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_varargs_param_name', None)
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_kwargs_param_name', None)
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_defaults', defaults)
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_varargs', varargs)
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        G95FCompiler.get_flags_debug.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'G95FCompiler.get_flags_debug', [], None, None, defaults, varargs, kwargs)

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

        
        # Obtaining an instance of the builtin type 'list' (line 38)
        list_60556 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 15), 'list')
        # Adding type elements to the builtin type 'list' instance (line 38)
        # Adding element type (line 38)
        str_60557 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 16), 'str', '-g')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 38, 15), list_60556, str_60557)
        
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', list_60556)
        
        # ################# End of 'get_flags_debug(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_debug' in the type store
        # Getting the type of 'stypy_return_type' (line 37)
        stypy_return_type_60558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_60558)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_debug'
        return stypy_return_type_60558


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'G95FCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'G95FCompiler' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'G95FCompiler', G95FCompiler)

# Assigning a Str to a Name (line 9):
str_60559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 20), 'str', 'g95')
# Getting the type of 'G95FCompiler'
G95FCompiler_60560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'G95FCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), G95FCompiler_60560, 'compiler_type', str_60559)

# Assigning a Str to a Name (line 10):
str_60561 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 18), 'str', 'G95 Fortran Compiler')
# Getting the type of 'G95FCompiler'
G95FCompiler_60562 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'G95FCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), G95FCompiler_60562, 'description', str_60561)

# Assigning a Str to a Name (line 16):
str_60563 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 22), 'str', 'G95 \\((GCC (?P<gccversion>[\\d.]+)|.*?) \\(g95 (?P<version>.*)!\\) (?P<date>.*)\\).*')
# Getting the type of 'G95FCompiler'
G95FCompiler_60564 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'G95FCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), G95FCompiler_60564, 'version_pattern', str_60563)

# Assigning a Dict to a Name (line 20):

# Obtaining an instance of the builtin type 'dict' (line 20)
dict_60565 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 20)
# Adding element type (key, value) (line 20)
str_60566 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 21)
list_60567 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_60568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_60567, str_60568)
# Adding element type (line 21)
str_60569 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 35), 'str', '--version')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_60567, str_60569)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_60565, (str_60566, list_60567))
# Adding element type (key, value) (line 20)
str_60570 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 22)
list_60571 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_60572 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'str', 'g95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_60571, str_60572)
# Adding element type (line 22)
str_60573 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 33), 'str', '-ffixed-form')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_60571, str_60573)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_60565, (str_60570, list_60571))
# Adding element type (key, value) (line 20)
str_60574 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 23)
list_60575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
str_60576 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'str', 'g95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_60575, str_60576)
# Adding element type (line 23)
str_60577 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 33), 'str', '-ffixed-form')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_60575, str_60577)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_60565, (str_60574, list_60575))
# Adding element type (key, value) (line 20)
str_60578 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 24)
list_60579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
str_60580 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'str', 'g95')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 25), list_60579, str_60580)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_60565, (str_60578, list_60579))
# Adding element type (key, value) (line 20)
str_60581 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 25)
list_60582 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
str_60583 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 25), list_60582, str_60583)
# Adding element type (line 25)
str_60584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 35), 'str', '-shared')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 25), list_60582, str_60584)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_60565, (str_60581, list_60582))
# Adding element type (key, value) (line 20)
str_60585 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 26)
list_60586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_60587 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), list_60586, str_60587)
# Adding element type (line 26)
str_60588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), list_60586, str_60588)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_60565, (str_60585, list_60586))
# Adding element type (key, value) (line 20)
str_60589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 27)
list_60590 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_60591 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 25), list_60590, str_60591)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_60565, (str_60589, list_60590))

# Getting the type of 'G95FCompiler'
G95FCompiler_60592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'G95FCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), G95FCompiler_60592, 'executables', dict_60565)

# Assigning a List to a Name (line 29):

# Obtaining an instance of the builtin type 'list' (line 29)
list_60593 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 16), 'list')
# Adding type elements to the builtin type 'list' instance (line 29)
# Adding element type (line 29)
str_60594 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 17), 'str', '-fpic')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 29, 16), list_60593, str_60594)

# Getting the type of 'G95FCompiler'
G95FCompiler_60595 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'G95FCompiler')
# Setting the type of the member 'pic_flags' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), G95FCompiler_60595, 'pic_flags', list_60593)

# Assigning a Str to a Name (line 30):
str_60596 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 24), 'str', '-fmod=')
# Getting the type of 'G95FCompiler'
G95FCompiler_60597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'G95FCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), G95FCompiler_60597, 'module_dir_switch', str_60596)

# Assigning a Str to a Name (line 31):
str_60598 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 28), 'str', '-I')
# Getting the type of 'G95FCompiler'
G95FCompiler_60599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'G95FCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), G95FCompiler_60599, 'module_include_switch', str_60598)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 41, 4))
    
    # 'from distutils import log' statement (line 41)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 41, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 42)
    # Processing the call arguments (line 42)
    int_60602 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 22), 'int')
    # Processing the call keyword arguments (line 42)
    kwargs_60603 = {}
    # Getting the type of 'log' (line 42)
    log_60600 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 42)
    set_verbosity_60601 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 4), log_60600, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 42)
    set_verbosity_call_result_60604 = invoke(stypy.reporting.localization.Localization(__file__, 42, 4), set_verbosity_60601, *[int_60602], **kwargs_60603)
    
    
    # Assigning a Call to a Name (line 43):
    
    # Call to G95FCompiler(...): (line 43)
    # Processing the call keyword arguments (line 43)
    kwargs_60606 = {}
    # Getting the type of 'G95FCompiler' (line 43)
    G95FCompiler_60605 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'G95FCompiler', False)
    # Calling G95FCompiler(args, kwargs) (line 43)
    G95FCompiler_call_result_60607 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), G95FCompiler_60605, *[], **kwargs_60606)
    
    # Assigning a type to the variable 'compiler' (line 43)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'compiler', G95FCompiler_call_result_60607)
    
    # Call to customize(...): (line 44)
    # Processing the call keyword arguments (line 44)
    kwargs_60610 = {}
    # Getting the type of 'compiler' (line 44)
    compiler_60608 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 44)
    customize_60609 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 4), compiler_60608, 'customize')
    # Calling customize(args, kwargs) (line 44)
    customize_call_result_60611 = invoke(stypy.reporting.localization.Localization(__file__, 44, 4), customize_60609, *[], **kwargs_60610)
    
    
    # Call to print(...): (line 45)
    # Processing the call arguments (line 45)
    
    # Call to get_version(...): (line 45)
    # Processing the call keyword arguments (line 45)
    kwargs_60615 = {}
    # Getting the type of 'compiler' (line 45)
    compiler_60613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 45)
    get_version_60614 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 10), compiler_60613, 'get_version')
    # Calling get_version(args, kwargs) (line 45)
    get_version_call_result_60616 = invoke(stypy.reporting.localization.Localization(__file__, 45, 10), get_version_60614, *[], **kwargs_60615)
    
    # Processing the call keyword arguments (line 45)
    kwargs_60617 = {}
    # Getting the type of 'print' (line 45)
    print_60612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'print', False)
    # Calling print(args, kwargs) (line 45)
    print_call_result_60618 = invoke(stypy.reporting.localization.Localization(__file__, 45, 4), print_60612, *[get_version_call_result_60616], **kwargs_60617)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
