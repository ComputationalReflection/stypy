
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: 
5: from numpy.distutils.fcompiler.gnu import GnuFCompiler
6: 
7: compilers = ['VastFCompiler']
8: 
9: class VastFCompiler(GnuFCompiler):
10:     compiler_type = 'vast'
11:     compiler_aliases = ()
12:     description = 'Pacific-Sierra Research Fortran 90 Compiler'
13:     version_pattern = r'\s*Pacific-Sierra Research vf90 '\
14:                       '(Personal|Professional)\s+(?P<version>[^\s]*)'
15: 
16:     # VAST f90 does not support -o with -c. So, object files are created
17:     # to the current directory and then moved to build directory
18:     object_switch = ' && function _mvfile { mv -v `basename $1` $1 ; } && _mvfile '
19: 
20:     executables = {
21:         'version_cmd'  : ["vf90", "-v"],
22:         'compiler_f77' : ["g77"],
23:         'compiler_fix' : ["f90", "-Wv,-ya"],
24:         'compiler_f90' : ["f90"],
25:         'linker_so'    : ["<F90>"],
26:         'archiver'     : ["ar", "-cr"],
27:         'ranlib'       : ["ranlib"]
28:         }
29:     module_dir_switch = None  #XXX Fix me
30:     module_include_switch = None #XXX Fix me
31: 
32:     def find_executables(self):
33:         pass
34: 
35:     def get_version_cmd(self):
36:         f90 = self.compiler_f90[0]
37:         d, b = os.path.split(f90)
38:         vf90 = os.path.join(d, 'v'+b)
39:         return vf90
40: 
41:     def get_flags_arch(self):
42:         vast_version = self.get_version()
43:         gnu = GnuFCompiler()
44:         gnu.customize(None)
45:         self.version = gnu.get_version()
46:         opt = GnuFCompiler.get_flags_arch(self)
47:         self.version = vast_version
48:         return opt
49: 
50: if __name__ == '__main__':
51:     from distutils import log
52:     log.set_verbosity(2)
53:     from numpy.distutils.fcompiler import new_fcompiler
54:     compiler = new_fcompiler(compiler='vast')
55:     compiler.customize()
56:     print(compiler.get_version())
57: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import os' statement (line 3)
import os

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from numpy.distutils.fcompiler.gnu import GnuFCompiler' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
import_63352 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.fcompiler.gnu')

if (type(import_63352) is not StypyTypeError):

    if (import_63352 != 'pyd_module'):
        __import__(import_63352)
        sys_modules_63353 = sys.modules[import_63352]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.fcompiler.gnu', sys_modules_63353.module_type_store, module_type_store, ['GnuFCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_63353, sys_modules_63353.module_type_store, module_type_store)
    else:
        from numpy.distutils.fcompiler.gnu import GnuFCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.fcompiler.gnu', None, module_type_store, ['GnuFCompiler'], [GnuFCompiler])

else:
    # Assigning a type to the variable 'numpy.distutils.fcompiler.gnu' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'numpy.distutils.fcompiler.gnu', import_63352)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')


# Assigning a List to a Name (line 7):

# Assigning a List to a Name (line 7):

# Obtaining an instance of the builtin type 'list' (line 7)
list_63354 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 12), 'list')
# Adding type elements to the builtin type 'list' instance (line 7)
# Adding element type (line 7)
str_63355 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 7, 13), 'str', 'VastFCompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 7, 12), list_63354, str_63355)

# Assigning a type to the variable 'compilers' (line 7)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'compilers', list_63354)
# Declaration of the 'VastFCompiler' class
# Getting the type of 'GnuFCompiler' (line 9)
GnuFCompiler_63356 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 20), 'GnuFCompiler')

class VastFCompiler(GnuFCompiler_63356, ):
    
    # Assigning a Str to a Name (line 10):
    
    # Assigning a Tuple to a Name (line 11):
    
    # Assigning a Str to a Name (line 12):
    
    # Assigning a Str to a Name (line 13):
    
    # Assigning a Str to a Name (line 18):
    
    # Assigning a Dict to a Name (line 20):
    
    # Assigning a Name to a Name (line 29):
    
    # Assigning a Name to a Name (line 30):

    @norecursion
    def find_executables(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'find_executables'
        module_type_store = module_type_store.open_function_context('find_executables', 32, 4, False)
        # Assigning a type to the variable 'self' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_localization', localization)
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_type_store', module_type_store)
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_function_name', 'VastFCompiler.find_executables')
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_param_names_list', [])
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_varargs_param_name', None)
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_call_defaults', defaults)
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_call_varargs', varargs)
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VastFCompiler.find_executables.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VastFCompiler.find_executables', [], None, None, defaults, varargs, kwargs)

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
        # Getting the type of 'stypy_return_type' (line 32)
        stypy_return_type_63357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63357)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'find_executables'
        return stypy_return_type_63357


    @norecursion
    def get_version_cmd(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_version_cmd'
        module_type_store = module_type_store.open_function_context('get_version_cmd', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_localization', localization)
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_type_store', module_type_store)
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_function_name', 'VastFCompiler.get_version_cmd')
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_param_names_list', [])
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_varargs_param_name', None)
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_call_defaults', defaults)
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_call_varargs', varargs)
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VastFCompiler.get_version_cmd.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VastFCompiler.get_version_cmd', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_version_cmd', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_version_cmd(...)' code ##################

        
        # Assigning a Subscript to a Name (line 36):
        
        # Assigning a Subscript to a Name (line 36):
        
        # Obtaining the type of the subscript
        int_63358 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, 32), 'int')
        # Getting the type of 'self' (line 36)
        self_63359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 14), 'self')
        # Obtaining the member 'compiler_f90' of a type (line 36)
        compiler_f90_63360 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 14), self_63359, 'compiler_f90')
        # Obtaining the member '__getitem__' of a type (line 36)
        getitem___63361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 14), compiler_f90_63360, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 36)
        subscript_call_result_63362 = invoke(stypy.reporting.localization.Localization(__file__, 36, 14), getitem___63361, int_63358)
        
        # Assigning a type to the variable 'f90' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'f90', subscript_call_result_63362)
        
        # Assigning a Call to a Tuple (line 37):
        
        # Assigning a Call to a Name:
        
        # Call to split(...): (line 37)
        # Processing the call arguments (line 37)
        # Getting the type of 'f90' (line 37)
        f90_63366 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 29), 'f90', False)
        # Processing the call keyword arguments (line 37)
        kwargs_63367 = {}
        # Getting the type of 'os' (line 37)
        os_63363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 37)
        path_63364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), os_63363, 'path')
        # Obtaining the member 'split' of a type (line 37)
        split_63365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 15), path_63364, 'split')
        # Calling split(args, kwargs) (line 37)
        split_call_result_63368 = invoke(stypy.reporting.localization.Localization(__file__, 37, 15), split_63365, *[f90_63366], **kwargs_63367)
        
        # Assigning a type to the variable 'call_assignment_63349' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'call_assignment_63349', split_call_result_63368)
        
        # Assigning a Call to a Name (line 37):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_63371 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'int')
        # Processing the call keyword arguments
        kwargs_63372 = {}
        # Getting the type of 'call_assignment_63349' (line 37)
        call_assignment_63349_63369 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'call_assignment_63349', False)
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___63370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), call_assignment_63349_63369, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_63373 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___63370, *[int_63371], **kwargs_63372)
        
        # Assigning a type to the variable 'call_assignment_63350' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'call_assignment_63350', getitem___call_result_63373)
        
        # Assigning a Name to a Name (line 37):
        # Getting the type of 'call_assignment_63350' (line 37)
        call_assignment_63350_63374 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'call_assignment_63350')
        # Assigning a type to the variable 'd' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'd', call_assignment_63350_63374)
        
        # Assigning a Call to a Name (line 37):
        
        # Call to __getitem__(...):
        # Processing the call arguments
        int_63377 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 8), 'int')
        # Processing the call keyword arguments
        kwargs_63378 = {}
        # Getting the type of 'call_assignment_63349' (line 37)
        call_assignment_63349_63375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'call_assignment_63349', False)
        # Obtaining the member '__getitem__' of a type (line 37)
        getitem___63376 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 8), call_assignment_63349_63375, '__getitem__')
        # Calling __getitem__(args, kwargs)
        getitem___call_result_63379 = invoke(stypy.reporting.localization.Localization(__file__, 0, 0), getitem___63376, *[int_63377], **kwargs_63378)
        
        # Assigning a type to the variable 'call_assignment_63351' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'call_assignment_63351', getitem___call_result_63379)
        
        # Assigning a Name to a Name (line 37):
        # Getting the type of 'call_assignment_63351' (line 37)
        call_assignment_63351_63380 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'call_assignment_63351')
        # Assigning a type to the variable 'b' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 11), 'b', call_assignment_63351_63380)
        
        # Assigning a Call to a Name (line 38):
        
        # Assigning a Call to a Name (line 38):
        
        # Call to join(...): (line 38)
        # Processing the call arguments (line 38)
        # Getting the type of 'd' (line 38)
        d_63384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 28), 'd', False)
        str_63385 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 31), 'str', 'v')
        # Getting the type of 'b' (line 38)
        b_63386 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 35), 'b', False)
        # Applying the binary operator '+' (line 38)
        result_add_63387 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 31), '+', str_63385, b_63386)
        
        # Processing the call keyword arguments (line 38)
        kwargs_63388 = {}
        # Getting the type of 'os' (line 38)
        os_63381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'os', False)
        # Obtaining the member 'path' of a type (line 38)
        path_63382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), os_63381, 'path')
        # Obtaining the member 'join' of a type (line 38)
        join_63383 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 15), path_63382, 'join')
        # Calling join(args, kwargs) (line 38)
        join_call_result_63389 = invoke(stypy.reporting.localization.Localization(__file__, 38, 15), join_63383, *[d_63384, result_add_63387], **kwargs_63388)
        
        # Assigning a type to the variable 'vf90' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'vf90', join_call_result_63389)
        # Getting the type of 'vf90' (line 39)
        vf90_63390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 15), 'vf90')
        # Assigning a type to the variable 'stypy_return_type' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'stypy_return_type', vf90_63390)
        
        # ################# End of 'get_version_cmd(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_version_cmd' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_63391 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63391)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_version_cmd'
        return stypy_return_type_63391


    @norecursion
    def get_flags_arch(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_flags_arch'
        module_type_store = module_type_store.open_function_context('get_flags_arch', 41, 4, False)
        # Assigning a type to the variable 'self' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_localization', localization)
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_type_store', module_type_store)
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_function_name', 'VastFCompiler.get_flags_arch')
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_param_names_list', [])
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_varargs_param_name', None)
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_kwargs_param_name', None)
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_defaults', defaults)
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_varargs', varargs)
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        VastFCompiler.get_flags_arch.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VastFCompiler.get_flags_arch', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_flags_arch', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_flags_arch(...)' code ##################

        
        # Assigning a Call to a Name (line 42):
        
        # Assigning a Call to a Name (line 42):
        
        # Call to get_version(...): (line 42)
        # Processing the call keyword arguments (line 42)
        kwargs_63394 = {}
        # Getting the type of 'self' (line 42)
        self_63392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'self', False)
        # Obtaining the member 'get_version' of a type (line 42)
        get_version_63393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), self_63392, 'get_version')
        # Calling get_version(args, kwargs) (line 42)
        get_version_call_result_63395 = invoke(stypy.reporting.localization.Localization(__file__, 42, 23), get_version_63393, *[], **kwargs_63394)
        
        # Assigning a type to the variable 'vast_version' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'vast_version', get_version_call_result_63395)
        
        # Assigning a Call to a Name (line 43):
        
        # Assigning a Call to a Name (line 43):
        
        # Call to GnuFCompiler(...): (line 43)
        # Processing the call keyword arguments (line 43)
        kwargs_63397 = {}
        # Getting the type of 'GnuFCompiler' (line 43)
        GnuFCompiler_63396 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 14), 'GnuFCompiler', False)
        # Calling GnuFCompiler(args, kwargs) (line 43)
        GnuFCompiler_call_result_63398 = invoke(stypy.reporting.localization.Localization(__file__, 43, 14), GnuFCompiler_63396, *[], **kwargs_63397)
        
        # Assigning a type to the variable 'gnu' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'gnu', GnuFCompiler_call_result_63398)
        
        # Call to customize(...): (line 44)
        # Processing the call arguments (line 44)
        # Getting the type of 'None' (line 44)
        None_63401 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 22), 'None', False)
        # Processing the call keyword arguments (line 44)
        kwargs_63402 = {}
        # Getting the type of 'gnu' (line 44)
        gnu_63399 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'gnu', False)
        # Obtaining the member 'customize' of a type (line 44)
        customize_63400 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), gnu_63399, 'customize')
        # Calling customize(args, kwargs) (line 44)
        customize_call_result_63403 = invoke(stypy.reporting.localization.Localization(__file__, 44, 8), customize_63400, *[None_63401], **kwargs_63402)
        
        
        # Assigning a Call to a Attribute (line 45):
        
        # Assigning a Call to a Attribute (line 45):
        
        # Call to get_version(...): (line 45)
        # Processing the call keyword arguments (line 45)
        kwargs_63406 = {}
        # Getting the type of 'gnu' (line 45)
        gnu_63404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 23), 'gnu', False)
        # Obtaining the member 'get_version' of a type (line 45)
        get_version_63405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 23), gnu_63404, 'get_version')
        # Calling get_version(args, kwargs) (line 45)
        get_version_call_result_63407 = invoke(stypy.reporting.localization.Localization(__file__, 45, 23), get_version_63405, *[], **kwargs_63406)
        
        # Getting the type of 'self' (line 45)
        self_63408 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member 'version' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_63408, 'version', get_version_call_result_63407)
        
        # Assigning a Call to a Name (line 46):
        
        # Assigning a Call to a Name (line 46):
        
        # Call to get_flags_arch(...): (line 46)
        # Processing the call arguments (line 46)
        # Getting the type of 'self' (line 46)
        self_63411 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 42), 'self', False)
        # Processing the call keyword arguments (line 46)
        kwargs_63412 = {}
        # Getting the type of 'GnuFCompiler' (line 46)
        GnuFCompiler_63409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 14), 'GnuFCompiler', False)
        # Obtaining the member 'get_flags_arch' of a type (line 46)
        get_flags_arch_63410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 14), GnuFCompiler_63409, 'get_flags_arch')
        # Calling get_flags_arch(args, kwargs) (line 46)
        get_flags_arch_call_result_63413 = invoke(stypy.reporting.localization.Localization(__file__, 46, 14), get_flags_arch_63410, *[self_63411], **kwargs_63412)
        
        # Assigning a type to the variable 'opt' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'opt', get_flags_arch_call_result_63413)
        
        # Assigning a Name to a Attribute (line 47):
        
        # Assigning a Name to a Attribute (line 47):
        # Getting the type of 'vast_version' (line 47)
        vast_version_63414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 23), 'vast_version')
        # Getting the type of 'self' (line 47)
        self_63415 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'self')
        # Setting the type of the member 'version' of a type (line 47)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), self_63415, 'version', vast_version_63414)
        # Getting the type of 'opt' (line 48)
        opt_63416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 15), 'opt')
        # Assigning a type to the variable 'stypy_return_type' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 8), 'stypy_return_type', opt_63416)
        
        # ################# End of 'get_flags_arch(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_flags_arch' in the type store
        # Getting the type of 'stypy_return_type' (line 41)
        stypy_return_type_63417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_63417)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_flags_arch'
        return stypy_return_type_63417


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 0, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'VastFCompiler.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'VastFCompiler' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'VastFCompiler', VastFCompiler)

# Assigning a Str to a Name (line 10):
str_63418 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 10, 20), 'str', 'vast')
# Getting the type of 'VastFCompiler'
VastFCompiler_63419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VastFCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VastFCompiler_63419, 'compiler_type', str_63418)

# Assigning a Tuple to a Name (line 11):

# Obtaining an instance of the builtin type 'tuple' (line 11)
tuple_63420 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 23), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 11)

# Getting the type of 'VastFCompiler'
VastFCompiler_63421 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VastFCompiler')
# Setting the type of the member 'compiler_aliases' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VastFCompiler_63421, 'compiler_aliases', tuple_63420)

# Assigning a Str to a Name (line 12):
str_63422 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 18), 'str', 'Pacific-Sierra Research Fortran 90 Compiler')
# Getting the type of 'VastFCompiler'
VastFCompiler_63423 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VastFCompiler')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VastFCompiler_63423, 'description', str_63422)

# Assigning a Str to a Name (line 13):
str_63424 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 22), 'str', '\\s*Pacific-Sierra Research vf90 (Personal|Professional)\\s+(?P<version>[^\\s]*)')
# Getting the type of 'VastFCompiler'
VastFCompiler_63425 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VastFCompiler')
# Setting the type of the member 'version_pattern' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VastFCompiler_63425, 'version_pattern', str_63424)

# Assigning a Str to a Name (line 18):
str_63426 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 20), 'str', ' && function _mvfile { mv -v `basename $1` $1 ; } && _mvfile ')
# Getting the type of 'VastFCompiler'
VastFCompiler_63427 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VastFCompiler')
# Setting the type of the member 'object_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VastFCompiler_63427, 'object_switch', str_63426)

# Assigning a Dict to a Name (line 20):

# Obtaining an instance of the builtin type 'dict' (line 20)
dict_63428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 18), 'dict')
# Adding type elements to the builtin type 'dict' instance (line 20)
# Adding element type (key, value) (line 20)
str_63429 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 8), 'str', 'version_cmd')

# Obtaining an instance of the builtin type 'list' (line 21)
list_63430 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 21)
# Adding element type (line 21)
str_63431 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 26), 'str', 'vf90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_63430, str_63431)
# Adding element type (line 21)
str_63432 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 34), 'str', '-v')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 25), list_63430, str_63432)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_63428, (str_63429, list_63430))
# Adding element type (key, value) (line 20)
str_63433 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 8), 'str', 'compiler_f77')

# Obtaining an instance of the builtin type 'list' (line 22)
list_63434 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_63435 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 26), 'str', 'g77')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 25), list_63434, str_63435)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_63428, (str_63433, list_63434))
# Adding element type (key, value) (line 20)
str_63436 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 8), 'str', 'compiler_fix')

# Obtaining an instance of the builtin type 'list' (line 23)
list_63437 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)
str_63438 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_63437, str_63438)
# Adding element type (line 23)
str_63439 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 33), 'str', '-Wv,-ya')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 25), list_63437, str_63439)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_63428, (str_63436, list_63437))
# Adding element type (key, value) (line 20)
str_63440 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 8), 'str', 'compiler_f90')

# Obtaining an instance of the builtin type 'list' (line 24)
list_63441 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 24)
# Adding element type (line 24)
str_63442 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 26), 'str', 'f90')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 25), list_63441, str_63442)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_63428, (str_63440, list_63441))
# Adding element type (key, value) (line 20)
str_63443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 8), 'str', 'linker_so')

# Obtaining an instance of the builtin type 'list' (line 25)
list_63444 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 25)
# Adding element type (line 25)
str_63445 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 26), 'str', '<F90>')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 25), list_63444, str_63445)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_63428, (str_63443, list_63444))
# Adding element type (key, value) (line 20)
str_63446 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 8), 'str', 'archiver')

# Obtaining an instance of the builtin type 'list' (line 26)
list_63447 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 26)
# Adding element type (line 26)
str_63448 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 26), 'str', 'ar')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), list_63447, str_63448)
# Adding element type (line 26)
str_63449 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 32), 'str', '-cr')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 25), list_63447, str_63449)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_63428, (str_63446, list_63447))
# Adding element type (key, value) (line 20)
str_63450 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 8), 'str', 'ranlib')

# Obtaining an instance of the builtin type 'list' (line 27)
list_63451 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 25), 'list')
# Adding type elements to the builtin type 'list' instance (line 27)
# Adding element type (line 27)
str_63452 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 26), 'str', 'ranlib')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 27, 25), list_63451, str_63452)

set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 20, 18), dict_63428, (str_63450, list_63451))

# Getting the type of 'VastFCompiler'
VastFCompiler_63453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VastFCompiler')
# Setting the type of the member 'executables' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VastFCompiler_63453, 'executables', dict_63428)

# Assigning a Name to a Name (line 29):
# Getting the type of 'None' (line 29)
None_63454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'None')
# Getting the type of 'VastFCompiler'
VastFCompiler_63455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VastFCompiler')
# Setting the type of the member 'module_dir_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VastFCompiler_63455, 'module_dir_switch', None_63454)

# Assigning a Name to a Name (line 30):
# Getting the type of 'None' (line 30)
None_63456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 28), 'None')
# Getting the type of 'VastFCompiler'
VastFCompiler_63457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'VastFCompiler')
# Setting the type of the member 'module_include_switch' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), VastFCompiler_63457, 'module_include_switch', None_63456)

if (__name__ == '__main__'):
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 51, 4))
    
    # 'from distutils import log' statement (line 51)
    from distutils import log

    import_from_module(stypy.reporting.localization.Localization(__file__, 51, 4), 'distutils', None, module_type_store, ['log'], [log])
    
    
    # Call to set_verbosity(...): (line 52)
    # Processing the call arguments (line 52)
    int_63460 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 52, 22), 'int')
    # Processing the call keyword arguments (line 52)
    kwargs_63461 = {}
    # Getting the type of 'log' (line 52)
    log_63458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 52, 4), 'log', False)
    # Obtaining the member 'set_verbosity' of a type (line 52)
    set_verbosity_63459 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 52, 4), log_63458, 'set_verbosity')
    # Calling set_verbosity(args, kwargs) (line 52)
    set_verbosity_call_result_63462 = invoke(stypy.reporting.localization.Localization(__file__, 52, 4), set_verbosity_63459, *[int_63460], **kwargs_63461)
    
    stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 53, 4))
    
    # 'from numpy.distutils.fcompiler import new_fcompiler' statement (line 53)
    update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    import_63463 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 53, 4), 'numpy.distutils.fcompiler')

    if (type(import_63463) is not StypyTypeError):

        if (import_63463 != 'pyd_module'):
            __import__(import_63463)
            sys_modules_63464 = sys.modules[import_63463]
            import_from_module(stypy.reporting.localization.Localization(__file__, 53, 4), 'numpy.distutils.fcompiler', sys_modules_63464.module_type_store, module_type_store, ['new_fcompiler'])
            nest_module(stypy.reporting.localization.Localization(__file__, 53, 4), __file__, sys_modules_63464, sys_modules_63464.module_type_store, module_type_store)
        else:
            from numpy.distutils.fcompiler import new_fcompiler

            import_from_module(stypy.reporting.localization.Localization(__file__, 53, 4), 'numpy.distutils.fcompiler', None, module_type_store, ['new_fcompiler'], [new_fcompiler])

    else:
        # Assigning a type to the variable 'numpy.distutils.fcompiler' (line 53)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'numpy.distutils.fcompiler', import_63463)

    remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/fcompiler/')
    
    
    # Assigning a Call to a Name (line 54):
    
    # Assigning a Call to a Name (line 54):
    
    # Call to new_fcompiler(...): (line 54)
    # Processing the call keyword arguments (line 54)
    str_63466 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 54, 38), 'str', 'vast')
    keyword_63467 = str_63466
    kwargs_63468 = {'compiler': keyword_63467}
    # Getting the type of 'new_fcompiler' (line 54)
    new_fcompiler_63465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 15), 'new_fcompiler', False)
    # Calling new_fcompiler(args, kwargs) (line 54)
    new_fcompiler_call_result_63469 = invoke(stypy.reporting.localization.Localization(__file__, 54, 15), new_fcompiler_63465, *[], **kwargs_63468)
    
    # Assigning a type to the variable 'compiler' (line 54)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'compiler', new_fcompiler_call_result_63469)
    
    # Call to customize(...): (line 55)
    # Processing the call keyword arguments (line 55)
    kwargs_63472 = {}
    # Getting the type of 'compiler' (line 55)
    compiler_63470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 4), 'compiler', False)
    # Obtaining the member 'customize' of a type (line 55)
    customize_63471 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 4), compiler_63470, 'customize')
    # Calling customize(args, kwargs) (line 55)
    customize_call_result_63473 = invoke(stypy.reporting.localization.Localization(__file__, 55, 4), customize_63471, *[], **kwargs_63472)
    
    
    # Call to print(...): (line 56)
    # Processing the call arguments (line 56)
    
    # Call to get_version(...): (line 56)
    # Processing the call keyword arguments (line 56)
    kwargs_63477 = {}
    # Getting the type of 'compiler' (line 56)
    compiler_63475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 10), 'compiler', False)
    # Obtaining the member 'get_version' of a type (line 56)
    get_version_63476 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 10), compiler_63475, 'get_version')
    # Calling get_version(args, kwargs) (line 56)
    get_version_call_result_63478 = invoke(stypy.reporting.localization.Localization(__file__, 56, 10), get_version_63476, *[], **kwargs_63477)
    
    # Processing the call keyword arguments (line 56)
    kwargs_63479 = {}
    # Getting the type of 'print' (line 56)
    print_63474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 4), 'print', False)
    # Calling print(args, kwargs) (line 56)
    print_call_result_63480 = invoke(stypy.reporting.localization.Localization(__file__, 56, 4), print_63474, *[get_version_call_result_63478], **kwargs_63479)
    


# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
