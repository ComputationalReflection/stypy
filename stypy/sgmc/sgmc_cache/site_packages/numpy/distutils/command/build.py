
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: import sys
5: from distutils.command.build import build as old_build
6: from distutils.util import get_platform
7: from numpy.distutils.command.config_compiler import show_fortran_compilers
8: 
9: class build(old_build):
10: 
11:     sub_commands = [('config_cc',     lambda *args: True),
12:                     ('config_fc',     lambda *args: True),
13:                     ('build_src',     old_build.has_ext_modules),
14:                     ] + old_build.sub_commands
15: 
16:     user_options = old_build.user_options + [
17:         ('fcompiler=', None,
18:          "specify the Fortran compiler type"),
19:         ('parallel=', 'j',
20:          "number of parallel jobs"),
21:         ]
22: 
23:     help_options = old_build.help_options + [
24:         ('help-fcompiler', None, "list available Fortran compilers",
25:          show_fortran_compilers),
26:         ]
27: 
28:     def initialize_options(self):
29:         old_build.initialize_options(self)
30:         self.fcompiler = None
31:         self.parallel = None
32: 
33:     def finalize_options(self):
34:         if self.parallel:
35:             try:
36:                 self.parallel = int(self.parallel)
37:             except ValueError:
38:                 raise ValueError("--parallel/-j argument must be an integer")
39:         build_scripts = self.build_scripts
40:         old_build.finalize_options(self)
41:         plat_specifier = ".%s-%s" % (get_platform(), sys.version[0:3])
42:         if build_scripts is None:
43:             self.build_scripts = os.path.join(self.build_base,
44:                                               'scripts' + plat_specifier)
45: 
46:     def run(self):
47:         old_build.run(self)
48: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'import sys' statement (line 4)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.command.build import old_build' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52432 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.command.build')

if (type(import_52432) is not StypyTypeError):

    if (import_52432 != 'pyd_module'):
        __import__(import_52432)
        sys_modules_52433 = sys.modules[import_52432]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.command.build', sys_modules_52433.module_type_store, module_type_store, ['build'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_52433, sys_modules_52433.module_type_store, module_type_store)
    else:
        from distutils.command.build import build as old_build

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.command.build', None, module_type_store, ['build'], [old_build])

else:
    # Assigning a type to the variable 'distutils.command.build' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.command.build', import_52432)

# Adding an alias
module_type_store.add_alias('old_build', 'build')
remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from distutils.util import get_platform' statement (line 6)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52434 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.util')

if (type(import_52434) is not StypyTypeError):

    if (import_52434 != 'pyd_module'):
        __import__(import_52434)
        sys_modules_52435 = sys.modules[import_52434]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.util', sys_modules_52435.module_type_store, module_type_store, ['get_platform'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_52435, sys_modules_52435.module_type_store, module_type_store)
    else:
        from distutils.util import get_platform

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.util', None, module_type_store, ['get_platform'], [get_platform])

else:
    # Assigning a type to the variable 'distutils.util' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'distutils.util', import_52434)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.distutils.command.config_compiler import show_fortran_compilers' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/command/')
import_52436 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.command.config_compiler')

if (type(import_52436) is not StypyTypeError):

    if (import_52436 != 'pyd_module'):
        __import__(import_52436)
        sys_modules_52437 = sys.modules[import_52436]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.command.config_compiler', sys_modules_52437.module_type_store, module_type_store, ['show_fortran_compilers'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_52437, sys_modules_52437.module_type_store, module_type_store)
    else:
        from numpy.distutils.command.config_compiler import show_fortran_compilers

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.command.config_compiler', None, module_type_store, ['show_fortran_compilers'], [show_fortran_compilers])

else:
    # Assigning a type to the variable 'numpy.distutils.command.config_compiler' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.command.config_compiler', import_52436)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/command/')

# Declaration of the 'build' class
# Getting the type of 'old_build' (line 9)
old_build_52438 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 12), 'old_build')

class build(old_build_52438, ):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 28, 4, False)
        # Assigning a type to the variable 'self' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        build.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.initialize_options.__dict__.__setitem__('stypy_function_name', 'build.initialize_options')
        build.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to initialize_options(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'self' (line 29)
        self_52441 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 37), 'self', False)
        # Processing the call keyword arguments (line 29)
        kwargs_52442 = {}
        # Getting the type of 'old_build' (line 29)
        old_build_52439 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'old_build', False)
        # Obtaining the member 'initialize_options' of a type (line 29)
        initialize_options_52440 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 8), old_build_52439, 'initialize_options')
        # Calling initialize_options(args, kwargs) (line 29)
        initialize_options_call_result_52443 = invoke(stypy.reporting.localization.Localization(__file__, 29, 8), initialize_options_52440, *[self_52441], **kwargs_52442)
        
        
        # Assigning a Name to a Attribute (line 30):
        # Getting the type of 'None' (line 30)
        None_52444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 25), 'None')
        # Getting the type of 'self' (line 30)
        self_52445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self')
        # Setting the type of the member 'fcompiler' of a type (line 30)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_52445, 'fcompiler', None_52444)
        
        # Assigning a Name to a Attribute (line 31):
        # Getting the type of 'None' (line 31)
        None_52446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 24), 'None')
        # Getting the type of 'self' (line 31)
        self_52447 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'self')
        # Setting the type of the member 'parallel' of a type (line 31)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 8), self_52447, 'parallel', None_52446)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 28)
        stypy_return_type_52448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52448)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_52448


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 33, 4, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        build.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.finalize_options.__dict__.__setitem__('stypy_function_name', 'build.finalize_options')
        build.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        build.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Getting the type of 'self' (line 34)
        self_52449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 11), 'self')
        # Obtaining the member 'parallel' of a type (line 34)
        parallel_52450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 11), self_52449, 'parallel')
        # Testing the type of an if condition (line 34)
        if_condition_52451 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 34, 8), parallel_52450)
        # Assigning a type to the variable 'if_condition_52451' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'if_condition_52451', if_condition_52451)
        # SSA begins for if statement (line 34)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        
        # SSA begins for try-except statement (line 35)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Attribute (line 36):
        
        # Call to int(...): (line 36)
        # Processing the call arguments (line 36)
        # Getting the type of 'self' (line 36)
        self_52453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 36), 'self', False)
        # Obtaining the member 'parallel' of a type (line 36)
        parallel_52454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 36), self_52453, 'parallel')
        # Processing the call keyword arguments (line 36)
        kwargs_52455 = {}
        # Getting the type of 'int' (line 36)
        int_52452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 32), 'int', False)
        # Calling int(args, kwargs) (line 36)
        int_call_result_52456 = invoke(stypy.reporting.localization.Localization(__file__, 36, 32), int_52452, *[parallel_52454], **kwargs_52455)
        
        # Getting the type of 'self' (line 36)
        self_52457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 16), 'self')
        # Setting the type of the member 'parallel' of a type (line 36)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 16), self_52457, 'parallel', int_call_result_52456)
        # SSA branch for the except part of a try statement (line 35)
        # SSA branch for the except 'ValueError' branch of a try statement (line 35)
        module_type_store.open_ssa_branch('except')
        
        # Call to ValueError(...): (line 38)
        # Processing the call arguments (line 38)
        str_52459 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 33), 'str', '--parallel/-j argument must be an integer')
        # Processing the call keyword arguments (line 38)
        kwargs_52460 = {}
        # Getting the type of 'ValueError' (line 38)
        ValueError_52458 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 22), 'ValueError', False)
        # Calling ValueError(args, kwargs) (line 38)
        ValueError_call_result_52461 = invoke(stypy.reporting.localization.Localization(__file__, 38, 22), ValueError_52458, *[str_52459], **kwargs_52460)
        
        ensure_var_of_types(stypy.reporting.localization.Localization(__file__, 38, 16), ValueError_call_result_52461, 'raise parameter', BaseException)
        # SSA join for try-except statement (line 35)
        module_type_store = module_type_store.join_ssa_context()
        
        # SSA join for if statement (line 34)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Assigning a Attribute to a Name (line 39):
        # Getting the type of 'self' (line 39)
        self_52462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 24), 'self')
        # Obtaining the member 'build_scripts' of a type (line 39)
        build_scripts_52463 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 24), self_52462, 'build_scripts')
        # Assigning a type to the variable 'build_scripts' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 8), 'build_scripts', build_scripts_52463)
        
        # Call to finalize_options(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_52466 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 35), 'self', False)
        # Processing the call keyword arguments (line 40)
        kwargs_52467 = {}
        # Getting the type of 'old_build' (line 40)
        old_build_52464 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'old_build', False)
        # Obtaining the member 'finalize_options' of a type (line 40)
        finalize_options_52465 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), old_build_52464, 'finalize_options')
        # Calling finalize_options(args, kwargs) (line 40)
        finalize_options_call_result_52468 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), finalize_options_52465, *[self_52466], **kwargs_52467)
        
        
        # Assigning a BinOp to a Name (line 41):
        str_52469 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 25), 'str', '.%s-%s')
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_52470 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 37), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        
        # Call to get_platform(...): (line 41)
        # Processing the call keyword arguments (line 41)
        kwargs_52472 = {}
        # Getting the type of 'get_platform' (line 41)
        get_platform_52471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'get_platform', False)
        # Calling get_platform(args, kwargs) (line 41)
        get_platform_call_result_52473 = invoke(stypy.reporting.localization.Localization(__file__, 41, 37), get_platform_52471, *[], **kwargs_52472)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 37), tuple_52470, get_platform_call_result_52473)
        # Adding element type (line 41)
        
        # Obtaining the type of the subscript
        int_52474 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 65), 'int')
        int_52475 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 67), 'int')
        slice_52476 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 41, 53), int_52474, int_52475, None)
        # Getting the type of 'sys' (line 41)
        sys_52477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 53), 'sys')
        # Obtaining the member 'version' of a type (line 41)
        version_52478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 53), sys_52477, 'version')
        # Obtaining the member '__getitem__' of a type (line 41)
        getitem___52479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 41, 53), version_52478, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 41)
        subscript_call_result_52480 = invoke(stypy.reporting.localization.Localization(__file__, 41, 53), getitem___52479, slice_52476)
        
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 37), tuple_52470, subscript_call_result_52480)
        
        # Applying the binary operator '%' (line 41)
        result_mod_52481 = python_operator(stypy.reporting.localization.Localization(__file__, 41, 25), '%', str_52469, tuple_52470)
        
        # Assigning a type to the variable 'plat_specifier' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'plat_specifier', result_mod_52481)
        
        # Type idiom detected: calculating its left and rigth part (line 42)
        # Getting the type of 'build_scripts' (line 42)
        build_scripts_52482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 11), 'build_scripts')
        # Getting the type of 'None' (line 42)
        None_52483 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 28), 'None')
        
        (may_be_52484, more_types_in_union_52485) = may_be_none(build_scripts_52482, None_52483)

        if may_be_52484:

            if more_types_in_union_52485:
                # Runtime conditional SSA (line 42)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Attribute (line 43):
            
            # Call to join(...): (line 43)
            # Processing the call arguments (line 43)
            # Getting the type of 'self' (line 43)
            self_52489 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 46), 'self', False)
            # Obtaining the member 'build_base' of a type (line 43)
            build_base_52490 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 46), self_52489, 'build_base')
            str_52491 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 46), 'str', 'scripts')
            # Getting the type of 'plat_specifier' (line 44)
            plat_specifier_52492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 58), 'plat_specifier', False)
            # Applying the binary operator '+' (line 44)
            result_add_52493 = python_operator(stypy.reporting.localization.Localization(__file__, 44, 46), '+', str_52491, plat_specifier_52492)
            
            # Processing the call keyword arguments (line 43)
            kwargs_52494 = {}
            # Getting the type of 'os' (line 43)
            os_52486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'os', False)
            # Obtaining the member 'path' of a type (line 43)
            path_52487 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 33), os_52486, 'path')
            # Obtaining the member 'join' of a type (line 43)
            join_52488 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 33), path_52487, 'join')
            # Calling join(args, kwargs) (line 43)
            join_call_result_52495 = invoke(stypy.reporting.localization.Localization(__file__, 43, 33), join_52488, *[build_base_52490, result_add_52493], **kwargs_52494)
            
            # Getting the type of 'self' (line 43)
            self_52496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self')
            # Setting the type of the member 'build_scripts' of a type (line 43)
            module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_52496, 'build_scripts', join_call_result_52495)

            if more_types_in_union_52485:
                # SSA join for if statement (line 42)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 33)
        stypy_return_type_52497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52497)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_52497


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 46, 4, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        build.run.__dict__.__setitem__('stypy_localization', localization)
        build.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        build.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        build.run.__dict__.__setitem__('stypy_function_name', 'build.run')
        build.run.__dict__.__setitem__('stypy_param_names_list', [])
        build.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        build.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        build.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        build.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        build.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        build.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to run(...): (line 47)
        # Processing the call arguments (line 47)
        # Getting the type of 'self' (line 47)
        self_52500 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 22), 'self', False)
        # Processing the call keyword arguments (line 47)
        kwargs_52501 = {}
        # Getting the type of 'old_build' (line 47)
        old_build_52498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 8), 'old_build', False)
        # Obtaining the member 'run' of a type (line 47)
        run_52499 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 47, 8), old_build_52498, 'run')
        # Calling run(args, kwargs) (line 47)
        run_call_result_52502 = invoke(stypy.reporting.localization.Localization(__file__, 47, 8), run_52499, *[self_52500], **kwargs_52501)
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 46)
        stypy_return_type_52503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_52503)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_52503


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
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'build.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'build' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'build', build)

# Assigning a BinOp to a Name (line 11):

# Obtaining an instance of the builtin type 'list' (line 11)
list_52504 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 11)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'tuple' (line 11)
tuple_52505 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 11)
# Adding element type (line 11)
str_52506 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 21), 'str', 'config_cc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 21), tuple_52505, str_52506)
# Adding element type (line 11)

@norecursion
def _stypy_temp_lambda_20(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_20'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_20', 11, 38, True)
    # Passed parameters checking function
    _stypy_temp_lambda_20.stypy_localization = localization
    _stypy_temp_lambda_20.stypy_type_of_self = None
    _stypy_temp_lambda_20.stypy_type_store = module_type_store
    _stypy_temp_lambda_20.stypy_function_name = '_stypy_temp_lambda_20'
    _stypy_temp_lambda_20.stypy_param_names_list = []
    _stypy_temp_lambda_20.stypy_varargs_param_name = 'args'
    _stypy_temp_lambda_20.stypy_kwargs_param_name = None
    _stypy_temp_lambda_20.stypy_call_defaults = defaults
    _stypy_temp_lambda_20.stypy_call_varargs = varargs
    _stypy_temp_lambda_20.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_20', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_20', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'True' (line 11)
    True_52507 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 52), 'True')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 11)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 38), 'stypy_return_type', True_52507)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_20' in the type store
    # Getting the type of 'stypy_return_type' (line 11)
    stypy_return_type_52508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 38), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52508)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_20'
    return stypy_return_type_52508

# Assigning a type to the variable '_stypy_temp_lambda_20' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 38), '_stypy_temp_lambda_20', _stypy_temp_lambda_20)
# Getting the type of '_stypy_temp_lambda_20' (line 11)
_stypy_temp_lambda_20_52509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 38), '_stypy_temp_lambda_20')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 21), tuple_52505, _stypy_temp_lambda_20_52509)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 19), list_52504, tuple_52505)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'tuple' (line 12)
tuple_52510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 12)
# Adding element type (line 12)
str_52511 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 21), 'str', 'config_fc')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), tuple_52510, str_52511)
# Adding element type (line 12)

@norecursion
def _stypy_temp_lambda_21(localization, *varargs, **kwargs):
    global module_type_store
    # Assign values to the parameters with defaults
    defaults = []
    # Create a new context for function '_stypy_temp_lambda_21'
    module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_21', 12, 38, True)
    # Passed parameters checking function
    _stypy_temp_lambda_21.stypy_localization = localization
    _stypy_temp_lambda_21.stypy_type_of_self = None
    _stypy_temp_lambda_21.stypy_type_store = module_type_store
    _stypy_temp_lambda_21.stypy_function_name = '_stypy_temp_lambda_21'
    _stypy_temp_lambda_21.stypy_param_names_list = []
    _stypy_temp_lambda_21.stypy_varargs_param_name = 'args'
    _stypy_temp_lambda_21.stypy_kwargs_param_name = None
    _stypy_temp_lambda_21.stypy_call_defaults = defaults
    _stypy_temp_lambda_21.stypy_call_varargs = varargs
    _stypy_temp_lambda_21.stypy_call_kwargs = kwargs
    arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_21', [], 'args', None, defaults, varargs, kwargs)

    if is_error_type(arguments):
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        return arguments

    # Stacktrace push for error reporting
    localization.set_stack_trace('_stypy_temp_lambda_21', [], arguments)
    # Default return type storage variable (SSA)
    # Assigning a type to the variable 'stypy_return_type'
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
    
    
    # ################# Begin of the lambda function code ##################

    # Getting the type of 'True' (line 12)
    True_52512 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 52), 'True')
    # Assigning the return type of the lambda function
    # Assigning a type to the variable 'stypy_return_type' (line 12)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 38), 'stypy_return_type', True_52512)
    
    # ################# End of the lambda function code ##################

    # Stacktrace pop (error reporting)
    localization.unset_stack_trace()
    
    # Storing the return type of function '_stypy_temp_lambda_21' in the type store
    # Getting the type of 'stypy_return_type' (line 12)
    stypy_return_type_52513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 38), 'stypy_return_type')
    module_type_store.store_return_type_of_current_context(stypy_return_type_52513)
    
    # Destroy the current context
    module_type_store = module_type_store.close_function_context()
    
    # Return type of the function '_stypy_temp_lambda_21'
    return stypy_return_type_52513

# Assigning a type to the variable '_stypy_temp_lambda_21' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 38), '_stypy_temp_lambda_21', _stypy_temp_lambda_21)
# Getting the type of '_stypy_temp_lambda_21' (line 12)
_stypy_temp_lambda_21_52514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 38), '_stypy_temp_lambda_21')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 12, 21), tuple_52510, _stypy_temp_lambda_21_52514)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 19), list_52504, tuple_52510)
# Adding element type (line 11)

# Obtaining an instance of the builtin type 'tuple' (line 13)
tuple_52515 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 13)
# Adding element type (line 13)
str_52516 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 21), 'str', 'build_src')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), tuple_52515, str_52516)
# Adding element type (line 13)
# Getting the type of 'old_build' (line 13)
old_build_52517 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 38), 'old_build')
# Obtaining the member 'has_ext_modules' of a type (line 13)
has_ext_modules_52518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 38), old_build_52517, 'has_ext_modules')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 13, 21), tuple_52515, has_ext_modules_52518)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 11, 19), list_52504, tuple_52515)

# Getting the type of 'old_build' (line 14)
old_build_52519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 24), 'old_build')
# Obtaining the member 'sub_commands' of a type (line 14)
sub_commands_52520 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 24), old_build_52519, 'sub_commands')
# Applying the binary operator '+' (line 11)
result_add_52521 = python_operator(stypy.reporting.localization.Localization(__file__, 11, 19), '+', list_52504, sub_commands_52520)

# Getting the type of 'build'
build_52522 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Setting the type of the member 'sub_commands' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_52522, 'sub_commands', result_add_52521)

# Assigning a BinOp to a Name (line 16):
# Getting the type of 'old_build' (line 16)
old_build_52523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 19), 'old_build')
# Obtaining the member 'user_options' of a type (line 16)
user_options_52524 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 19), old_build_52523, 'user_options')

# Obtaining an instance of the builtin type 'list' (line 16)
list_52525 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 44), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'tuple' (line 17)
tuple_52526 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 17)
# Adding element type (line 17)
str_52527 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 9), 'str', 'fcompiler=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 9), tuple_52526, str_52527)
# Adding element type (line 17)
# Getting the type of 'None' (line 17)
None_52528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 23), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 9), tuple_52526, None_52528)
# Adding element type (line 17)
str_52529 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 9), 'str', 'specify the Fortran compiler type')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 9), tuple_52526, str_52529)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 44), list_52525, tuple_52526)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'tuple' (line 19)
tuple_52530 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 19)
# Adding element type (line 19)
str_52531 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 9), 'str', 'parallel=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 9), tuple_52530, str_52531)
# Adding element type (line 19)
str_52532 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 22), 'str', 'j')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 9), tuple_52530, str_52532)
# Adding element type (line 19)
str_52533 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 20, 9), 'str', 'number of parallel jobs')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 9), tuple_52530, str_52533)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 44), list_52525, tuple_52530)

# Applying the binary operator '+' (line 16)
result_add_52534 = python_operator(stypy.reporting.localization.Localization(__file__, 16, 19), '+', user_options_52524, list_52525)

# Getting the type of 'build'
build_52535 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_52535, 'user_options', result_add_52534)

# Assigning a BinOp to a Name (line 23):
# Getting the type of 'old_build' (line 23)
old_build_52536 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 19), 'old_build')
# Obtaining the member 'help_options' of a type (line 23)
help_options_52537 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 19), old_build_52536, 'help_options')

# Obtaining an instance of the builtin type 'list' (line 23)
list_52538 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 44), 'list')
# Adding type elements to the builtin type 'list' instance (line 23)
# Adding element type (line 23)

# Obtaining an instance of the builtin type 'tuple' (line 24)
tuple_52539 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 24)
# Adding element type (line 24)
str_52540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 9), 'str', 'help-fcompiler')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_52539, str_52540)
# Adding element type (line 24)
# Getting the type of 'None' (line 24)
None_52541 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 27), 'None')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_52539, None_52541)
# Adding element type (line 24)
str_52542 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, 33), 'str', 'list available Fortran compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_52539, str_52542)
# Adding element type (line 24)
# Getting the type of 'show_fortran_compilers' (line 25)
show_fortran_compilers_52543 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 9), 'show_fortran_compilers')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 24, 9), tuple_52539, show_fortran_compilers_52543)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 44), list_52538, tuple_52539)

# Applying the binary operator '+' (line 23)
result_add_52544 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 19), '+', help_options_52537, list_52538)

# Getting the type of 'build'
build_52545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'build')
# Setting the type of the member 'help_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), build_52545, 'help_options', result_add_52544)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
