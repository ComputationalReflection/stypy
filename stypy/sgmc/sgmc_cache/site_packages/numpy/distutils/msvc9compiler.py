
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: import distutils.msvc9compiler
5: from distutils.msvc9compiler import *
6: 
7: 
8: class MSVCCompiler(distutils.msvc9compiler.MSVCCompiler):
9:     def __init__(self, verbose=0, dry_run=0, force=0):
10:         distutils.msvc9compiler.MSVCCompiler.__init__(self, verbose, dry_run, force)
11: 
12:     def initialize(self, plat_name=None):
13:         environ_lib = os.getenv('lib')
14:         environ_include = os.getenv('include')
15:         distutils.msvc9compiler.MSVCCompiler.initialize(self, plat_name)
16:         if environ_lib is not None:
17:             os.environ['lib'] = environ_lib + os.environ['lib']
18:         if environ_include is not None:
19:             os.environ['include'] = environ_include + os.environ['include']
20: 
21:     def manifest_setup_ldargs(self, output_filename, build_temp, ld_args):
22:         ld_args.append('/MANIFEST')
23:         distutils.msvc9compiler.MSVCCompiler.manifest_setup_ldargs(self,
24:                                                                    output_filename,
25:                                                                    build_temp, ld_args)
26: 

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

# 'import distutils.msvc9compiler' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_43966 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.msvc9compiler')

if (type(import_43966) is not StypyTypeError):

    if (import_43966 != 'pyd_module'):
        __import__(import_43966)
        sys_modules_43967 = sys.modules[import_43966]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.msvc9compiler', sys_modules_43967.module_type_store, module_type_store)
    else:
        import distutils.msvc9compiler

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.msvc9compiler', distutils.msvc9compiler, module_type_store)

else:
    # Assigning a type to the variable 'distutils.msvc9compiler' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.msvc9compiler', import_43966)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.msvc9compiler import ' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_43968 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.msvc9compiler')

if (type(import_43968) is not StypyTypeError):

    if (import_43968 != 'pyd_module'):
        __import__(import_43968)
        sys_modules_43969 = sys.modules[import_43968]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.msvc9compiler', sys_modules_43969.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_43969, sys_modules_43969.module_type_store, module_type_store)
    else:
        from distutils.msvc9compiler import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.msvc9compiler', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'distutils.msvc9compiler' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.msvc9compiler', import_43968)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# Declaration of the 'MSVCCompiler' class
# Getting the type of 'distutils' (line 8)
distutils_43970 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 8, 19), 'distutils')
# Obtaining the member 'msvc9compiler' of a type (line 8)
msvc9compiler_43971 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 19), distutils_43970, 'msvc9compiler')
# Obtaining the member 'MSVCCompiler' of a type (line 8)
MSVCCompiler_43972 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 8, 19), msvc9compiler_43971, 'MSVCCompiler')

class MSVCCompiler(MSVCCompiler_43972, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_43973 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 31), 'int')
        int_43974 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 42), 'int')
        int_43975 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, 51), 'int')
        defaults = [int_43973, int_43974, int_43975]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 9, 4, False)
        # Assigning a type to the variable 'self' (line 10)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['verbose', 'dry_run', 'force'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 10)
        # Processing the call arguments (line 10)
        # Getting the type of 'self' (line 10)
        self_43980 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 54), 'self', False)
        # Getting the type of 'verbose' (line 10)
        verbose_43981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 60), 'verbose', False)
        # Getting the type of 'dry_run' (line 10)
        dry_run_43982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 69), 'dry_run', False)
        # Getting the type of 'force' (line 10)
        force_43983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 78), 'force', False)
        # Processing the call keyword arguments (line 10)
        kwargs_43984 = {}
        # Getting the type of 'distutils' (line 10)
        distutils_43976 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 8), 'distutils', False)
        # Obtaining the member 'msvc9compiler' of a type (line 10)
        msvc9compiler_43977 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), distutils_43976, 'msvc9compiler')
        # Obtaining the member 'MSVCCompiler' of a type (line 10)
        MSVCCompiler_43978 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), msvc9compiler_43977, 'MSVCCompiler')
        # Obtaining the member '__init__' of a type (line 10)
        init___43979 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 8), MSVCCompiler_43978, '__init__')
        # Calling __init__(args, kwargs) (line 10)
        init___call_result_43985 = invoke(stypy.reporting.localization.Localization(__file__, 10, 8), init___43979, *[self_43980, verbose_43981, dry_run_43982, force_43983], **kwargs_43984)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def initialize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 12)
        None_43986 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 35), 'None')
        defaults = [None_43986]
        # Create a new context for function 'initialize'
        module_type_store = module_type_store.open_function_context('initialize', 12, 4, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.initialize')
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_param_names_list', ['plat_name'])
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.initialize.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.initialize', ['plat_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'initialize', localization, ['plat_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'initialize(...)' code ##################

        
        # Assigning a Call to a Name (line 13):
        
        # Call to getenv(...): (line 13)
        # Processing the call arguments (line 13)
        str_43989 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 32), 'str', 'lib')
        # Processing the call keyword arguments (line 13)
        kwargs_43990 = {}
        # Getting the type of 'os' (line 13)
        os_43987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 22), 'os', False)
        # Obtaining the member 'getenv' of a type (line 13)
        getenv_43988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 13, 22), os_43987, 'getenv')
        # Calling getenv(args, kwargs) (line 13)
        getenv_call_result_43991 = invoke(stypy.reporting.localization.Localization(__file__, 13, 22), getenv_43988, *[str_43989], **kwargs_43990)
        
        # Assigning a type to the variable 'environ_lib' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 8), 'environ_lib', getenv_call_result_43991)
        
        # Assigning a Call to a Name (line 14):
        
        # Call to getenv(...): (line 14)
        # Processing the call arguments (line 14)
        str_43994 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 36), 'str', 'include')
        # Processing the call keyword arguments (line 14)
        kwargs_43995 = {}
        # Getting the type of 'os' (line 14)
        os_43992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 26), 'os', False)
        # Obtaining the member 'getenv' of a type (line 14)
        getenv_43993 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 14, 26), os_43992, 'getenv')
        # Calling getenv(args, kwargs) (line 14)
        getenv_call_result_43996 = invoke(stypy.reporting.localization.Localization(__file__, 14, 26), getenv_43993, *[str_43994], **kwargs_43995)
        
        # Assigning a type to the variable 'environ_include' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 8), 'environ_include', getenv_call_result_43996)
        
        # Call to initialize(...): (line 15)
        # Processing the call arguments (line 15)
        # Getting the type of 'self' (line 15)
        self_44001 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 56), 'self', False)
        # Getting the type of 'plat_name' (line 15)
        plat_name_44002 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 62), 'plat_name', False)
        # Processing the call keyword arguments (line 15)
        kwargs_44003 = {}
        # Getting the type of 'distutils' (line 15)
        distutils_43997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'distutils', False)
        # Obtaining the member 'msvc9compiler' of a type (line 15)
        msvc9compiler_43998 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), distutils_43997, 'msvc9compiler')
        # Obtaining the member 'MSVCCompiler' of a type (line 15)
        MSVCCompiler_43999 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), msvc9compiler_43998, 'MSVCCompiler')
        # Obtaining the member 'initialize' of a type (line 15)
        initialize_44000 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 8), MSVCCompiler_43999, 'initialize')
        # Calling initialize(args, kwargs) (line 15)
        initialize_call_result_44004 = invoke(stypy.reporting.localization.Localization(__file__, 15, 8), initialize_44000, *[self_44001, plat_name_44002], **kwargs_44003)
        
        
        # Type idiom detected: calculating its left and rigth part (line 16)
        # Getting the type of 'environ_lib' (line 16)
        environ_lib_44005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'environ_lib')
        # Getting the type of 'None' (line 16)
        None_44006 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 30), 'None')
        
        (may_be_44007, more_types_in_union_44008) = may_not_be_none(environ_lib_44005, None_44006)

        if may_be_44007:

            if more_types_in_union_44008:
                # Runtime conditional SSA (line 16)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Subscript (line 17):
            # Getting the type of 'environ_lib' (line 17)
            environ_lib_44009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 32), 'environ_lib')
            
            # Obtaining the type of the subscript
            str_44010 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 57), 'str', 'lib')
            # Getting the type of 'os' (line 17)
            os_44011 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 46), 'os')
            # Obtaining the member 'environ' of a type (line 17)
            environ_44012 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 46), os_44011, 'environ')
            # Obtaining the member '__getitem__' of a type (line 17)
            getitem___44013 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 46), environ_44012, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 17)
            subscript_call_result_44014 = invoke(stypy.reporting.localization.Localization(__file__, 17, 46), getitem___44013, str_44010)
            
            # Applying the binary operator '+' (line 17)
            result_add_44015 = python_operator(stypy.reporting.localization.Localization(__file__, 17, 32), '+', environ_lib_44009, subscript_call_result_44014)
            
            # Getting the type of 'os' (line 17)
            os_44016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 12), 'os')
            # Obtaining the member 'environ' of a type (line 17)
            environ_44017 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 12), os_44016, 'environ')
            str_44018 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 23), 'str', 'lib')
            # Storing an element on a container (line 17)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 17, 12), environ_44017, (str_44018, result_add_44015))

            if more_types_in_union_44008:
                # SSA join for if statement (line 16)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 18)
        # Getting the type of 'environ_include' (line 18)
        environ_include_44019 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'environ_include')
        # Getting the type of 'None' (line 18)
        None_44020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 34), 'None')
        
        (may_be_44021, more_types_in_union_44022) = may_not_be_none(environ_include_44019, None_44020)

        if may_be_44021:

            if more_types_in_union_44022:
                # Runtime conditional SSA (line 18)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Subscript (line 19):
            # Getting the type of 'environ_include' (line 19)
            environ_include_44023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 36), 'environ_include')
            
            # Obtaining the type of the subscript
            str_44024 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 65), 'str', 'include')
            # Getting the type of 'os' (line 19)
            os_44025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 54), 'os')
            # Obtaining the member 'environ' of a type (line 19)
            environ_44026 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 54), os_44025, 'environ')
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___44027 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 54), environ_44026, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_44028 = invoke(stypy.reporting.localization.Localization(__file__, 19, 54), getitem___44027, str_44024)
            
            # Applying the binary operator '+' (line 19)
            result_add_44029 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 36), '+', environ_include_44023, subscript_call_result_44028)
            
            # Getting the type of 'os' (line 19)
            os_44030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'os')
            # Obtaining the member 'environ' of a type (line 19)
            environ_44031 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), os_44030, 'environ')
            str_44032 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', 'include')
            # Storing an element on a container (line 19)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 12), environ_44031, (str_44032, result_add_44029))

            if more_types_in_union_44022:
                # SSA join for if statement (line 18)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # ################# End of 'initialize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize' in the type store
        # Getting the type of 'stypy_return_type' (line 12)
        stypy_return_type_44033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44033)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize'
        return stypy_return_type_44033


    @norecursion
    def manifest_setup_ldargs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'manifest_setup_ldargs'
        module_type_store = module_type_store.open_function_context('manifest_setup_ldargs', 21, 4, False)
        # Assigning a type to the variable 'self' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_localization', localization)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_type_store', module_type_store)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_function_name', 'MSVCCompiler.manifest_setup_ldargs')
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_param_names_list', ['output_filename', 'build_temp', 'ld_args'])
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_varargs_param_name', None)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_call_defaults', defaults)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_call_varargs', varargs)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        MSVCCompiler.manifest_setup_ldargs.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'MSVCCompiler.manifest_setup_ldargs', ['output_filename', 'build_temp', 'ld_args'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'manifest_setup_ldargs', localization, ['output_filename', 'build_temp', 'ld_args'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'manifest_setup_ldargs(...)' code ##################

        
        # Call to append(...): (line 22)
        # Processing the call arguments (line 22)
        str_44036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', '/MANIFEST')
        # Processing the call keyword arguments (line 22)
        kwargs_44037 = {}
        # Getting the type of 'ld_args' (line 22)
        ld_args_44034 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'ld_args', False)
        # Obtaining the member 'append' of a type (line 22)
        append_44035 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 22, 8), ld_args_44034, 'append')
        # Calling append(args, kwargs) (line 22)
        append_call_result_44038 = invoke(stypy.reporting.localization.Localization(__file__, 22, 8), append_44035, *[str_44036], **kwargs_44037)
        
        
        # Call to manifest_setup_ldargs(...): (line 23)
        # Processing the call arguments (line 23)
        # Getting the type of 'self' (line 23)
        self_44043 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 67), 'self', False)
        # Getting the type of 'output_filename' (line 24)
        output_filename_44044 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 67), 'output_filename', False)
        # Getting the type of 'build_temp' (line 25)
        build_temp_44045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 67), 'build_temp', False)
        # Getting the type of 'ld_args' (line 25)
        ld_args_44046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 79), 'ld_args', False)
        # Processing the call keyword arguments (line 23)
        kwargs_44047 = {}
        # Getting the type of 'distutils' (line 23)
        distutils_44039 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'distutils', False)
        # Obtaining the member 'msvc9compiler' of a type (line 23)
        msvc9compiler_44040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), distutils_44039, 'msvc9compiler')
        # Obtaining the member 'MSVCCompiler' of a type (line 23)
        MSVCCompiler_44041 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), msvc9compiler_44040, 'MSVCCompiler')
        # Obtaining the member 'manifest_setup_ldargs' of a type (line 23)
        manifest_setup_ldargs_44042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), MSVCCompiler_44041, 'manifest_setup_ldargs')
        # Calling manifest_setup_ldargs(args, kwargs) (line 23)
        manifest_setup_ldargs_call_result_44048 = invoke(stypy.reporting.localization.Localization(__file__, 23, 8), manifest_setup_ldargs_44042, *[self_44043, output_filename_44044, build_temp_44045, ld_args_44046], **kwargs_44047)
        
        
        # ################# End of 'manifest_setup_ldargs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'manifest_setup_ldargs' in the type store
        # Getting the type of 'stypy_return_type' (line 21)
        stypy_return_type_44049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44049)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'manifest_setup_ldargs'
        return stypy_return_type_44049


# Assigning a type to the variable 'MSVCCompiler' (line 8)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'MSVCCompiler', MSVCCompiler)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
