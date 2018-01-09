
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: import os
4: import distutils.msvccompiler
5: from distutils.msvccompiler import *
6: 
7: from .system_info import platform_bits
8: 
9: 
10: class MSVCCompiler(distutils.msvccompiler.MSVCCompiler):
11:     def __init__(self, verbose=0, dry_run=0, force=0):
12:         distutils.msvccompiler.MSVCCompiler.__init__(self, verbose, dry_run, force)
13: 
14:     def initialize(self, plat_name=None):
15:         environ_lib = os.getenv('lib')
16:         environ_include = os.getenv('include')
17:         distutils.msvccompiler.MSVCCompiler.initialize(self, plat_name)
18:         if environ_lib is not None:
19:             os.environ['lib'] = environ_lib + os.environ['lib']
20:         if environ_include is not None:
21:             os.environ['include'] = environ_include + os.environ['include']
22:         if platform_bits == 32:
23:             # msvc9 building for 32 bits requires SSE2 to work around a
24:             # compiler bug.
25:             self.compile_options += ['/arch:SSE2']
26:             self.compile_options_debug += ['/arch:SSE2']
27: 

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

# 'import distutils.msvccompiler' statement (line 4)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_44050 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.msvccompiler')

if (type(import_44050) is not StypyTypeError):

    if (import_44050 != 'pyd_module'):
        __import__(import_44050)
        sys_modules_44051 = sys.modules[import_44050]
        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.msvccompiler', sys_modules_44051.module_type_store, module_type_store)
    else:
        import distutils.msvccompiler

        import_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.msvccompiler', distutils.msvccompiler, module_type_store)

else:
    # Assigning a type to the variable 'distutils.msvccompiler' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'distutils.msvccompiler', import_44050)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from distutils.msvccompiler import ' statement (line 5)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_44052 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.msvccompiler')

if (type(import_44052) is not StypyTypeError):

    if (import_44052 != 'pyd_module'):
        __import__(import_44052)
        sys_modules_44053 = sys.modules[import_44052]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.msvccompiler', sys_modules_44053.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_44053, sys_modules_44053.module_type_store, module_type_store)
    else:
        from distutils.msvccompiler import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.msvccompiler', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'distutils.msvccompiler' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'distutils.msvccompiler', import_44052)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from numpy.distutils.system_info import platform_bits' statement (line 7)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_44054 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.system_info')

if (type(import_44054) is not StypyTypeError):

    if (import_44054 != 'pyd_module'):
        __import__(import_44054)
        sys_modules_44055 = sys.modules[import_44054]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.system_info', sys_modules_44055.module_type_store, module_type_store, ['platform_bits'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_44055, sys_modules_44055.module_type_store, module_type_store)
    else:
        from numpy.distutils.system_info import platform_bits

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.system_info', None, module_type_store, ['platform_bits'], [platform_bits])

else:
    # Assigning a type to the variable 'numpy.distutils.system_info' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'numpy.distutils.system_info', import_44054)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# Declaration of the 'MSVCCompiler' class
# Getting the type of 'distutils' (line 10)
distutils_44056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 10, 19), 'distutils')
# Obtaining the member 'msvccompiler' of a type (line 10)
msvccompiler_44057 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 19), distutils_44056, 'msvccompiler')
# Obtaining the member 'MSVCCompiler' of a type (line 10)
MSVCCompiler_44058 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 10, 19), msvccompiler_44057, 'MSVCCompiler')

class MSVCCompiler(MSVCCompiler_44058, ):

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_44059 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 31), 'int')
        int_44060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 42), 'int')
        int_44061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 51), 'int')
        defaults = [int_44059, int_44060, int_44061]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
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

        
        # Call to __init__(...): (line 12)
        # Processing the call arguments (line 12)
        # Getting the type of 'self' (line 12)
        self_44066 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 53), 'self', False)
        # Getting the type of 'verbose' (line 12)
        verbose_44067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 59), 'verbose', False)
        # Getting the type of 'dry_run' (line 12)
        dry_run_44068 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 68), 'dry_run', False)
        # Getting the type of 'force' (line 12)
        force_44069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 77), 'force', False)
        # Processing the call keyword arguments (line 12)
        kwargs_44070 = {}
        # Getting the type of 'distutils' (line 12)
        distutils_44062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 8), 'distutils', False)
        # Obtaining the member 'msvccompiler' of a type (line 12)
        msvccompiler_44063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), distutils_44062, 'msvccompiler')
        # Obtaining the member 'MSVCCompiler' of a type (line 12)
        MSVCCompiler_44064 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), msvccompiler_44063, 'MSVCCompiler')
        # Obtaining the member '__init__' of a type (line 12)
        init___44065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 12, 8), MSVCCompiler_44064, '__init__')
        # Calling __init__(args, kwargs) (line 12)
        init___call_result_44071 = invoke(stypy.reporting.localization.Localization(__file__, 12, 8), init___44065, *[self_44066, verbose_44067, dry_run_44068, force_44069], **kwargs_44070)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def initialize(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 14)
        None_44072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 35), 'None')
        defaults = [None_44072]
        # Create a new context for function 'initialize'
        module_type_store = module_type_store.open_function_context('initialize', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
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

        
        # Assigning a Call to a Name (line 15):
        
        # Call to getenv(...): (line 15)
        # Processing the call arguments (line 15)
        str_44075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'str', 'lib')
        # Processing the call keyword arguments (line 15)
        kwargs_44076 = {}
        # Getting the type of 'os' (line 15)
        os_44073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 22), 'os', False)
        # Obtaining the member 'getenv' of a type (line 15)
        getenv_44074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 15, 22), os_44073, 'getenv')
        # Calling getenv(args, kwargs) (line 15)
        getenv_call_result_44077 = invoke(stypy.reporting.localization.Localization(__file__, 15, 22), getenv_44074, *[str_44075], **kwargs_44076)
        
        # Assigning a type to the variable 'environ_lib' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'environ_lib', getenv_call_result_44077)
        
        # Assigning a Call to a Name (line 16):
        
        # Call to getenv(...): (line 16)
        # Processing the call arguments (line 16)
        str_44080 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 36), 'str', 'include')
        # Processing the call keyword arguments (line 16)
        kwargs_44081 = {}
        # Getting the type of 'os' (line 16)
        os_44078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 26), 'os', False)
        # Obtaining the member 'getenv' of a type (line 16)
        getenv_44079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 26), os_44078, 'getenv')
        # Calling getenv(args, kwargs) (line 16)
        getenv_call_result_44082 = invoke(stypy.reporting.localization.Localization(__file__, 16, 26), getenv_44079, *[str_44080], **kwargs_44081)
        
        # Assigning a type to the variable 'environ_include' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'environ_include', getenv_call_result_44082)
        
        # Call to initialize(...): (line 17)
        # Processing the call arguments (line 17)
        # Getting the type of 'self' (line 17)
        self_44087 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 55), 'self', False)
        # Getting the type of 'plat_name' (line 17)
        plat_name_44088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 61), 'plat_name', False)
        # Processing the call keyword arguments (line 17)
        kwargs_44089 = {}
        # Getting the type of 'distutils' (line 17)
        distutils_44083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'distutils', False)
        # Obtaining the member 'msvccompiler' of a type (line 17)
        msvccompiler_44084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), distutils_44083, 'msvccompiler')
        # Obtaining the member 'MSVCCompiler' of a type (line 17)
        MSVCCompiler_44085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), msvccompiler_44084, 'MSVCCompiler')
        # Obtaining the member 'initialize' of a type (line 17)
        initialize_44086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 8), MSVCCompiler_44085, 'initialize')
        # Calling initialize(args, kwargs) (line 17)
        initialize_call_result_44090 = invoke(stypy.reporting.localization.Localization(__file__, 17, 8), initialize_44086, *[self_44087, plat_name_44088], **kwargs_44089)
        
        
        # Type idiom detected: calculating its left and rigth part (line 18)
        # Getting the type of 'environ_lib' (line 18)
        environ_lib_44091 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'environ_lib')
        # Getting the type of 'None' (line 18)
        None_44092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 30), 'None')
        
        (may_be_44093, more_types_in_union_44094) = may_not_be_none(environ_lib_44091, None_44092)

        if may_be_44093:

            if more_types_in_union_44094:
                # Runtime conditional SSA (line 18)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Subscript (line 19):
            # Getting the type of 'environ_lib' (line 19)
            environ_lib_44095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 32), 'environ_lib')
            
            # Obtaining the type of the subscript
            str_44096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 57), 'str', 'lib')
            # Getting the type of 'os' (line 19)
            os_44097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 46), 'os')
            # Obtaining the member 'environ' of a type (line 19)
            environ_44098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 46), os_44097, 'environ')
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___44099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 46), environ_44098, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_44100 = invoke(stypy.reporting.localization.Localization(__file__, 19, 46), getitem___44099, str_44096)
            
            # Applying the binary operator '+' (line 19)
            result_add_44101 = python_operator(stypy.reporting.localization.Localization(__file__, 19, 32), '+', environ_lib_44095, subscript_call_result_44100)
            
            # Getting the type of 'os' (line 19)
            os_44102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'os')
            # Obtaining the member 'environ' of a type (line 19)
            environ_44103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 12), os_44102, 'environ')
            str_44104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 23), 'str', 'lib')
            # Storing an element on a container (line 19)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 19, 12), environ_44103, (str_44104, result_add_44101))

            if more_types_in_union_44094:
                # SSA join for if statement (line 18)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 20)
        # Getting the type of 'environ_include' (line 20)
        environ_include_44105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 8), 'environ_include')
        # Getting the type of 'None' (line 20)
        None_44106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 34), 'None')
        
        (may_be_44107, more_types_in_union_44108) = may_not_be_none(environ_include_44105, None_44106)

        if may_be_44107:

            if more_types_in_union_44108:
                # Runtime conditional SSA (line 20)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a BinOp to a Subscript (line 21):
            # Getting the type of 'environ_include' (line 21)
            environ_include_44109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 36), 'environ_include')
            
            # Obtaining the type of the subscript
            str_44110 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 65), 'str', 'include')
            # Getting the type of 'os' (line 21)
            os_44111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 54), 'os')
            # Obtaining the member 'environ' of a type (line 21)
            environ_44112 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 54), os_44111, 'environ')
            # Obtaining the member '__getitem__' of a type (line 21)
            getitem___44113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 54), environ_44112, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 21)
            subscript_call_result_44114 = invoke(stypy.reporting.localization.Localization(__file__, 21, 54), getitem___44113, str_44110)
            
            # Applying the binary operator '+' (line 21)
            result_add_44115 = python_operator(stypy.reporting.localization.Localization(__file__, 21, 36), '+', environ_include_44109, subscript_call_result_44114)
            
            # Getting the type of 'os' (line 21)
            os_44116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 12), 'os')
            # Obtaining the member 'environ' of a type (line 21)
            environ_44117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 12), os_44116, 'environ')
            str_44118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 21, 23), 'str', 'include')
            # Storing an element on a container (line 21)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 21, 12), environ_44117, (str_44118, result_add_44115))

            if more_types_in_union_44108:
                # SSA join for if statement (line 20)
                module_type_store = module_type_store.join_ssa_context()


        
        
        
        # Getting the type of 'platform_bits' (line 22)
        platform_bits_44119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 11), 'platform_bits')
        int_44120 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 28), 'int')
        # Applying the binary operator '==' (line 22)
        result_eq_44121 = python_operator(stypy.reporting.localization.Localization(__file__, 22, 11), '==', platform_bits_44119, int_44120)
        
        # Testing the type of an if condition (line 22)
        if_condition_44122 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 22, 8), result_eq_44121)
        # Assigning a type to the variable 'if_condition_44122' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'if_condition_44122', if_condition_44122)
        # SSA begins for if statement (line 22)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        
        # Getting the type of 'self' (line 25)
        self_44123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'self')
        # Obtaining the member 'compile_options' of a type (line 25)
        compile_options_44124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), self_44123, 'compile_options')
        
        # Obtaining an instance of the builtin type 'list' (line 25)
        list_44125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 36), 'list')
        # Adding type elements to the builtin type 'list' instance (line 25)
        # Adding element type (line 25)
        str_44126 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 25, 37), 'str', '/arch:SSE2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 25, 36), list_44125, str_44126)
        
        # Applying the binary operator '+=' (line 25)
        result_iadd_44127 = python_operator(stypy.reporting.localization.Localization(__file__, 25, 12), '+=', compile_options_44124, list_44125)
        # Getting the type of 'self' (line 25)
        self_44128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 12), 'self')
        # Setting the type of the member 'compile_options' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 12), self_44128, 'compile_options', result_iadd_44127)
        
        
        # Getting the type of 'self' (line 26)
        self_44129 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'self')
        # Obtaining the member 'compile_options_debug' of a type (line 26)
        compile_options_debug_44130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), self_44129, 'compile_options_debug')
        
        # Obtaining an instance of the builtin type 'list' (line 26)
        list_44131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 42), 'list')
        # Adding type elements to the builtin type 'list' instance (line 26)
        # Adding element type (line 26)
        str_44132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 43), 'str', '/arch:SSE2')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 26, 42), list_44131, str_44132)
        
        # Applying the binary operator '+=' (line 26)
        result_iadd_44133 = python_operator(stypy.reporting.localization.Localization(__file__, 26, 12), '+=', compile_options_debug_44130, list_44131)
        # Getting the type of 'self' (line 26)
        self_44134 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 12), 'self')
        # Setting the type of the member 'compile_options_debug' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 12), self_44134, 'compile_options_debug', result_iadd_44133)
        
        # SSA join for if statement (line 22)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'initialize(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_44135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_44135)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize'
        return stypy_return_type_44135


# Assigning a type to the variable 'MSVCCompiler' (line 10)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 10, 0), 'MSVCCompiler', MSVCCompiler)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
