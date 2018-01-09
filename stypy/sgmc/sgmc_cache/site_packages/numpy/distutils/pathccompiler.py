
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from __future__ import division, absolute_import, print_function
2: 
3: from distutils.unixccompiler import UnixCCompiler
4: 
5: class PathScaleCCompiler(UnixCCompiler):
6: 
7:     '''
8:     PathScale compiler compatible with an gcc built Python.
9:     '''
10: 
11:     compiler_type = 'pathcc'
12:     cc_exe = 'pathcc'
13:     cxx_exe = 'pathCC'
14: 
15:     def __init__ (self, verbose=0, dry_run=0, force=0):
16:         UnixCCompiler.__init__ (self, verbose, dry_run, force)
17:         cc_compiler = self.cc_exe
18:         cxx_compiler = self.cxx_exe
19:         self.set_executables(compiler=cc_compiler,
20:                              compiler_so=cc_compiler,
21:                              compiler_cxx=cxx_compiler,
22:                              linker_exe=cc_compiler,
23:                              linker_so=cc_compiler + ' -shared')
24: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from distutils.unixccompiler import UnixCCompiler' statement (line 3)
update_path_to_current_file_folder('C:/Python27/lib/site-packages/numpy/distutils/')
import_45245 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.unixccompiler')

if (type(import_45245) is not StypyTypeError):

    if (import_45245 != 'pyd_module'):
        __import__(import_45245)
        sys_modules_45246 = sys.modules[import_45245]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.unixccompiler', sys_modules_45246.module_type_store, module_type_store, ['UnixCCompiler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_45246, sys_modules_45246.module_type_store, module_type_store)
    else:
        from distutils.unixccompiler import UnixCCompiler

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.unixccompiler', None, module_type_store, ['UnixCCompiler'], [UnixCCompiler])

else:
    # Assigning a type to the variable 'distutils.unixccompiler' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'distutils.unixccompiler', import_45245)

remove_current_file_folder_from_path('C:/Python27/lib/site-packages/numpy/distutils/')

# Declaration of the 'PathScaleCCompiler' class
# Getting the type of 'UnixCCompiler' (line 5)
UnixCCompiler_45247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 25), 'UnixCCompiler')

class PathScaleCCompiler(UnixCCompiler_45247, ):
    str_45248 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', '\n    PathScale compiler compatible with an gcc built Python.\n    ')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        int_45249 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 32), 'int')
        int_45250 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 43), 'int')
        int_45251 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 15, 52), 'int')
        defaults = [int_45249, int_45250, int_45251]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 15, 4, False)
        # Assigning a type to the variable 'self' (line 16)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PathScaleCCompiler.__init__', ['verbose', 'dry_run', 'force'], None, None, defaults, varargs, kwargs)

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

        
        # Call to __init__(...): (line 16)
        # Processing the call arguments (line 16)
        # Getting the type of 'self' (line 16)
        self_45254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 32), 'self', False)
        # Getting the type of 'verbose' (line 16)
        verbose_45255 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 38), 'verbose', False)
        # Getting the type of 'dry_run' (line 16)
        dry_run_45256 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 47), 'dry_run', False)
        # Getting the type of 'force' (line 16)
        force_45257 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 56), 'force', False)
        # Processing the call keyword arguments (line 16)
        kwargs_45258 = {}
        # Getting the type of 'UnixCCompiler' (line 16)
        UnixCCompiler_45252 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 8), 'UnixCCompiler', False)
        # Obtaining the member '__init__' of a type (line 16)
        init___45253 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 16, 8), UnixCCompiler_45252, '__init__')
        # Calling __init__(args, kwargs) (line 16)
        init___call_result_45259 = invoke(stypy.reporting.localization.Localization(__file__, 16, 8), init___45253, *[self_45254, verbose_45255, dry_run_45256, force_45257], **kwargs_45258)
        
        
        # Assigning a Attribute to a Name (line 17):
        # Getting the type of 'self' (line 17)
        self_45260 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 22), 'self')
        # Obtaining the member 'cc_exe' of a type (line 17)
        cc_exe_45261 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 17, 22), self_45260, 'cc_exe')
        # Assigning a type to the variable 'cc_compiler' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 8), 'cc_compiler', cc_exe_45261)
        
        # Assigning a Attribute to a Name (line 18):
        # Getting the type of 'self' (line 18)
        self_45262 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 23), 'self')
        # Obtaining the member 'cxx_exe' of a type (line 18)
        cxx_exe_45263 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 23), self_45262, 'cxx_exe')
        # Assigning a type to the variable 'cxx_compiler' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'cxx_compiler', cxx_exe_45263)
        
        # Call to set_executables(...): (line 19)
        # Processing the call keyword arguments (line 19)
        # Getting the type of 'cc_compiler' (line 19)
        cc_compiler_45266 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 38), 'cc_compiler', False)
        keyword_45267 = cc_compiler_45266
        # Getting the type of 'cc_compiler' (line 20)
        cc_compiler_45268 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 41), 'cc_compiler', False)
        keyword_45269 = cc_compiler_45268
        # Getting the type of 'cxx_compiler' (line 21)
        cxx_compiler_45270 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 42), 'cxx_compiler', False)
        keyword_45271 = cxx_compiler_45270
        # Getting the type of 'cc_compiler' (line 22)
        cc_compiler_45272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 40), 'cc_compiler', False)
        keyword_45273 = cc_compiler_45272
        # Getting the type of 'cc_compiler' (line 23)
        cc_compiler_45274 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 39), 'cc_compiler', False)
        str_45275 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 23, 53), 'str', ' -shared')
        # Applying the binary operator '+' (line 23)
        result_add_45276 = python_operator(stypy.reporting.localization.Localization(__file__, 23, 39), '+', cc_compiler_45274, str_45275)
        
        keyword_45277 = result_add_45276
        kwargs_45278 = {'compiler_cxx': keyword_45271, 'linker_exe': keyword_45273, 'compiler_so': keyword_45269, 'linker_so': keyword_45277, 'compiler': keyword_45267}
        # Getting the type of 'self' (line 19)
        self_45264 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 8), 'self', False)
        # Obtaining the member 'set_executables' of a type (line 19)
        set_executables_45265 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 8), self_45264, 'set_executables')
        # Calling set_executables(args, kwargs) (line 19)
        set_executables_call_result_45279 = invoke(stypy.reporting.localization.Localization(__file__, 19, 8), set_executables_45265, *[], **kwargs_45278)
        
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'PathScaleCCompiler' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'PathScaleCCompiler', PathScaleCCompiler)

# Assigning a Str to a Name (line 11):
str_45280 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 20), 'str', 'pathcc')
# Getting the type of 'PathScaleCCompiler'
PathScaleCCompiler_45281 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleCCompiler')
# Setting the type of the member 'compiler_type' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleCCompiler_45281, 'compiler_type', str_45280)

# Assigning a Str to a Name (line 12):
str_45282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 12, 13), 'str', 'pathcc')
# Getting the type of 'PathScaleCCompiler'
PathScaleCCompiler_45283 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleCCompiler')
# Setting the type of the member 'cc_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleCCompiler_45283, 'cc_exe', str_45282)

# Assigning a Str to a Name (line 13):
str_45284 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 13, 14), 'str', 'pathCC')
# Getting the type of 'PathScaleCCompiler'
PathScaleCCompiler_45285 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'PathScaleCCompiler')
# Setting the type of the member 'cxx_exe' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), PathScaleCCompiler_45285, 'cxx_exe', str_45284)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
