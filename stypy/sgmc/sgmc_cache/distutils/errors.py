
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.errors
2: 
3: Provides exceptions used by the Distutils modules.  Note that Distutils
4: modules may raise standard exceptions; in particular, SystemExit is
5: usually raised for errors that are obviously the end-user's fault
6: (eg. bad command-line arguments).
7: 
8: This module is safe to use in "from ... import *" mode; it only exports
9: symbols whose names start with "Distutils" and end with "Error".'''
10: 
11: __revision__ = "$Id$"
12: 
13: class DistutilsError(Exception):
14:     '''The root of all Distutils evil.'''
15: 
16: class DistutilsModuleError(DistutilsError):
17:     '''Unable to load an expected module, or to find an expected class
18:     within some module (in particular, command modules and classes).'''
19: 
20: class DistutilsClassError(DistutilsError):
21:     '''Some command class (or possibly distribution class, if anyone
22:     feels a need to subclass Distribution) is found not to be holding
23:     up its end of the bargain, ie. implementing some part of the
24:     "command "interface.'''
25: 
26: class DistutilsGetoptError(DistutilsError):
27:     '''The option table provided to 'fancy_getopt()' is bogus.'''
28: 
29: class DistutilsArgError(DistutilsError):
30:     '''Raised by fancy_getopt in response to getopt.error -- ie. an
31:     error in the command line usage.'''
32: 
33: class DistutilsFileError(DistutilsError):
34:     '''Any problems in the filesystem: expected file not found, etc.
35:     Typically this is for problems that we detect before IOError or
36:     OSError could be raised.'''
37: 
38: class DistutilsOptionError(DistutilsError):
39:     '''Syntactic/semantic errors in command options, such as use of
40:     mutually conflicting options, or inconsistent options,
41:     badly-spelled values, etc.  No distinction is made between option
42:     values originating in the setup script, the command line, config
43:     files, or what-have-you -- but if we *know* something originated in
44:     the setup script, we'll raise DistutilsSetupError instead.'''
45: 
46: class DistutilsSetupError(DistutilsError):
47:     '''For errors that can be definitely blamed on the setup script,
48:     such as invalid keyword arguments to 'setup()'.'''
49: 
50: class DistutilsPlatformError(DistutilsError):
51:     '''We don't know how to do something on the current platform (but
52:     we do know how to do it on some platform) -- eg. trying to compile
53:     C files on a platform not supported by a CCompiler subclass.'''
54: 
55: class DistutilsExecError(DistutilsError):
56:     '''Any problems executing an external program (such as the C
57:     compiler, when compiling C files).'''
58: 
59: class DistutilsInternalError(DistutilsError):
60:     '''Internal inconsistencies or impossibilities (obviously, this
61:     should never be seen if the code is working!).'''
62: 
63: class DistutilsTemplateError(DistutilsError):
64:     '''Syntax error in a file list template.'''
65: 
66: class DistutilsByteCompileError(DistutilsError):
67:     '''Byte compile error.'''
68: 
69: # Exception classes used by the CCompiler implementation classes
70: class CCompilerError(Exception):
71:     '''Some compile/link operation failed.'''
72: 
73: class PreprocessError(CCompilerError):
74:     '''Failure to preprocess one or more C/C++ files.'''
75: 
76: class CompileError(CCompilerError):
77:     '''Failure to compile one or more C/C++ source files.'''
78: 
79: class LibError(CCompilerError):
80:     '''Failure to create a static library from one or more C/C++ object
81:     files.'''
82: 
83: class LinkError(CCompilerError):
84:     '''Failure to link one or more C/C++ object files into an executable
85:     or shared library file.'''
86: 
87: class UnknownFileError(CCompilerError):
88:     '''Attempt to process an unknown file type.'''
89: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_3677 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 9, (-1)), 'str', 'distutils.errors\n\nProvides exceptions used by the Distutils modules.  Note that Distutils\nmodules may raise standard exceptions; in particular, SystemExit is\nusually raised for errors that are obviously the end-user\'s fault\n(eg. bad command-line arguments).\n\nThis module is safe to use in "from ... import *" mode; it only exports\nsymbols whose names start with "Distutils" and end with "Error".')

# Assigning a Str to a Name (line 11):
str_3678 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), '__revision__', str_3678)
# Declaration of the 'DistutilsError' class
# Getting the type of 'Exception' (line 13)
Exception_3679 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 13, 21), 'Exception')

class DistutilsError(Exception_3679, ):
    str_3680 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 4), 'str', 'The root of all Distutils evil.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 13, 0, False)
        # Assigning a type to the variable 'self' (line 14)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 14, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsError' (line 13)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'DistutilsError', DistutilsError)
# Declaration of the 'DistutilsModuleError' class
# Getting the type of 'DistutilsError' (line 16)
DistutilsError_3681 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 16, 27), 'DistutilsError')

class DistutilsModuleError(DistutilsError_3681, ):
    str_3682 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', 'Unable to load an expected module, or to find an expected class\n    within some module (in particular, command modules and classes).')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 16, 0, False)
        # Assigning a type to the variable 'self' (line 17)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 17, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsModuleError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsModuleError' (line 16)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 16, 0), 'DistutilsModuleError', DistutilsModuleError)
# Declaration of the 'DistutilsClassError' class
# Getting the type of 'DistutilsError' (line 20)
DistutilsError_3683 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 26), 'DistutilsError')

class DistutilsClassError(DistutilsError_3683, ):
    str_3684 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 24, (-1)), 'str', 'Some command class (or possibly distribution class, if anyone\n    feels a need to subclass Distribution) is found not to be holding\n    up its end of the bargain, ie. implementing some part of the\n    "command "interface.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 20, 0, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsClassError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsClassError' (line 20)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 20, 0), 'DistutilsClassError', DistutilsClassError)
# Declaration of the 'DistutilsGetoptError' class
# Getting the type of 'DistutilsError' (line 26)
DistutilsError_3685 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 27), 'DistutilsError')

class DistutilsGetoptError(DistutilsError_3685, ):
    str_3686 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 4), 'str', "The option table provided to 'fancy_getopt()' is bogus.")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 26, 0, False)
        # Assigning a type to the variable 'self' (line 27)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsGetoptError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsGetoptError' (line 26)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 0), 'DistutilsGetoptError', DistutilsGetoptError)
# Declaration of the 'DistutilsArgError' class
# Getting the type of 'DistutilsError' (line 29)
DistutilsError_3687 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 24), 'DistutilsError')

class DistutilsArgError(DistutilsError_3687, ):
    str_3688 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, (-1)), 'str', 'Raised by fancy_getopt in response to getopt.error -- ie. an\n    error in the command line usage.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 29, 0, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsArgError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsArgError' (line 29)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 0), 'DistutilsArgError', DistutilsArgError)
# Declaration of the 'DistutilsFileError' class
# Getting the type of 'DistutilsError' (line 33)
DistutilsError_3689 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 25), 'DistutilsError')

class DistutilsFileError(DistutilsError_3689, ):
    str_3690 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 36, (-1)), 'str', 'Any problems in the filesystem: expected file not found, etc.\n    Typically this is for problems that we detect before IOError or\n    OSError could be raised.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 33, 0, False)
        # Assigning a type to the variable 'self' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsFileError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsFileError' (line 33)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 0), 'DistutilsFileError', DistutilsFileError)
# Declaration of the 'DistutilsOptionError' class
# Getting the type of 'DistutilsError' (line 38)
DistutilsError_3691 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 27), 'DistutilsError')

class DistutilsOptionError(DistutilsError_3691, ):
    str_3692 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, (-1)), 'str', "Syntactic/semantic errors in command options, such as use of\n    mutually conflicting options, or inconsistent options,\n    badly-spelled values, etc.  No distinction is made between option\n    values originating in the setup script, the command line, config\n    files, or what-have-you -- but if we *know* something originated in\n    the setup script, we'll raise DistutilsSetupError instead.")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 38, 0, False)
        # Assigning a type to the variable 'self' (line 39)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsOptionError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsOptionError' (line 38)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 0), 'DistutilsOptionError', DistutilsOptionError)
# Declaration of the 'DistutilsSetupError' class
# Getting the type of 'DistutilsError' (line 46)
DistutilsError_3693 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 26), 'DistutilsError')

class DistutilsSetupError(DistutilsError_3693, ):
    str_3694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 48, (-1)), 'str', "For errors that can be definitely blamed on the setup script,\n    such as invalid keyword arguments to 'setup()'.")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 46, 0, False)
        # Assigning a type to the variable 'self' (line 47)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 47, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsSetupError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsSetupError' (line 46)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 0), 'DistutilsSetupError', DistutilsSetupError)
# Declaration of the 'DistutilsPlatformError' class
# Getting the type of 'DistutilsError' (line 50)
DistutilsError_3695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 29), 'DistutilsError')

class DistutilsPlatformError(DistutilsError_3695, ):
    str_3696 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 53, (-1)), 'str', "We don't know how to do something on the current platform (but\n    we do know how to do it on some platform) -- eg. trying to compile\n    C files on a platform not supported by a CCompiler subclass.")

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 50, 0, False)
        # Assigning a type to the variable 'self' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsPlatformError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsPlatformError' (line 50)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 50, 0), 'DistutilsPlatformError', DistutilsPlatformError)
# Declaration of the 'DistutilsExecError' class
# Getting the type of 'DistutilsError' (line 55)
DistutilsError_3697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 25), 'DistutilsError')

class DistutilsExecError(DistutilsError_3697, ):
    str_3698 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 57, (-1)), 'str', 'Any problems executing an external program (such as the C\n    compiler, when compiling C files).')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 55, 0, False)
        # Assigning a type to the variable 'self' (line 56)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsExecError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsExecError' (line 55)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 55, 0), 'DistutilsExecError', DistutilsExecError)
# Declaration of the 'DistutilsInternalError' class
# Getting the type of 'DistutilsError' (line 59)
DistutilsError_3699 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 29), 'DistutilsError')

class DistutilsInternalError(DistutilsError_3699, ):
    str_3700 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 61, (-1)), 'str', 'Internal inconsistencies or impossibilities (obviously, this\n    should never be seen if the code is working!).')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 59, 0, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsInternalError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsInternalError' (line 59)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 0), 'DistutilsInternalError', DistutilsInternalError)
# Declaration of the 'DistutilsTemplateError' class
# Getting the type of 'DistutilsError' (line 63)
DistutilsError_3701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 29), 'DistutilsError')

class DistutilsTemplateError(DistutilsError_3701, ):
    str_3702 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 64, 4), 'str', 'Syntax error in a file list template.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 63, 0, False)
        # Assigning a type to the variable 'self' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsTemplateError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsTemplateError' (line 63)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 0), 'DistutilsTemplateError', DistutilsTemplateError)
# Declaration of the 'DistutilsByteCompileError' class
# Getting the type of 'DistutilsError' (line 66)
DistutilsError_3703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 32), 'DistutilsError')

class DistutilsByteCompileError(DistutilsError_3703, ):
    str_3704 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 67, 4), 'str', 'Byte compile error.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 66, 0, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'DistutilsByteCompileError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'DistutilsByteCompileError' (line 66)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 0), 'DistutilsByteCompileError', DistutilsByteCompileError)
# Declaration of the 'CCompilerError' class
# Getting the type of 'Exception' (line 70)
Exception_3705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 21), 'Exception')

class CCompilerError(Exception_3705, ):
    str_3706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 71, 4), 'str', 'Some compile/link operation failed.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 70, 0, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CCompilerError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CCompilerError' (line 70)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 0), 'CCompilerError', CCompilerError)
# Declaration of the 'PreprocessError' class
# Getting the type of 'CCompilerError' (line 73)
CCompilerError_3707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 22), 'CCompilerError')

class PreprocessError(CCompilerError_3707, ):
    str_3708 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, 4), 'str', 'Failure to preprocess one or more C/C++ files.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 73, 0, False)
        # Assigning a type to the variable 'self' (line 74)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'PreprocessError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'PreprocessError' (line 73)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 0), 'PreprocessError', PreprocessError)
# Declaration of the 'CompileError' class
# Getting the type of 'CCompilerError' (line 76)
CCompilerError_3709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 19), 'CCompilerError')

class CompileError(CCompilerError_3709, ):
    str_3710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 4), 'str', 'Failure to compile one or more C/C++ source files.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 76, 0, False)
        # Assigning a type to the variable 'self' (line 77)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'CompileError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'CompileError' (line 76)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 0), 'CompileError', CompileError)
# Declaration of the 'LibError' class
# Getting the type of 'CCompilerError' (line 79)
CCompilerError_3711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 15), 'CCompilerError')

class LibError(CCompilerError_3711, ):
    str_3712 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 81, (-1)), 'str', 'Failure to create a static library from one or more C/C++ object\n    files.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 79, 0, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LibError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LibError' (line 79)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 79, 0), 'LibError', LibError)
# Declaration of the 'LinkError' class
# Getting the type of 'CCompilerError' (line 83)
CCompilerError_3713 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 16), 'CCompilerError')

class LinkError(CCompilerError_3713, ):
    str_3714 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 85, (-1)), 'str', 'Failure to link one or more C/C++ object files into an executable\n    or shared library file.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 83, 0, False)
        # Assigning a type to the variable 'self' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'LinkError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'LinkError' (line 83)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 0), 'LinkError', LinkError)
# Declaration of the 'UnknownFileError' class
# Getting the type of 'CCompilerError' (line 87)
CCompilerError_3715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 23), 'CCompilerError')

class UnknownFileError(CCompilerError_3715, ):
    str_3716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 88, 4), 'str', 'Attempt to process an unknown file type.')

    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 87, 0, False)
        # Assigning a type to the variable 'self' (line 88)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 88, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UnknownFileError.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'UnknownFileError' (line 87)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 0), 'UnknownFileError', UnknownFileError)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
