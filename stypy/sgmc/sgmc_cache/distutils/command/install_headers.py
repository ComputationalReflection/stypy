
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: '''distutils.command.install_headers
2: 
3: Implements the Distutils 'install_headers' command, to install C/C++ header
4: files to the Python include directory.'''
5: 
6: __revision__ = "$Id$"
7: 
8: from distutils.core import Command
9: 
10: 
11: # XXX force is never used
12: class install_headers(Command):
13: 
14:     description = "install C/C++ header files"
15: 
16:     user_options = [('install-dir=', 'd',
17:                      "directory to install header files to"),
18:                     ('force', 'f',
19:                      "force installation (overwrite existing files)"),
20:                    ]
21: 
22:     boolean_options = ['force']
23: 
24:     def initialize_options(self):
25:         self.install_dir = None
26:         self.force = 0
27:         self.outfiles = []
28: 
29:     def finalize_options(self):
30:         self.set_undefined_options('install',
31:                                    ('install_headers', 'install_dir'),
32:                                    ('force', 'force'))
33: 
34: 
35:     def run(self):
36:         headers = self.distribution.headers
37:         if not headers:
38:             return
39: 
40:         self.mkpath(self.install_dir)
41:         for header in headers:
42:             (out, _) = self.copy_file(header, self.install_dir)
43:             self.outfiles.append(out)
44: 
45:     def get_inputs(self):
46:         return self.distribution.headers or []
47: 
48:     def get_outputs(self):
49:         return self.outfiles
50: 
51: # class install_headers
52: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

str_24053 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 4, (-1)), 'str', "distutils.command.install_headers\n\nImplements the Distutils 'install_headers' command, to install C/C++ header\nfiles to the Python include directory.")

# Assigning a Str to a Name (line 6):

# Assigning a Str to a Name (line 6):
str_24054 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 6, 15), 'str', '$Id$')
# Assigning a type to the variable '__revision__' (line 6)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), '__revision__', str_24054)
stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from distutils.core import Command' statement (line 8)
update_path_to_current_file_folder('C:/Python27/lib/distutils/command/')
import_24055 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.core')

if (type(import_24055) is not StypyTypeError):

    if (import_24055 != 'pyd_module'):
        __import__(import_24055)
        sys_modules_24056 = sys.modules[import_24055]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.core', sys_modules_24056.module_type_store, module_type_store, ['Command'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_24056, sys_modules_24056.module_type_store, module_type_store)
    else:
        from distutils.core import Command

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.core', None, module_type_store, ['Command'], [Command])

else:
    # Assigning a type to the variable 'distutils.core' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'distutils.core', import_24055)

remove_current_file_folder_from_path('C:/Python27/lib/distutils/command/')

# Declaration of the 'install_headers' class
# Getting the type of 'Command' (line 12)
Command_24057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 12, 22), 'Command')

class install_headers(Command_24057, ):
    
    # Assigning a Str to a Name (line 14):
    
    # Assigning a List to a Name (line 16):
    
    # Assigning a List to a Name (line 22):

    @norecursion
    def initialize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'initialize_options'
        module_type_store = module_type_store.open_function_context('initialize_options', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_headers.initialize_options.__dict__.__setitem__('stypy_localization', localization)
        install_headers.initialize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_headers.initialize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_headers.initialize_options.__dict__.__setitem__('stypy_function_name', 'install_headers.initialize_options')
        install_headers.initialize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_headers.initialize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_headers.initialize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_headers.initialize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_headers.initialize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_headers.initialize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_headers.initialize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_headers.initialize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Name to a Attribute (line 25):
        
        # Assigning a Name to a Attribute (line 25):
        # Getting the type of 'None' (line 25)
        None_24058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 27), 'None')
        # Getting the type of 'self' (line 25)
        self_24059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'self')
        # Setting the type of the member 'install_dir' of a type (line 25)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 25, 8), self_24059, 'install_dir', None_24058)
        
        # Assigning a Num to a Attribute (line 26):
        
        # Assigning a Num to a Attribute (line 26):
        int_24060 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, 21), 'int')
        # Getting the type of 'self' (line 26)
        self_24061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 26, 8), 'self')
        # Setting the type of the member 'force' of a type (line 26)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 26, 8), self_24061, 'force', int_24060)
        
        # Assigning a List to a Attribute (line 27):
        
        # Assigning a List to a Attribute (line 27):
        
        # Obtaining an instance of the builtin type 'list' (line 27)
        list_24062 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 27, 24), 'list')
        # Adding type elements to the builtin type 'list' instance (line 27)
        
        # Getting the type of 'self' (line 27)
        self_24063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'self')
        # Setting the type of the member 'outfiles' of a type (line 27)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 27, 8), self_24063, 'outfiles', list_24062)
        
        # ################# End of 'initialize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'initialize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_24064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24064)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'initialize_options'
        return stypy_return_type_24064


    @norecursion
    def finalize_options(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'finalize_options'
        module_type_store = module_type_store.open_function_context('finalize_options', 29, 4, False)
        # Assigning a type to the variable 'self' (line 30)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 30, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_headers.finalize_options.__dict__.__setitem__('stypy_localization', localization)
        install_headers.finalize_options.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_headers.finalize_options.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_headers.finalize_options.__dict__.__setitem__('stypy_function_name', 'install_headers.finalize_options')
        install_headers.finalize_options.__dict__.__setitem__('stypy_param_names_list', [])
        install_headers.finalize_options.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_headers.finalize_options.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_headers.finalize_options.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_headers.finalize_options.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_headers.finalize_options.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_headers.finalize_options.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_headers.finalize_options', [], None, None, defaults, varargs, kwargs)

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

        
        # Call to set_undefined_options(...): (line 30)
        # Processing the call arguments (line 30)
        str_24067 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, 35), 'str', 'install')
        
        # Obtaining an instance of the builtin type 'tuple' (line 31)
        tuple_24068 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 31)
        # Adding element type (line 31)
        str_24069 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 36), 'str', 'install_headers')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 36), tuple_24068, str_24069)
        # Adding element type (line 31)
        str_24070 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 31, 55), 'str', 'install_dir')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 31, 36), tuple_24068, str_24070)
        
        
        # Obtaining an instance of the builtin type 'tuple' (line 32)
        tuple_24071 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 32)
        # Adding element type (line 32)
        str_24072 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 36), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 36), tuple_24071, str_24072)
        # Adding element type (line 32)
        str_24073 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, 45), 'str', 'force')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 32, 36), tuple_24071, str_24073)
        
        # Processing the call keyword arguments (line 30)
        kwargs_24074 = {}
        # Getting the type of 'self' (line 30)
        self_24065 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 8), 'self', False)
        # Obtaining the member 'set_undefined_options' of a type (line 30)
        set_undefined_options_24066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 30, 8), self_24065, 'set_undefined_options')
        # Calling set_undefined_options(args, kwargs) (line 30)
        set_undefined_options_call_result_24075 = invoke(stypy.reporting.localization.Localization(__file__, 30, 8), set_undefined_options_24066, *[str_24067, tuple_24068, tuple_24071], **kwargs_24074)
        
        
        # ################# End of 'finalize_options(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'finalize_options' in the type store
        # Getting the type of 'stypy_return_type' (line 29)
        stypy_return_type_24076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24076)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'finalize_options'
        return stypy_return_type_24076


    @norecursion
    def run(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'run'
        module_type_store = module_type_store.open_function_context('run', 35, 4, False)
        # Assigning a type to the variable 'self' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_headers.run.__dict__.__setitem__('stypy_localization', localization)
        install_headers.run.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_headers.run.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_headers.run.__dict__.__setitem__('stypy_function_name', 'install_headers.run')
        install_headers.run.__dict__.__setitem__('stypy_param_names_list', [])
        install_headers.run.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_headers.run.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_headers.run.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_headers.run.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_headers.run.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_headers.run.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_headers.run', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a Attribute to a Name (line 36):
        
        # Assigning a Attribute to a Name (line 36):
        # Getting the type of 'self' (line 36)
        self_24077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 18), 'self')
        # Obtaining the member 'distribution' of a type (line 36)
        distribution_24078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 18), self_24077, 'distribution')
        # Obtaining the member 'headers' of a type (line 36)
        headers_24079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 18), distribution_24078, 'headers')
        # Assigning a type to the variable 'headers' (line 36)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 8), 'headers', headers_24079)
        
        
        # Getting the type of 'headers' (line 37)
        headers_24080 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 15), 'headers')
        # Applying the 'not' unary operator (line 37)
        result_not__24081 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 11), 'not', headers_24080)
        
        # Testing the type of an if condition (line 37)
        if_condition_24082 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 37, 8), result_not__24081)
        # Assigning a type to the variable 'if_condition_24082' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'if_condition_24082', if_condition_24082)
        # SSA begins for if statement (line 37)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 12), 'stypy_return_type', types.NoneType)
        # SSA join for if statement (line 37)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # Call to mkpath(...): (line 40)
        # Processing the call arguments (line 40)
        # Getting the type of 'self' (line 40)
        self_24085 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 20), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 40)
        install_dir_24086 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 20), self_24085, 'install_dir')
        # Processing the call keyword arguments (line 40)
        kwargs_24087 = {}
        # Getting the type of 'self' (line 40)
        self_24083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 40, 8), 'self', False)
        # Obtaining the member 'mkpath' of a type (line 40)
        mkpath_24084 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 40, 8), self_24083, 'mkpath')
        # Calling mkpath(args, kwargs) (line 40)
        mkpath_call_result_24088 = invoke(stypy.reporting.localization.Localization(__file__, 40, 8), mkpath_24084, *[install_dir_24086], **kwargs_24087)
        
        
        # Getting the type of 'headers' (line 41)
        headers_24089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 22), 'headers')
        # Testing the type of a for loop iterable (line 41)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 41, 8), headers_24089)
        # Getting the type of the for loop variable (line 41)
        for_loop_var_24090 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 41, 8), headers_24089)
        # Assigning a type to the variable 'header' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'header', for_loop_var_24090)
        # SSA begins for a for statement (line 41)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
        
        # Assigning a Call to a Tuple (line 42):
        
        # Assigning a Subscript to a Name (line 42):
        
        # Obtaining the type of the subscript
        int_24091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 12), 'int')
        
        # Call to copy_file(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'header' (line 42)
        header_24094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'header', False)
        # Getting the type of 'self' (line 42)
        self_24095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 46), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 42)
        install_dir_24096 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 46), self_24095, 'install_dir')
        # Processing the call keyword arguments (line 42)
        kwargs_24097 = {}
        # Getting the type of 'self' (line 42)
        self_24092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 42)
        copy_file_24093 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), self_24092, 'copy_file')
        # Calling copy_file(args, kwargs) (line 42)
        copy_file_call_result_24098 = invoke(stypy.reporting.localization.Localization(__file__, 42, 23), copy_file_24093, *[header_24094, install_dir_24096], **kwargs_24097)
        
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___24099 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), copy_file_call_result_24098, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_24100 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), getitem___24099, int_24091)
        
        # Assigning a type to the variable 'tuple_var_assignment_24051' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'tuple_var_assignment_24051', subscript_call_result_24100)
        
        # Assigning a Subscript to a Name (line 42):
        
        # Obtaining the type of the subscript
        int_24101 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, 12), 'int')
        
        # Call to copy_file(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'header' (line 42)
        header_24104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 38), 'header', False)
        # Getting the type of 'self' (line 42)
        self_24105 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 46), 'self', False)
        # Obtaining the member 'install_dir' of a type (line 42)
        install_dir_24106 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 46), self_24105, 'install_dir')
        # Processing the call keyword arguments (line 42)
        kwargs_24107 = {}
        # Getting the type of 'self' (line 42)
        self_24102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 23), 'self', False)
        # Obtaining the member 'copy_file' of a type (line 42)
        copy_file_24103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 23), self_24102, 'copy_file')
        # Calling copy_file(args, kwargs) (line 42)
        copy_file_call_result_24108 = invoke(stypy.reporting.localization.Localization(__file__, 42, 23), copy_file_24103, *[header_24104, install_dir_24106], **kwargs_24107)
        
        # Obtaining the member '__getitem__' of a type (line 42)
        getitem___24109 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 42, 12), copy_file_call_result_24108, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 42)
        subscript_call_result_24110 = invoke(stypy.reporting.localization.Localization(__file__, 42, 12), getitem___24109, int_24101)
        
        # Assigning a type to the variable 'tuple_var_assignment_24052' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'tuple_var_assignment_24052', subscript_call_result_24110)
        
        # Assigning a Name to a Name (line 42):
        # Getting the type of 'tuple_var_assignment_24051' (line 42)
        tuple_var_assignment_24051_24111 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'tuple_var_assignment_24051')
        # Assigning a type to the variable 'out' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 13), 'out', tuple_var_assignment_24051_24111)
        
        # Assigning a Name to a Name (line 42):
        # Getting the type of 'tuple_var_assignment_24052' (line 42)
        tuple_var_assignment_24052_24112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 12), 'tuple_var_assignment_24052')
        # Assigning a type to the variable '_' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 18), '_', tuple_var_assignment_24052_24112)
        
        # Call to append(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'out' (line 43)
        out_24116 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 33), 'out', False)
        # Processing the call keyword arguments (line 43)
        kwargs_24117 = {}
        # Getting the type of 'self' (line 43)
        self_24113 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 12), 'self', False)
        # Obtaining the member 'outfiles' of a type (line 43)
        outfiles_24114 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), self_24113, 'outfiles')
        # Obtaining the member 'append' of a type (line 43)
        append_24115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 12), outfiles_24114, 'append')
        # Calling append(args, kwargs) (line 43)
        append_call_result_24118 = invoke(stypy.reporting.localization.Localization(__file__, 43, 12), append_24115, *[out_24116], **kwargs_24117)
        
        # SSA join for a for statement
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of 'run(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'run' in the type store
        # Getting the type of 'stypy_return_type' (line 35)
        stypy_return_type_24119 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24119)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'run'
        return stypy_return_type_24119


    @norecursion
    def get_inputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_inputs'
        module_type_store = module_type_store.open_function_context('get_inputs', 45, 4, False)
        # Assigning a type to the variable 'self' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_headers.get_inputs.__dict__.__setitem__('stypy_localization', localization)
        install_headers.get_inputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_headers.get_inputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_headers.get_inputs.__dict__.__setitem__('stypy_function_name', 'install_headers.get_inputs')
        install_headers.get_inputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_headers.get_inputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_headers.get_inputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_headers.get_inputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_headers.get_inputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_headers.get_inputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_headers.get_inputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_headers.get_inputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_inputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_inputs(...)' code ##################

        
        # Evaluating a boolean operation
        # Getting the type of 'self' (line 46)
        self_24120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 15), 'self')
        # Obtaining the member 'distribution' of a type (line 46)
        distribution_24121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), self_24120, 'distribution')
        # Obtaining the member 'headers' of a type (line 46)
        headers_24122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 46, 15), distribution_24121, 'headers')
        
        # Obtaining an instance of the builtin type 'list' (line 46)
        list_24123 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 46, 44), 'list')
        # Adding type elements to the builtin type 'list' instance (line 46)
        
        # Applying the binary operator 'or' (line 46)
        result_or_keyword_24124 = python_operator(stypy.reporting.localization.Localization(__file__, 46, 15), 'or', headers_24122, list_24123)
        
        # Assigning a type to the variable 'stypy_return_type' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'stypy_return_type', result_or_keyword_24124)
        
        # ################# End of 'get_inputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_inputs' in the type store
        # Getting the type of 'stypy_return_type' (line 45)
        stypy_return_type_24125 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24125)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_inputs'
        return stypy_return_type_24125


    @norecursion
    def get_outputs(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_outputs'
        module_type_store = module_type_store.open_function_context('get_outputs', 48, 4, False)
        # Assigning a type to the variable 'self' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        install_headers.get_outputs.__dict__.__setitem__('stypy_localization', localization)
        install_headers.get_outputs.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        install_headers.get_outputs.__dict__.__setitem__('stypy_type_store', module_type_store)
        install_headers.get_outputs.__dict__.__setitem__('stypy_function_name', 'install_headers.get_outputs')
        install_headers.get_outputs.__dict__.__setitem__('stypy_param_names_list', [])
        install_headers.get_outputs.__dict__.__setitem__('stypy_varargs_param_name', None)
        install_headers.get_outputs.__dict__.__setitem__('stypy_kwargs_param_name', None)
        install_headers.get_outputs.__dict__.__setitem__('stypy_call_defaults', defaults)
        install_headers.get_outputs.__dict__.__setitem__('stypy_call_varargs', varargs)
        install_headers.get_outputs.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        install_headers.get_outputs.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_headers.get_outputs', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_outputs', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_outputs(...)' code ##################

        # Getting the type of 'self' (line 49)
        self_24126 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 49, 15), 'self')
        # Obtaining the member 'outfiles' of a type (line 49)
        outfiles_24127 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 49, 15), self_24126, 'outfiles')
        # Assigning a type to the variable 'stypy_return_type' (line 49)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 49, 8), 'stypy_return_type', outfiles_24127)
        
        # ################# End of 'get_outputs(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_outputs' in the type store
        # Getting the type of 'stypy_return_type' (line 48)
        stypy_return_type_24128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_24128)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_outputs'
        return stypy_return_type_24128


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 12, 0, False)
        # Assigning a type to the variable 'self' (line 13)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 13, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'install_headers.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'install_headers' (line 12)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 0), 'install_headers', install_headers)

# Assigning a Str to a Name (line 14):
str_24129 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 14, 18), 'str', 'install C/C++ header files')
# Getting the type of 'install_headers'
install_headers_24130 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_headers')
# Setting the type of the member 'description' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_headers_24130, 'description', str_24129)

# Assigning a List to a Name (line 16):

# Obtaining an instance of the builtin type 'list' (line 16)
list_24131 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 19), 'list')
# Adding type elements to the builtin type 'list' instance (line 16)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'tuple' (line 16)
tuple_24132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 16)
# Adding element type (line 16)
str_24133 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 21), 'str', 'install-dir=')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 21), tuple_24132, str_24133)
# Adding element type (line 16)
str_24134 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 16, 37), 'str', 'd')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 21), tuple_24132, str_24134)
# Adding element type (line 16)
str_24135 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, 21), 'str', 'directory to install header files to')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 21), tuple_24132, str_24135)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 19), list_24131, tuple_24132)
# Adding element type (line 16)

# Obtaining an instance of the builtin type 'tuple' (line 18)
tuple_24136 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'tuple')
# Adding type elements to the builtin type 'tuple' instance (line 18)
# Adding element type (line 18)
str_24137 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 21), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), tuple_24136, str_24137)
# Adding element type (line 18)
str_24138 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, 30), 'str', 'f')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), tuple_24136, str_24138)
# Adding element type (line 18)
str_24139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 19, 21), 'str', 'force installation (overwrite existing files)')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 18, 21), tuple_24136, str_24139)

add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 16, 19), list_24131, tuple_24136)

# Getting the type of 'install_headers'
install_headers_24140 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_headers')
# Setting the type of the member 'user_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_headers_24140, 'user_options', list_24131)

# Assigning a List to a Name (line 22):

# Obtaining an instance of the builtin type 'list' (line 22)
list_24141 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 22), 'list')
# Adding type elements to the builtin type 'list' instance (line 22)
# Adding element type (line 22)
str_24142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, 23), 'str', 'force')
add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 22, 22), list_24141, str_24142)

# Getting the type of 'install_headers'
install_headers_24143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'install_headers')
# Setting the type of the member 'boolean_options' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), install_headers_24143, 'boolean_options', list_24141)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
