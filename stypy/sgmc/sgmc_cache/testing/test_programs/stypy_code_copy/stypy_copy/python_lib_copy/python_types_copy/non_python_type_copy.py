
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import type_copy
2: #from stypy_copy.errors_copy.type_error_copy import TypeError
3: 
4: 
5: class NonPythonType(type_copy.Type):
6:     '''
7:     Types store common Python language types. This subclass is used to be the parent of some types used by stypy
8:     that are not Python types (such as DynamicType), but are needed for modeling some operations. Much of this type
9:     methods are overriden to return errors if called, as non-python types are not meant to be called on normal
10:     code execution
11:     '''
12:     # #################### STORED PYTHON ENTITY (CLASS, METHOD...) AND PYTHON TYPE/INSTANCE OF THE ENTITY ############
13: 
14:     def get_python_entity(self):
15:         return self
16: 
17:     def get_python_type(self):
18:         return self
19: 
20:     def get_instance(self):
21:         return None
22: 
23:     # ############################## MEMBER TYPE GET / SET ###############################
24: 
25:     def get_type_of_member(self, localization, member_name):
26:         '''
27:         Returns an error if called
28:         '''
29:         return TypeError(localization, "Cannot get the type of a member over a {0}".format(self.__class__.__name__))
30: 
31:     def set_type_of_member(self, localization, member_name, member_value):
32:         '''
33:         Returns an error if called
34:         '''
35:         return TypeError(localization, "Cannot set the type of a member over a {0}".format(self.__class__.__name__))
36: 
37:     # ############################## MEMBER INVOKATION ###############################
38: 
39:     def invoke(self, localization, *args, **kwargs):
40:         '''
41:         Returns an error if called
42:         '''
43:         return TypeError(localization, "Cannot invoke a method over a {0}".format(self.__class__.__name__))
44: 
45:     # ############################## STRUCTURAL REFLECTION ###############################
46: 
47:     def delete_member(self, localization, member):
48:         '''
49:         Returns an error if called
50:         '''
51:         return TypeError(localization, "Cannot delete a member of a {0}".format(self.__class__.__name__))
52: 
53:     def supports_structural_reflection(self):
54:         '''
55:         Returns an error if called
56:         '''
57:         return False
58: 
59:     def change_type(self, localization, new_type):
60:         '''
61:         Returns an error if called
62:         '''
63:         return TypeError(localization, "Cannot change the type of a {0}".format(self.__class__.__name__))
64: 
65:     def change_base_types(self, localization, new_types):
66:         '''
67:         Returns an error if called
68:         '''
69:         return TypeError(localization, "Cannot change the base types of a {0}".format(self.__class__.__name__))
70: 
71:     def add_base_types(self, localization, new_types):
72:         '''
73:         Returns an error if called
74:         '''
75:         self.change_base_types(localization, new_types)
76: 
77:     # ############################## TYPE CLONING ###############################
78: 
79:     def clone(self):
80:         return self
81: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import type_copy' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/')
import_8702 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_copy')

if (type(import_8702) is not StypyTypeError):

    if (import_8702 != 'pyd_module'):
        __import__(import_8702)
        sys_modules_8703 = sys.modules[import_8702]
        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_copy', sys_modules_8703.module_type_store, module_type_store)
    else:
        import type_copy

        import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_copy', type_copy, module_type_store)

else:
    # Assigning a type to the variable 'type_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_copy', import_8702)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/python_types_copy/')

# Declaration of the 'NonPythonType' class
# Getting the type of 'type_copy' (line 5)
type_copy_8704 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 5, 20), 'type_copy')
# Obtaining the member 'Type' of a type (line 5)
Type_8705 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 5, 20), type_copy_8704, 'Type')

class NonPythonType(Type_8705, ):
    str_8706 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 11, (-1)), 'str', '\n    Types store common Python language types. This subclass is used to be the parent of some types used by stypy\n    that are not Python types (such as DynamicType), but are needed for modeling some operations. Much of this type\n    methods are overriden to return errors if called, as non-python types are not meant to be called on normal\n    code execution\n    ')

    @norecursion
    def get_python_entity(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_python_entity'
        module_type_store = module_type_store.open_function_context('get_python_entity', 14, 4, False)
        # Assigning a type to the variable 'self' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_function_name', 'NonPythonType.get_python_entity')
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_param_names_list', [])
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.get_python_entity.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.get_python_entity', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_python_entity', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_python_entity(...)' code ##################

        # Getting the type of 'self' (line 15)
        self_8707 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 15, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 15)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 15, 8), 'stypy_return_type', self_8707)
        
        # ################# End of 'get_python_entity(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_entity' in the type store
        # Getting the type of 'stypy_return_type' (line 14)
        stypy_return_type_8708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 14, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8708)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_entity'
        return stypy_return_type_8708


    @norecursion
    def get_python_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_python_type'
        module_type_store = module_type_store.open_function_context('get_python_type', 17, 4, False)
        # Assigning a type to the variable 'self' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_function_name', 'NonPythonType.get_python_type')
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_param_names_list', [])
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.get_python_type.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.get_python_type', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_python_type', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_python_type(...)' code ##################

        # Getting the type of 'self' (line 18)
        self_8709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 18)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'stypy_return_type', self_8709)
        
        # ################# End of 'get_python_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_python_type' in the type store
        # Getting the type of 'stypy_return_type' (line 17)
        stypy_return_type_8710 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 17, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8710)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_python_type'
        return stypy_return_type_8710


    @norecursion
    def get_instance(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_instance'
        module_type_store = module_type_store.open_function_context('get_instance', 20, 4, False)
        # Assigning a type to the variable 'self' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.get_instance.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.get_instance.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.get_instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.get_instance.__dict__.__setitem__('stypy_function_name', 'NonPythonType.get_instance')
        NonPythonType.get_instance.__dict__.__setitem__('stypy_param_names_list', [])
        NonPythonType.get_instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.get_instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.get_instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.get_instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.get_instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.get_instance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.get_instance', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_instance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_instance(...)' code ##################

        # Getting the type of 'None' (line 21)
        None_8711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 15), 'None')
        # Assigning a type to the variable 'stypy_return_type' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'stypy_return_type', None_8711)
        
        # ################# End of 'get_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_8712 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8712)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_instance'
        return stypy_return_type_8712


    @norecursion
    def get_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'get_type_of_member'
        module_type_store = module_type_store.open_function_context('get_type_of_member', 25, 4, False)
        # Assigning a type to the variable 'self' (line 26)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 26, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_function_name', 'NonPythonType.get_type_of_member')
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name'])
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.get_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.get_type_of_member', ['localization', 'member_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'get_type_of_member', localization, ['localization', 'member_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'get_type_of_member(...)' code ##################

        str_8713 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 28, (-1)), 'str', '\n        Returns an error if called\n        ')
        
        # Call to TypeError(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'localization' (line 29)
        localization_8715 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 25), 'localization', False)
        
        # Call to format(...): (line 29)
        # Processing the call arguments (line 29)
        # Getting the type of 'self' (line 29)
        self_8718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 91), 'self', False)
        # Obtaining the member '__class__' of a type (line 29)
        class___8719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 91), self_8718, '__class__')
        # Obtaining the member '__name__' of a type (line 29)
        name___8720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 91), class___8719, '__name__')
        # Processing the call keyword arguments (line 29)
        kwargs_8721 = {}
        str_8716 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 29, 39), 'str', 'Cannot get the type of a member over a {0}')
        # Obtaining the member 'format' of a type (line 29)
        format_8717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 39), str_8716, 'format')
        # Calling format(args, kwargs) (line 29)
        format_call_result_8722 = invoke(stypy.reporting.localization.Localization(__file__, 29, 39), format_8717, *[name___8720], **kwargs_8721)
        
        # Processing the call keyword arguments (line 29)
        kwargs_8723 = {}
        # Getting the type of 'TypeError' (line 29)
        TypeError_8714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 29)
        TypeError_call_result_8724 = invoke(stypy.reporting.localization.Localization(__file__, 29, 15), TypeError_8714, *[localization_8715, format_call_result_8722], **kwargs_8723)
        
        # Assigning a type to the variable 'stypy_return_type' (line 29)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 8), 'stypy_return_type', TypeError_call_result_8724)
        
        # ################# End of 'get_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'get_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_8725 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8725)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'get_type_of_member'
        return stypy_return_type_8725


    @norecursion
    def set_type_of_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'set_type_of_member'
        module_type_store = module_type_store.open_function_context('set_type_of_member', 31, 4, False)
        # Assigning a type to the variable 'self' (line 32)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 32, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_function_name', 'NonPythonType.set_type_of_member')
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member_name', 'member_value'])
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.set_type_of_member.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.set_type_of_member', ['localization', 'member_name', 'member_value'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'set_type_of_member', localization, ['localization', 'member_name', 'member_value'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'set_type_of_member(...)' code ##################

        str_8726 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, (-1)), 'str', '\n        Returns an error if called\n        ')
        
        # Call to TypeError(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'localization' (line 35)
        localization_8728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 25), 'localization', False)
        
        # Call to format(...): (line 35)
        # Processing the call arguments (line 35)
        # Getting the type of 'self' (line 35)
        self_8731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 91), 'self', False)
        # Obtaining the member '__class__' of a type (line 35)
        class___8732 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 91), self_8731, '__class__')
        # Obtaining the member '__name__' of a type (line 35)
        name___8733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 91), class___8732, '__name__')
        # Processing the call keyword arguments (line 35)
        kwargs_8734 = {}
        str_8729 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 39), 'str', 'Cannot set the type of a member over a {0}')
        # Obtaining the member 'format' of a type (line 35)
        format_8730 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 39), str_8729, 'format')
        # Calling format(args, kwargs) (line 35)
        format_call_result_8735 = invoke(stypy.reporting.localization.Localization(__file__, 35, 39), format_8730, *[name___8733], **kwargs_8734)
        
        # Processing the call keyword arguments (line 35)
        kwargs_8736 = {}
        # Getting the type of 'TypeError' (line 35)
        TypeError_8727 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 35)
        TypeError_call_result_8737 = invoke(stypy.reporting.localization.Localization(__file__, 35, 15), TypeError_8727, *[localization_8728, format_call_result_8735], **kwargs_8736)
        
        # Assigning a type to the variable 'stypy_return_type' (line 35)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 8), 'stypy_return_type', TypeError_call_result_8737)
        
        # ################# End of 'set_type_of_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'set_type_of_member' in the type store
        # Getting the type of 'stypy_return_type' (line 31)
        stypy_return_type_8738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8738)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'set_type_of_member'
        return stypy_return_type_8738


    @norecursion
    def invoke(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'invoke'
        module_type_store = module_type_store.open_function_context('invoke', 39, 4, False)
        # Assigning a type to the variable 'self' (line 40)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 40, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.invoke.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.invoke.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.invoke.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.invoke.__dict__.__setitem__('stypy_function_name', 'NonPythonType.invoke')
        NonPythonType.invoke.__dict__.__setitem__('stypy_param_names_list', ['localization'])
        NonPythonType.invoke.__dict__.__setitem__('stypy_varargs_param_name', 'args')
        NonPythonType.invoke.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs')
        NonPythonType.invoke.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.invoke.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.invoke.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.invoke.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.invoke', ['localization'], 'args', 'kwargs', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'invoke', localization, ['localization'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'invoke(...)' code ##################

        str_8739 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 42, (-1)), 'str', '\n        Returns an error if called\n        ')
        
        # Call to TypeError(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'localization' (line 43)
        localization_8741 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 25), 'localization', False)
        
        # Call to format(...): (line 43)
        # Processing the call arguments (line 43)
        # Getting the type of 'self' (line 43)
        self_8744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 82), 'self', False)
        # Obtaining the member '__class__' of a type (line 43)
        class___8745 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 82), self_8744, '__class__')
        # Obtaining the member '__name__' of a type (line 43)
        name___8746 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 82), class___8745, '__name__')
        # Processing the call keyword arguments (line 43)
        kwargs_8747 = {}
        str_8742 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 43, 39), 'str', 'Cannot invoke a method over a {0}')
        # Obtaining the member 'format' of a type (line 43)
        format_8743 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 43, 39), str_8742, 'format')
        # Calling format(args, kwargs) (line 43)
        format_call_result_8748 = invoke(stypy.reporting.localization.Localization(__file__, 43, 39), format_8743, *[name___8746], **kwargs_8747)
        
        # Processing the call keyword arguments (line 43)
        kwargs_8749 = {}
        # Getting the type of 'TypeError' (line 43)
        TypeError_8740 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 43)
        TypeError_call_result_8750 = invoke(stypy.reporting.localization.Localization(__file__, 43, 15), TypeError_8740, *[localization_8741, format_call_result_8748], **kwargs_8749)
        
        # Assigning a type to the variable 'stypy_return_type' (line 43)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 43, 8), 'stypy_return_type', TypeError_call_result_8750)
        
        # ################# End of 'invoke(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'invoke' in the type store
        # Getting the type of 'stypy_return_type' (line 39)
        stypy_return_type_8751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8751)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'invoke'
        return stypy_return_type_8751


    @norecursion
    def delete_member(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'delete_member'
        module_type_store = module_type_store.open_function_context('delete_member', 47, 4, False)
        # Assigning a type to the variable 'self' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.delete_member.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.delete_member.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.delete_member.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.delete_member.__dict__.__setitem__('stypy_function_name', 'NonPythonType.delete_member')
        NonPythonType.delete_member.__dict__.__setitem__('stypy_param_names_list', ['localization', 'member'])
        NonPythonType.delete_member.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.delete_member.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.delete_member.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.delete_member.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.delete_member.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.delete_member.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.delete_member', ['localization', 'member'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'delete_member', localization, ['localization', 'member'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'delete_member(...)' code ##################

        str_8752 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n        Returns an error if called\n        ')
        
        # Call to TypeError(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'localization' (line 51)
        localization_8754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 25), 'localization', False)
        
        # Call to format(...): (line 51)
        # Processing the call arguments (line 51)
        # Getting the type of 'self' (line 51)
        self_8757 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 80), 'self', False)
        # Obtaining the member '__class__' of a type (line 51)
        class___8758 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 80), self_8757, '__class__')
        # Obtaining the member '__name__' of a type (line 51)
        name___8759 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 80), class___8758, '__name__')
        # Processing the call keyword arguments (line 51)
        kwargs_8760 = {}
        str_8755 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 51, 39), 'str', 'Cannot delete a member of a {0}')
        # Obtaining the member 'format' of a type (line 51)
        format_8756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 39), str_8755, 'format')
        # Calling format(args, kwargs) (line 51)
        format_call_result_8761 = invoke(stypy.reporting.localization.Localization(__file__, 51, 39), format_8756, *[name___8759], **kwargs_8760)
        
        # Processing the call keyword arguments (line 51)
        kwargs_8762 = {}
        # Getting the type of 'TypeError' (line 51)
        TypeError_8753 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 51)
        TypeError_call_result_8763 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), TypeError_8753, *[localization_8754, format_call_result_8761], **kwargs_8762)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', TypeError_call_result_8763)
        
        # ################# End of 'delete_member(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'delete_member' in the type store
        # Getting the type of 'stypy_return_type' (line 47)
        stypy_return_type_8764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 47, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8764)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'delete_member'
        return stypy_return_type_8764


    @norecursion
    def supports_structural_reflection(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'supports_structural_reflection'
        module_type_store = module_type_store.open_function_context('supports_structural_reflection', 53, 4, False)
        # Assigning a type to the variable 'self' (line 54)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_function_name', 'NonPythonType.supports_structural_reflection')
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_param_names_list', [])
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.supports_structural_reflection.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.supports_structural_reflection', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'supports_structural_reflection', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'supports_structural_reflection(...)' code ##################

        str_8765 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n        Returns an error if called\n        ')
        # Getting the type of 'False' (line 57)
        False_8766 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 15), 'False')
        # Assigning a type to the variable 'stypy_return_type' (line 57)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 57, 8), 'stypy_return_type', False_8766)
        
        # ################# End of 'supports_structural_reflection(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'supports_structural_reflection' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_8767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8767)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'supports_structural_reflection'
        return stypy_return_type_8767


    @norecursion
    def change_type(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_type'
        module_type_store = module_type_store.open_function_context('change_type', 59, 4, False)
        # Assigning a type to the variable 'self' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.change_type.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.change_type.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.change_type.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.change_type.__dict__.__setitem__('stypy_function_name', 'NonPythonType.change_type')
        NonPythonType.change_type.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_type'])
        NonPythonType.change_type.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.change_type.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.change_type.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.change_type.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.change_type.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.change_type.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.change_type', ['localization', 'new_type'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_type', localization, ['localization', 'new_type'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_type(...)' code ##################

        str_8768 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, (-1)), 'str', '\n        Returns an error if called\n        ')
        
        # Call to TypeError(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'localization' (line 63)
        localization_8770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 25), 'localization', False)
        
        # Call to format(...): (line 63)
        # Processing the call arguments (line 63)
        # Getting the type of 'self' (line 63)
        self_8773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 80), 'self', False)
        # Obtaining the member '__class__' of a type (line 63)
        class___8774 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 80), self_8773, '__class__')
        # Obtaining the member '__name__' of a type (line 63)
        name___8775 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 80), class___8774, '__name__')
        # Processing the call keyword arguments (line 63)
        kwargs_8776 = {}
        str_8771 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 63, 39), 'str', 'Cannot change the type of a {0}')
        # Obtaining the member 'format' of a type (line 63)
        format_8772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 63, 39), str_8771, 'format')
        # Calling format(args, kwargs) (line 63)
        format_call_result_8777 = invoke(stypy.reporting.localization.Localization(__file__, 63, 39), format_8772, *[name___8775], **kwargs_8776)
        
        # Processing the call keyword arguments (line 63)
        kwargs_8778 = {}
        # Getting the type of 'TypeError' (line 63)
        TypeError_8769 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 63)
        TypeError_call_result_8779 = invoke(stypy.reporting.localization.Localization(__file__, 63, 15), TypeError_8769, *[localization_8770, format_call_result_8777], **kwargs_8778)
        
        # Assigning a type to the variable 'stypy_return_type' (line 63)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 63, 8), 'stypy_return_type', TypeError_call_result_8779)
        
        # ################# End of 'change_type(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_type' in the type store
        # Getting the type of 'stypy_return_type' (line 59)
        stypy_return_type_8780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8780)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_type'
        return stypy_return_type_8780


    @norecursion
    def change_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'change_base_types'
        module_type_store = module_type_store.open_function_context('change_base_types', 65, 4, False)
        # Assigning a type to the variable 'self' (line 66)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_function_name', 'NonPythonType.change_base_types')
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.change_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.change_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'change_base_types', localization, ['localization', 'new_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'change_base_types(...)' code ##################

        str_8781 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 68, (-1)), 'str', '\n        Returns an error if called\n        ')
        
        # Call to TypeError(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'localization' (line 69)
        localization_8783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 25), 'localization', False)
        
        # Call to format(...): (line 69)
        # Processing the call arguments (line 69)
        # Getting the type of 'self' (line 69)
        self_8786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 86), 'self', False)
        # Obtaining the member '__class__' of a type (line 69)
        class___8787 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 86), self_8786, '__class__')
        # Obtaining the member '__name__' of a type (line 69)
        name___8788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 86), class___8787, '__name__')
        # Processing the call keyword arguments (line 69)
        kwargs_8789 = {}
        str_8784 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 69, 39), 'str', 'Cannot change the base types of a {0}')
        # Obtaining the member 'format' of a type (line 69)
        format_8785 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 69, 39), str_8784, 'format')
        # Calling format(args, kwargs) (line 69)
        format_call_result_8790 = invoke(stypy.reporting.localization.Localization(__file__, 69, 39), format_8785, *[name___8788], **kwargs_8789)
        
        # Processing the call keyword arguments (line 69)
        kwargs_8791 = {}
        # Getting the type of 'TypeError' (line 69)
        TypeError_8782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 15), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 69)
        TypeError_call_result_8792 = invoke(stypy.reporting.localization.Localization(__file__, 69, 15), TypeError_8782, *[localization_8783, format_call_result_8790], **kwargs_8791)
        
        # Assigning a type to the variable 'stypy_return_type' (line 69)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 8), 'stypy_return_type', TypeError_call_result_8792)
        
        # ################# End of 'change_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'change_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 65)
        stypy_return_type_8793 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8793)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'change_base_types'
        return stypy_return_type_8793


    @norecursion
    def add_base_types(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'add_base_types'
        module_type_store = module_type_store.open_function_context('add_base_types', 71, 4, False)
        # Assigning a type to the variable 'self' (line 72)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_function_name', 'NonPythonType.add_base_types')
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_param_names_list', ['localization', 'new_types'])
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.add_base_types.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.add_base_types', ['localization', 'new_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'add_base_types', localization, ['localization', 'new_types'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'add_base_types(...)' code ##################

        str_8794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 74, (-1)), 'str', '\n        Returns an error if called\n        ')
        
        # Call to change_base_types(...): (line 75)
        # Processing the call arguments (line 75)
        # Getting the type of 'localization' (line 75)
        localization_8797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 31), 'localization', False)
        # Getting the type of 'new_types' (line 75)
        new_types_8798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 45), 'new_types', False)
        # Processing the call keyword arguments (line 75)
        kwargs_8799 = {}
        # Getting the type of 'self' (line 75)
        self_8795 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 8), 'self', False)
        # Obtaining the member 'change_base_types' of a type (line 75)
        change_base_types_8796 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 8), self_8795, 'change_base_types')
        # Calling change_base_types(args, kwargs) (line 75)
        change_base_types_call_result_8800 = invoke(stypy.reporting.localization.Localization(__file__, 75, 8), change_base_types_8796, *[localization_8797, new_types_8798], **kwargs_8799)
        
        
        # ################# End of 'add_base_types(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'add_base_types' in the type store
        # Getting the type of 'stypy_return_type' (line 71)
        stypy_return_type_8801 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8801)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'add_base_types'
        return stypy_return_type_8801


    @norecursion
    def clone(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'clone'
        module_type_store = module_type_store.open_function_context('clone', 79, 4, False)
        # Assigning a type to the variable 'self' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        NonPythonType.clone.__dict__.__setitem__('stypy_localization', localization)
        NonPythonType.clone.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        NonPythonType.clone.__dict__.__setitem__('stypy_type_store', module_type_store)
        NonPythonType.clone.__dict__.__setitem__('stypy_function_name', 'NonPythonType.clone')
        NonPythonType.clone.__dict__.__setitem__('stypy_param_names_list', [])
        NonPythonType.clone.__dict__.__setitem__('stypy_varargs_param_name', None)
        NonPythonType.clone.__dict__.__setitem__('stypy_kwargs_param_name', None)
        NonPythonType.clone.__dict__.__setitem__('stypy_call_defaults', defaults)
        NonPythonType.clone.__dict__.__setitem__('stypy_call_varargs', varargs)
        NonPythonType.clone.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        NonPythonType.clone.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.clone', [], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'clone', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'clone(...)' code ##################

        # Getting the type of 'self' (line 80)
        self_8802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 15), 'self')
        # Assigning a type to the variable 'stypy_return_type' (line 80)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 8), 'stypy_return_type', self_8802)
        
        # ################# End of 'clone(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'clone' in the type store
        # Getting the type of 'stypy_return_type' (line 79)
        stypy_return_type_8803 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_8803)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'clone'
        return stypy_return_type_8803


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 5, 0, False)
        # Assigning a type to the variable 'self' (line 6)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'NonPythonType.__init__', [], None, None, defaults, varargs, kwargs)

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


# Assigning a type to the variable 'NonPythonType' (line 5)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'NonPythonType', NonPythonType)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
