
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: import types
3: 
4: from ....errors_copy.type_error_copy import TypeError
5: from ....python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy import get_type_sample_value
6: from ....python_lib_copy.python_types_copy.type_copy import Type
7: from call_handler_copy import CallHandler
8: from ....python_lib_copy.member_call_copy.call_handlers_helper_methods_copy import format_call
9: 
10: 
11: class FakeParamValuesCallHandler(CallHandler):
12:     '''
13:     This handler simulated the call to a callable entity by creating fake values of the passed parameters, actually
14:     call the callable entity and returned the type of the result. It is used when no other call handler can be
15:     used to call this entity, meaning that this is third-party code or Python library modules with no source code
16:     available to be transformed (and therefore using the UserCallablesCallHandler) or this module has no type rule
17:     file associated (which can be done in the future, and stypy will use the TypeRuleCallHandler instead).
18:     '''
19: 
20:     @staticmethod
21:     def __get_type_instance(arg):
22:         '''
23:         Obtain a fake value for the type represented by arg
24:         :param arg: Type
25:         :return: Value for that type
26:         '''
27:         if isinstance(arg, Type):
28:             # If the TypeInferenceProxy holds an instance, return that instance
29:             instance = arg.get_instance()
30:             if instance is not None:
31:                 return instance
32:             else:
33:                 # If the TypeInferenceProxy holds a value, return that value
34:                 if hasattr(arg, "has_value"):
35:                     if arg.has_value():
36:                         return arg.get_value()
37: 
38:                 # Else obtain a predefined value for that type
39:                 return get_type_sample_value(arg.get_python_type())
40: 
41:         # Else obtain a predefined value for that type
42:         return get_type_sample_value(arg)
43: 
44:     @staticmethod
45:     def __get_arg_sample_values(arg_types):
46:         '''
47:         Obtain a fake value for all the types passed in a list
48:         :param arg_types: List of types
49:         :return: List of values
50:         '''
51:         return map(lambda arg: FakeParamValuesCallHandler.__get_type_instance(arg), arg_types)
52: 
53:     @staticmethod
54:     def __get_kwarg_sample_values(kwargs_types):
55:         '''
56:         Obtain a fake value for all the types passed on a dict. This is used for keyword arguments
57:         :param kwargs_types: Dict of types
58:         :return: Dict of values
59:         '''
60:         kwargs_values = {}
61:         for value in kwargs_types:
62:             kwargs_values[value] = FakeParamValuesCallHandler.__get_type_instance(kwargs_types[value])
63: 
64:         return kwargs_values
65: 
66:     def __init__(self, fake_self=None):
67:         CallHandler.__init__(self)
68:         self.fake_self = fake_self
69: 
70:     def applies_to(self, proxy_obj, callable_entity):
71:         '''
72:         This method determines if this call handler is able to respond to a call to callable_entity. The call handler
73:         respond to any callable code, as it is the last to be used.
74:         :param proxy_obj: TypeInferenceProxy that hold the callable entity
75:         :param callable_entity: Callable entity
76:         :return: Always True
77:         '''
78:         return True
79: 
80:     def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
81:         '''
82:         This call handler substitutes the param types by fake values and perform a call to the real python callable
83:         entity, returning the type of the return type if the called entity is not a class (in that case it returns the
84:         created instance)
85: 
86:         :param proxy_obj: TypeInferenceProxy that hold the callable entity
87:         :param localization: Caller information
88:         :param callable_entity: Callable entity
89:         :param arg_types: Arguments
90:         :param kwargs_types: Keyword arguments
91:         :return: Return type of the call
92:         '''
93: 
94:         # Obtain values for all parameters
95:         arg_values = FakeParamValuesCallHandler.__get_arg_sample_values(arg_types)
96:         kwargs_values = FakeParamValuesCallHandler.__get_kwarg_sample_values(kwargs_types)
97: 
98:         callable_python_entity = callable_entity
99:         try:
100:             if (self.fake_self is not None) and ((not hasattr(callable_python_entity, '__self__')) or (
101:                         hasattr(callable_python_entity, '__self__') and callable_python_entity.__self__ is None)):
102:                 arg_values = [self.fake_self] + arg_values
103: 
104:             # Call
105:             call_result = callable_python_entity(*arg_values, **kwargs_values)
106: 
107:             # Calculate the return type
108:             if call_result is not None:
109:                 if not inspect.isclass(callable_entity):
110:                     return type(call_result)
111: 
112:                 if isinstance(type(call_result).__dict__, types.DictProxyType):
113:                     if hasattr(call_result, '__dict__'):
114:                         if not isinstance(call_result.__dict__, types.DictProxyType):
115:                             return call_result
116: 
117:                     return type(call_result)
118: 
119:             return call_result
120:         except Exception as ex:
121:             str_call = format_call(callable_entity, arg_types, kwargs_types)
122:             return TypeError(localization, "{0}: {1}".format(str_call, str(ex)))
123: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import inspect' statement (line 1)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import types' statement (line 2)
import types

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'types', types, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5968 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_5968) is not StypyTypeError):

    if (import_5968 != 'pyd_module'):
        __import__(import_5968)
        sys_modules_5969 = sys.modules[import_5968]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_5969.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_5969, sys_modules_5969.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_5968)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy import get_type_sample_value' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5970 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy')

if (type(import_5970) is not StypyTypeError):

    if (import_5970 != 'pyd_module'):
        __import__(import_5970)
        sys_modules_5971 = sys.modules[import_5970]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy', sys_modules_5971.module_type_store, module_type_store, ['get_type_sample_value'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_5971, sys_modules_5971.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy import get_type_sample_value

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy', None, module_type_store, ['get_type_sample_value'], [get_type_sample_value])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy', import_5970)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5972 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_5972) is not StypyTypeError):

    if (import_5972 != 'pyd_module'):
        __import__(import_5972)
        sys_modules_5973 = sys.modules[import_5972]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_5973.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_5973, sys_modules_5973.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_copy', import_5972)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from call_handler_copy import CallHandler' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5974 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_handler_copy')

if (type(import_5974) is not StypyTypeError):

    if (import_5974 != 'pyd_module'):
        __import__(import_5974)
        sys_modules_5975 = sys.modules[import_5974]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_handler_copy', sys_modules_5975.module_type_store, module_type_store, ['CallHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_5975, sys_modules_5975.module_type_store, module_type_store)
    else:
        from call_handler_copy import CallHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_handler_copy', None, module_type_store, ['CallHandler'], [CallHandler])

else:
    # Assigning a type to the variable 'call_handler_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_handler_copy', import_5974)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy import format_call' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5976 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy')

if (type(import_5976) is not StypyTypeError):

    if (import_5976 != 'pyd_module'):
        __import__(import_5976)
        sys_modules_5977 = sys.modules[import_5976]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy', sys_modules_5977.module_type_store, module_type_store, ['format_call'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_5977, sys_modules_5977.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy import format_call

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy', None, module_type_store, ['format_call'], [format_call])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy', import_5976)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

# Declaration of the 'FakeParamValuesCallHandler' class
# Getting the type of 'CallHandler' (line 11)
CallHandler_5978 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 33), 'CallHandler')

class FakeParamValuesCallHandler(CallHandler_5978, ):
    str_5979 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', '\n    This handler simulated the call to a callable entity by creating fake values of the passed parameters, actually\n    call the callable entity and returned the type of the result. It is used when no other call handler can be\n    used to call this entity, meaning that this is third-party code or Python library modules with no source code\n    available to be transformed (and therefore using the UserCallablesCallHandler) or this module has no type rule\n    file associated (which can be done in the future, and stypy will use the TypeRuleCallHandler instead).\n    ')

    @staticmethod
    @norecursion
    def __get_type_instance(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_type_instance'
        module_type_store = module_type_store.open_function_context('__get_type_instance', 20, 4, False)
        
        # Passed parameters checking function
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_localization', localization)
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_type_of_self', None)
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_function_name', '__get_type_instance')
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_param_names_list', ['arg'])
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeParamValuesCallHandler.__get_type_instance.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__get_type_instance', ['arg'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_type_instance', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_type_instance(...)' code ##################

        str_5980 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', '\n        Obtain a fake value for the type represented by arg\n        :param arg: Type\n        :return: Value for that type\n        ')
        
        # Call to isinstance(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'arg' (line 27)
        arg_5982 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'arg', False)
        # Getting the type of 'Type' (line 27)
        Type_5983 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'Type', False)
        # Processing the call keyword arguments (line 27)
        kwargs_5984 = {}
        # Getting the type of 'isinstance' (line 27)
        isinstance_5981 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 27)
        isinstance_call_result_5985 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), isinstance_5981, *[arg_5982, Type_5983], **kwargs_5984)
        
        # Testing if the type of an if condition is none (line 27)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 27, 8), isinstance_call_result_5985):
            pass
        else:
            
            # Testing the type of an if condition (line 27)
            if_condition_5986 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 8), isinstance_call_result_5985)
            # Assigning a type to the variable 'if_condition_5986' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_condition_5986', if_condition_5986)
            # SSA begins for if statement (line 27)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 29):
            
            # Call to get_instance(...): (line 29)
            # Processing the call keyword arguments (line 29)
            kwargs_5989 = {}
            # Getting the type of 'arg' (line 29)
            arg_5987 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'arg', False)
            # Obtaining the member 'get_instance' of a type (line 29)
            get_instance_5988 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), arg_5987, 'get_instance')
            # Calling get_instance(args, kwargs) (line 29)
            get_instance_call_result_5990 = invoke(stypy.reporting.localization.Localization(__file__, 29, 23), get_instance_5988, *[], **kwargs_5989)
            
            # Assigning a type to the variable 'instance' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'instance', get_instance_call_result_5990)
            
            # Type idiom detected: calculating its left and rigth part (line 30)
            # Getting the type of 'instance' (line 30)
            instance_5991 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'instance')
            # Getting the type of 'None' (line 30)
            None_5992 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'None')
            
            (may_be_5993, more_types_in_union_5994) = may_not_be_none(instance_5991, None_5992)

            if may_be_5993:

                if more_types_in_union_5994:
                    # Runtime conditional SSA (line 30)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'instance' (line 31)
                instance_5995 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'instance')
                # Assigning a type to the variable 'stypy_return_type' (line 31)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'stypy_return_type', instance_5995)

                if more_types_in_union_5994:
                    # Runtime conditional SSA for else branch (line 30)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_5993) or more_types_in_union_5994):
                
                # Type idiom detected: calculating its left and rigth part (line 34)
                str_5996 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'str', 'has_value')
                # Getting the type of 'arg' (line 34)
                arg_5997 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'arg')
                
                (may_be_5998, more_types_in_union_5999) = may_provide_member(str_5996, arg_5997)

                if may_be_5998:

                    if more_types_in_union_5999:
                        # Runtime conditional SSA (line 34)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'arg' (line 34)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'arg', remove_not_member_provider_from_union(arg_5997, 'has_value'))
                    
                    # Call to has_value(...): (line 35)
                    # Processing the call keyword arguments (line 35)
                    kwargs_6002 = {}
                    # Getting the type of 'arg' (line 35)
                    arg_6000 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'arg', False)
                    # Obtaining the member 'has_value' of a type (line 35)
                    has_value_6001 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), arg_6000, 'has_value')
                    # Calling has_value(args, kwargs) (line 35)
                    has_value_call_result_6003 = invoke(stypy.reporting.localization.Localization(__file__, 35, 23), has_value_6001, *[], **kwargs_6002)
                    
                    # Testing if the type of an if condition is none (line 35)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 20), has_value_call_result_6003):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 35)
                        if_condition_6004 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 20), has_value_call_result_6003)
                        # Assigning a type to the variable 'if_condition_6004' (line 35)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'if_condition_6004', if_condition_6004)
                        # SSA begins for if statement (line 35)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to get_value(...): (line 36)
                        # Processing the call keyword arguments (line 36)
                        kwargs_6007 = {}
                        # Getting the type of 'arg' (line 36)
                        arg_6005 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'arg', False)
                        # Obtaining the member 'get_value' of a type (line 36)
                        get_value_6006 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 31), arg_6005, 'get_value')
                        # Calling get_value(args, kwargs) (line 36)
                        get_value_call_result_6008 = invoke(stypy.reporting.localization.Localization(__file__, 36, 31), get_value_6006, *[], **kwargs_6007)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 36)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'stypy_return_type', get_value_call_result_6008)
                        # SSA join for if statement (line 35)
                        module_type_store = module_type_store.join_ssa_context()
                        


                    if more_types_in_union_5999:
                        # SSA join for if statement (line 34)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Call to get_type_sample_value(...): (line 39)
                # Processing the call arguments (line 39)
                
                # Call to get_python_type(...): (line 39)
                # Processing the call keyword arguments (line 39)
                kwargs_6012 = {}
                # Getting the type of 'arg' (line 39)
                arg_6010 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 45), 'arg', False)
                # Obtaining the member 'get_python_type' of a type (line 39)
                get_python_type_6011 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 45), arg_6010, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 39)
                get_python_type_call_result_6013 = invoke(stypy.reporting.localization.Localization(__file__, 39, 45), get_python_type_6011, *[], **kwargs_6012)
                
                # Processing the call keyword arguments (line 39)
                kwargs_6014 = {}
                # Getting the type of 'get_type_sample_value' (line 39)
                get_type_sample_value_6009 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'get_type_sample_value', False)
                # Calling get_type_sample_value(args, kwargs) (line 39)
                get_type_sample_value_call_result_6015 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), get_type_sample_value_6009, *[get_python_type_call_result_6013], **kwargs_6014)
                
                # Assigning a type to the variable 'stypy_return_type' (line 39)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'stypy_return_type', get_type_sample_value_call_result_6015)

                if (may_be_5993 and more_types_in_union_5994):
                    # SSA join for if statement (line 30)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 27)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to get_type_sample_value(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'arg' (line 42)
        arg_6017 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 37), 'arg', False)
        # Processing the call keyword arguments (line 42)
        kwargs_6018 = {}
        # Getting the type of 'get_type_sample_value' (line 42)
        get_type_sample_value_6016 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'get_type_sample_value', False)
        # Calling get_type_sample_value(args, kwargs) (line 42)
        get_type_sample_value_call_result_6019 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), get_type_sample_value_6016, *[arg_6017], **kwargs_6018)
        
        # Assigning a type to the variable 'stypy_return_type' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', get_type_sample_value_call_result_6019)
        
        # ################# End of '__get_type_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_type_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_6020 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6020)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_type_instance'
        return stypy_return_type_6020


    @staticmethod
    @norecursion
    def __get_arg_sample_values(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_arg_sample_values'
        module_type_store = module_type_store.open_function_context('__get_arg_sample_values', 44, 4, False)
        
        # Passed parameters checking function
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_localization', localization)
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_type_of_self', None)
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_function_name', '__get_arg_sample_values')
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_param_names_list', ['arg_types'])
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeParamValuesCallHandler.__get_arg_sample_values.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__get_arg_sample_values', ['arg_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_arg_sample_values', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_arg_sample_values(...)' code ##################

        str_6021 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n        Obtain a fake value for all the types passed in a list\n        :param arg_types: List of types\n        :return: List of values\n        ')
        
        # Call to map(...): (line 51)
        # Processing the call arguments (line 51)

        @norecursion
        def _stypy_temp_lambda_13(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_13'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_13', 51, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_13.stypy_localization = localization
            _stypy_temp_lambda_13.stypy_type_of_self = None
            _stypy_temp_lambda_13.stypy_type_store = module_type_store
            _stypy_temp_lambda_13.stypy_function_name = '_stypy_temp_lambda_13'
            _stypy_temp_lambda_13.stypy_param_names_list = ['arg']
            _stypy_temp_lambda_13.stypy_varargs_param_name = None
            _stypy_temp_lambda_13.stypy_kwargs_param_name = None
            _stypy_temp_lambda_13.stypy_call_defaults = defaults
            _stypy_temp_lambda_13.stypy_call_varargs = varargs
            _stypy_temp_lambda_13.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_13', ['arg'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_13', ['arg'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to __get_type_instance(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'arg' (line 51)
            arg_6025 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 78), 'arg', False)
            # Processing the call keyword arguments (line 51)
            kwargs_6026 = {}
            # Getting the type of 'FakeParamValuesCallHandler' (line 51)
            FakeParamValuesCallHandler_6023 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'FakeParamValuesCallHandler', False)
            # Obtaining the member '__get_type_instance' of a type (line 51)
            get_type_instance_6024 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), FakeParamValuesCallHandler_6023, '__get_type_instance')
            # Calling __get_type_instance(args, kwargs) (line 51)
            get_type_instance_call_result_6027 = invoke(stypy.reporting.localization.Localization(__file__, 51, 31), get_type_instance_6024, *[arg_6025], **kwargs_6026)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'stypy_return_type', get_type_instance_call_result_6027)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_13' in the type store
            # Getting the type of 'stypy_return_type' (line 51)
            stypy_return_type_6028 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_6028)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_13'
            return stypy_return_type_6028

        # Assigning a type to the variable '_stypy_temp_lambda_13' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), '_stypy_temp_lambda_13', _stypy_temp_lambda_13)
        # Getting the type of '_stypy_temp_lambda_13' (line 51)
        _stypy_temp_lambda_13_6029 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), '_stypy_temp_lambda_13')
        # Getting the type of 'arg_types' (line 51)
        arg_types_6030 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 84), 'arg_types', False)
        # Processing the call keyword arguments (line 51)
        kwargs_6031 = {}
        # Getting the type of 'map' (line 51)
        map_6022 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'map', False)
        # Calling map(args, kwargs) (line 51)
        map_call_result_6032 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), map_6022, *[_stypy_temp_lambda_13_6029, arg_types_6030], **kwargs_6031)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', map_call_result_6032)
        
        # ################# End of '__get_arg_sample_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_arg_sample_values' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_6033 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6033)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_arg_sample_values'
        return stypy_return_type_6033


    @staticmethod
    @norecursion
    def __get_kwarg_sample_values(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__get_kwarg_sample_values'
        module_type_store = module_type_store.open_function_context('__get_kwarg_sample_values', 53, 4, False)
        
        # Passed parameters checking function
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_localization', localization)
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_type_of_self', None)
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_function_name', '__get_kwarg_sample_values')
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_param_names_list', ['kwargs_types'])
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeParamValuesCallHandler.__get_kwarg_sample_values.__dict__.__setitem__('stypy_declared_arg_number', 1)
        arguments = process_argument_values(localization, None, module_type_store, '__get_kwarg_sample_values', ['kwargs_types'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__get_kwarg_sample_values', localization, [], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__get_kwarg_sample_values(...)' code ##################

        str_6034 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n        Obtain a fake value for all the types passed on a dict. This is used for keyword arguments\n        :param kwargs_types: Dict of types\n        :return: Dict of values\n        ')
        
        # Assigning a Dict to a Name (line 60):
        
        # Obtaining an instance of the builtin type 'dict' (line 60)
        dict_6035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 60)
        
        # Assigning a type to the variable 'kwargs_values' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'kwargs_values', dict_6035)
        
        # Getting the type of 'kwargs_types' (line 61)
        kwargs_types_6036 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'kwargs_types')
        # Assigning a type to the variable 'kwargs_types_6036' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'kwargs_types_6036', kwargs_types_6036)
        # Testing if the for loop is going to be iterated (line 61)
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), kwargs_types_6036)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 8), kwargs_types_6036):
            # Getting the type of the for loop variable (line 61)
            for_loop_var_6037 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), kwargs_types_6036)
            # Assigning a type to the variable 'value' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'value', for_loop_var_6037)
            # SSA begins for a for statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Subscript (line 62):
            
            # Call to __get_type_instance(...): (line 62)
            # Processing the call arguments (line 62)
            
            # Obtaining the type of the subscript
            # Getting the type of 'value' (line 62)
            value_6040 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 95), 'value', False)
            # Getting the type of 'kwargs_types' (line 62)
            kwargs_types_6041 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 82), 'kwargs_types', False)
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___6042 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 82), kwargs_types_6041, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_6043 = invoke(stypy.reporting.localization.Localization(__file__, 62, 82), getitem___6042, value_6040)
            
            # Processing the call keyword arguments (line 62)
            kwargs_6044 = {}
            # Getting the type of 'FakeParamValuesCallHandler' (line 62)
            FakeParamValuesCallHandler_6038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 35), 'FakeParamValuesCallHandler', False)
            # Obtaining the member '__get_type_instance' of a type (line 62)
            get_type_instance_6039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 35), FakeParamValuesCallHandler_6038, '__get_type_instance')
            # Calling __get_type_instance(args, kwargs) (line 62)
            get_type_instance_call_result_6045 = invoke(stypy.reporting.localization.Localization(__file__, 62, 35), get_type_instance_6039, *[subscript_call_result_6043], **kwargs_6044)
            
            # Getting the type of 'kwargs_values' (line 62)
            kwargs_values_6046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'kwargs_values')
            # Getting the type of 'value' (line 62)
            value_6047 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'value')
            # Storing an element on a container (line 62)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 12), kwargs_values_6046, (value_6047, get_type_instance_call_result_6045))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'kwargs_values' (line 64)
        kwargs_values_6048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'kwargs_values')
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', kwargs_values_6048)
        
        # ################# End of '__get_kwarg_sample_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_kwarg_sample_values' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_6049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6049)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_kwarg_sample_values'
        return stypy_return_type_6049


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 66)
        None_6050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 33), 'None')
        defaults = [None_6050]
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 66, 4, False)
        # Assigning a type to the variable 'self' (line 67)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 67, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeParamValuesCallHandler.__init__', ['fake_self'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return

        # Initialize method data
        init_call_information(module_type_store, '__init__', localization, ['fake_self'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__init__(...)' code ##################

        
        # Call to __init__(...): (line 67)
        # Processing the call arguments (line 67)
        # Getting the type of 'self' (line 67)
        self_6053 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'self', False)
        # Processing the call keyword arguments (line 67)
        kwargs_6054 = {}
        # Getting the type of 'CallHandler' (line 67)
        CallHandler_6051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'CallHandler', False)
        # Obtaining the member '__init__' of a type (line 67)
        init___6052 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), CallHandler_6051, '__init__')
        # Calling __init__(args, kwargs) (line 67)
        init___call_result_6055 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), init___6052, *[self_6053], **kwargs_6054)
        
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'fake_self' (line 68)
        fake_self_6056 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'fake_self')
        # Getting the type of 'self' (line 68)
        self_6057 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'fake_self' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_6057, 'fake_self', fake_self_6056)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


    @norecursion
    def applies_to(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'applies_to'
        module_type_store = module_type_store.open_function_context('applies_to', 70, 4, False)
        # Assigning a type to the variable 'self' (line 71)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_localization', localization)
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_function_name', 'FakeParamValuesCallHandler.applies_to')
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'callable_entity'])
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_varargs_param_name', None)
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeParamValuesCallHandler.applies_to.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeParamValuesCallHandler.applies_to', ['proxy_obj', 'callable_entity'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'applies_to', localization, ['proxy_obj', 'callable_entity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'applies_to(...)' code ##################

        str_6058 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', '\n        This method determines if this call handler is able to respond to a call to callable_entity. The call handler\n        respond to any callable code, as it is the last to be used.\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param callable_entity: Callable entity\n        :return: Always True\n        ')
        # Getting the type of 'True' (line 78)
        True_6059 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', True_6059)
        
        # ################# End of 'applies_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'applies_to' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_6060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6060)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'applies_to'
        return stypy_return_type_6060


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 80, 4, False)
        # Assigning a type to the variable 'self' (line 81)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_localization', localization)
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_function_name', 'FakeParamValuesCallHandler.__call__')
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'localization', 'callable_entity'])
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'arg_types')
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs_types')
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FakeParamValuesCallHandler.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FakeParamValuesCallHandler.__call__', ['proxy_obj', 'localization', 'callable_entity'], 'arg_types', 'kwargs_types', defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__call__', localization, ['proxy_obj', 'localization', 'callable_entity'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__call__(...)' code ##################

        str_6061 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', '\n        This call handler substitutes the param types by fake values and perform a call to the real python callable\n        entity, returning the type of the return type if the called entity is not a class (in that case it returns the\n        created instance)\n\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param localization: Caller information\n        :param callable_entity: Callable entity\n        :param arg_types: Arguments\n        :param kwargs_types: Keyword arguments\n        :return: Return type of the call\n        ')
        
        # Assigning a Call to a Name (line 95):
        
        # Call to __get_arg_sample_values(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'arg_types' (line 95)
        arg_types_6064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 72), 'arg_types', False)
        # Processing the call keyword arguments (line 95)
        kwargs_6065 = {}
        # Getting the type of 'FakeParamValuesCallHandler' (line 95)
        FakeParamValuesCallHandler_6062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'FakeParamValuesCallHandler', False)
        # Obtaining the member '__get_arg_sample_values' of a type (line 95)
        get_arg_sample_values_6063 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 21), FakeParamValuesCallHandler_6062, '__get_arg_sample_values')
        # Calling __get_arg_sample_values(args, kwargs) (line 95)
        get_arg_sample_values_call_result_6066 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), get_arg_sample_values_6063, *[arg_types_6064], **kwargs_6065)
        
        # Assigning a type to the variable 'arg_values' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'arg_values', get_arg_sample_values_call_result_6066)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to __get_kwarg_sample_values(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'kwargs_types' (line 96)
        kwargs_types_6069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 77), 'kwargs_types', False)
        # Processing the call keyword arguments (line 96)
        kwargs_6070 = {}
        # Getting the type of 'FakeParamValuesCallHandler' (line 96)
        FakeParamValuesCallHandler_6067 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'FakeParamValuesCallHandler', False)
        # Obtaining the member '__get_kwarg_sample_values' of a type (line 96)
        get_kwarg_sample_values_6068 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 24), FakeParamValuesCallHandler_6067, '__get_kwarg_sample_values')
        # Calling __get_kwarg_sample_values(args, kwargs) (line 96)
        get_kwarg_sample_values_call_result_6071 = invoke(stypy.reporting.localization.Localization(__file__, 96, 24), get_kwarg_sample_values_6068, *[kwargs_types_6069], **kwargs_6070)
        
        # Assigning a type to the variable 'kwargs_values' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'kwargs_values', get_kwarg_sample_values_call_result_6071)
        
        # Assigning a Name to a Name (line 98):
        # Getting the type of 'callable_entity' (line 98)
        callable_entity_6072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'callable_entity')
        # Assigning a type to the variable 'callable_python_entity' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'callable_python_entity', callable_entity_6072)
        
        
        # SSA begins for try-except statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 100)
        self_6073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'self')
        # Obtaining the member 'fake_self' of a type (line 100)
        fake_self_6074 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 16), self_6073, 'fake_self')
        # Getting the type of 'None' (line 100)
        None_6075 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'None')
        # Applying the binary operator 'isnot' (line 100)
        result_is_not_6076 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 16), 'isnot', fake_self_6074, None_6075)
        
        
        # Evaluating a boolean operation
        
        
        # Call to hasattr(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'callable_python_entity' (line 100)
        callable_python_entity_6078 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 62), 'callable_python_entity', False)
        str_6079 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 86), 'str', '__self__')
        # Processing the call keyword arguments (line 100)
        kwargs_6080 = {}
        # Getting the type of 'hasattr' (line 100)
        hasattr_6077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 54), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 100)
        hasattr_call_result_6081 = invoke(stypy.reporting.localization.Localization(__file__, 100, 54), hasattr_6077, *[callable_python_entity_6078, str_6079], **kwargs_6080)
        
        # Applying the 'not' unary operator (line 100)
        result_not__6082 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 50), 'not', hasattr_call_result_6081)
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'callable_python_entity' (line 101)
        callable_python_entity_6084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'callable_python_entity', False)
        str_6085 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 56), 'str', '__self__')
        # Processing the call keyword arguments (line 101)
        kwargs_6086 = {}
        # Getting the type of 'hasattr' (line 101)
        hasattr_6083 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 101)
        hasattr_call_result_6087 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), hasattr_6083, *[callable_python_entity_6084, str_6085], **kwargs_6086)
        
        
        # Getting the type of 'callable_python_entity' (line 101)
        callable_python_entity_6088 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 72), 'callable_python_entity')
        # Obtaining the member '__self__' of a type (line 101)
        self___6089 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 72), callable_python_entity_6088, '__self__')
        # Getting the type of 'None' (line 101)
        None_6090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 107), 'None')
        # Applying the binary operator 'is' (line 101)
        result_is__6091 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 72), 'is', self___6089, None_6090)
        
        # Applying the binary operator 'and' (line 101)
        result_and_keyword_6092 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), 'and', hasattr_call_result_6087, result_is__6091)
        
        # Applying the binary operator 'or' (line 100)
        result_or_keyword_6093 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 49), 'or', result_not__6082, result_and_keyword_6092)
        
        # Applying the binary operator 'and' (line 100)
        result_and_keyword_6094 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'and', result_is_not_6076, result_or_keyword_6093)
        
        # Testing if the type of an if condition is none (line 100)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 12), result_and_keyword_6094):
            pass
        else:
            
            # Testing the type of an if condition (line 100)
            if_condition_6095 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_and_keyword_6094)
            # Assigning a type to the variable 'if_condition_6095' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_6095', if_condition_6095)
            # SSA begins for if statement (line 100)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 102):
            
            # Obtaining an instance of the builtin type 'list' (line 102)
            list_6096 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 102)
            # Adding element type (line 102)
            # Getting the type of 'self' (line 102)
            self_6097 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'self')
            # Obtaining the member 'fake_self' of a type (line 102)
            fake_self_6098 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 30), self_6097, 'fake_self')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 29), list_6096, fake_self_6098)
            
            # Getting the type of 'arg_values' (line 102)
            arg_values_6099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 48), 'arg_values')
            # Applying the binary operator '+' (line 102)
            result_add_6100 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 29), '+', list_6096, arg_values_6099)
            
            # Assigning a type to the variable 'arg_values' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'arg_values', result_add_6100)
            # SSA join for if statement (line 100)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 105):
        
        # Call to callable_python_entity(...): (line 105)
        # Getting the type of 'arg_values' (line 105)
        arg_values_6102 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 50), 'arg_values', False)
        # Processing the call keyword arguments (line 105)
        # Getting the type of 'kwargs_values' (line 105)
        kwargs_values_6103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 64), 'kwargs_values', False)
        kwargs_6104 = {'kwargs_values_6103': kwargs_values_6103}
        # Getting the type of 'callable_python_entity' (line 105)
        callable_python_entity_6101 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'callable_python_entity', False)
        # Calling callable_python_entity(args, kwargs) (line 105)
        callable_python_entity_call_result_6105 = invoke(stypy.reporting.localization.Localization(__file__, 105, 26), callable_python_entity_6101, *[arg_values_6102], **kwargs_6104)
        
        # Assigning a type to the variable 'call_result' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'call_result', callable_python_entity_call_result_6105)
        
        # Type idiom detected: calculating its left and rigth part (line 108)
        # Getting the type of 'call_result' (line 108)
        call_result_6106 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_result')
        # Getting the type of 'None' (line 108)
        None_6107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'None')
        
        (may_be_6108, more_types_in_union_6109) = may_not_be_none(call_result_6106, None_6107)

        if may_be_6108:

            if more_types_in_union_6109:
                # Runtime conditional SSA (line 108)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to isclass(...): (line 109)
            # Processing the call arguments (line 109)
            # Getting the type of 'callable_entity' (line 109)
            callable_entity_6112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 39), 'callable_entity', False)
            # Processing the call keyword arguments (line 109)
            kwargs_6113 = {}
            # Getting the type of 'inspect' (line 109)
            inspect_6110 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 109)
            isclass_6111 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 23), inspect_6110, 'isclass')
            # Calling isclass(args, kwargs) (line 109)
            isclass_call_result_6114 = invoke(stypy.reporting.localization.Localization(__file__, 109, 23), isclass_6111, *[callable_entity_6112], **kwargs_6113)
            
            # Applying the 'not' unary operator (line 109)
            result_not__6115 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 19), 'not', isclass_call_result_6114)
            
            # Testing if the type of an if condition is none (line 109)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 16), result_not__6115):
                pass
            else:
                
                # Testing the type of an if condition (line 109)
                if_condition_6116 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 16), result_not__6115)
                # Assigning a type to the variable 'if_condition_6116' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'if_condition_6116', if_condition_6116)
                # SSA begins for if statement (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to type(...): (line 110)
                # Processing the call arguments (line 110)
                # Getting the type of 'call_result' (line 110)
                call_result_6118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'call_result', False)
                # Processing the call keyword arguments (line 110)
                kwargs_6119 = {}
                # Getting the type of 'type' (line 110)
                type_6117 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'type', False)
                # Calling type(args, kwargs) (line 110)
                type_call_result_6120 = invoke(stypy.reporting.localization.Localization(__file__, 110, 27), type_6117, *[call_result_6118], **kwargs_6119)
                
                # Assigning a type to the variable 'stypy_return_type' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'stypy_return_type', type_call_result_6120)
                # SSA join for if statement (line 109)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to isinstance(...): (line 112)
            # Processing the call arguments (line 112)
            
            # Call to type(...): (line 112)
            # Processing the call arguments (line 112)
            # Getting the type of 'call_result' (line 112)
            call_result_6123 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'call_result', False)
            # Processing the call keyword arguments (line 112)
            kwargs_6124 = {}
            # Getting the type of 'type' (line 112)
            type_6122 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'type', False)
            # Calling type(args, kwargs) (line 112)
            type_call_result_6125 = invoke(stypy.reporting.localization.Localization(__file__, 112, 30), type_6122, *[call_result_6123], **kwargs_6124)
            
            # Obtaining the member '__dict__' of a type (line 112)
            dict___6126 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 30), type_call_result_6125, '__dict__')
            # Getting the type of 'types' (line 112)
            types_6127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 58), 'types', False)
            # Obtaining the member 'DictProxyType' of a type (line 112)
            DictProxyType_6128 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 58), types_6127, 'DictProxyType')
            # Processing the call keyword arguments (line 112)
            kwargs_6129 = {}
            # Getting the type of 'isinstance' (line 112)
            isinstance_6121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 112)
            isinstance_call_result_6130 = invoke(stypy.reporting.localization.Localization(__file__, 112, 19), isinstance_6121, *[dict___6126, DictProxyType_6128], **kwargs_6129)
            
            # Testing if the type of an if condition is none (line 112)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 16), isinstance_call_result_6130):
                pass
            else:
                
                # Testing the type of an if condition (line 112)
                if_condition_6131 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), isinstance_call_result_6130)
                # Assigning a type to the variable 'if_condition_6131' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_6131', if_condition_6131)
                # SSA begins for if statement (line 112)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Type idiom detected: calculating its left and rigth part (line 113)
                str_6132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'str', '__dict__')
                # Getting the type of 'call_result' (line 113)
                call_result_6133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'call_result')
                
                (may_be_6134, more_types_in_union_6135) = may_provide_member(str_6132, call_result_6133)

                if may_be_6134:

                    if more_types_in_union_6135:
                        # Runtime conditional SSA (line 113)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'call_result' (line 113)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'call_result', remove_not_member_provider_from_union(call_result_6133, '__dict__'))
                    
                    
                    # Call to isinstance(...): (line 114)
                    # Processing the call arguments (line 114)
                    # Getting the type of 'call_result' (line 114)
                    call_result_6137 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 42), 'call_result', False)
                    # Obtaining the member '__dict__' of a type (line 114)
                    dict___6138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 42), call_result_6137, '__dict__')
                    # Getting the type of 'types' (line 114)
                    types_6139 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 64), 'types', False)
                    # Obtaining the member 'DictProxyType' of a type (line 114)
                    DictProxyType_6140 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 64), types_6139, 'DictProxyType')
                    # Processing the call keyword arguments (line 114)
                    kwargs_6141 = {}
                    # Getting the type of 'isinstance' (line 114)
                    isinstance_6136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 31), 'isinstance', False)
                    # Calling isinstance(args, kwargs) (line 114)
                    isinstance_call_result_6142 = invoke(stypy.reporting.localization.Localization(__file__, 114, 31), isinstance_6136, *[dict___6138, DictProxyType_6140], **kwargs_6141)
                    
                    # Applying the 'not' unary operator (line 114)
                    result_not__6143 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 27), 'not', isinstance_call_result_6142)
                    
                    # Testing if the type of an if condition is none (line 114)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 114, 24), result_not__6143):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 114)
                        if_condition_6144 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 24), result_not__6143)
                        # Assigning a type to the variable 'if_condition_6144' (line 114)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'if_condition_6144', if_condition_6144)
                        # SSA begins for if statement (line 114)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'call_result' (line 115)
                        call_result_6145 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'call_result')
                        # Assigning a type to the variable 'stypy_return_type' (line 115)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'stypy_return_type', call_result_6145)
                        # SSA join for if statement (line 114)
                        module_type_store = module_type_store.join_ssa_context()
                        


                    if more_types_in_union_6135:
                        # SSA join for if statement (line 113)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Call to type(...): (line 117)
                # Processing the call arguments (line 117)
                # Getting the type of 'call_result' (line 117)
                call_result_6147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'call_result', False)
                # Processing the call keyword arguments (line 117)
                kwargs_6148 = {}
                # Getting the type of 'type' (line 117)
                type_6146 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'type', False)
                # Calling type(args, kwargs) (line 117)
                type_call_result_6149 = invoke(stypy.reporting.localization.Localization(__file__, 117, 27), type_6146, *[call_result_6147], **kwargs_6148)
                
                # Assigning a type to the variable 'stypy_return_type' (line 117)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'stypy_return_type', type_call_result_6149)
                # SSA join for if statement (line 112)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_6109:
                # SSA join for if statement (line 108)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'call_result' (line 119)
        call_result_6150 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'call_result')
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'stypy_return_type', call_result_6150)
        # SSA branch for the except part of a try statement (line 99)
        # SSA branch for the except 'Exception' branch of a try statement (line 99)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 120)
        Exception_6151 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'Exception')
        # Assigning a type to the variable 'ex' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'ex', Exception_6151)
        
        # Assigning a Call to a Name (line 121):
        
        # Call to format_call(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'callable_entity' (line 121)
        callable_entity_6153 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'callable_entity', False)
        # Getting the type of 'arg_types' (line 121)
        arg_types_6154 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 52), 'arg_types', False)
        # Getting the type of 'kwargs_types' (line 121)
        kwargs_types_6155 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 63), 'kwargs_types', False)
        # Processing the call keyword arguments (line 121)
        kwargs_6156 = {}
        # Getting the type of 'format_call' (line 121)
        format_call_6152 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'format_call', False)
        # Calling format_call(args, kwargs) (line 121)
        format_call_call_result_6157 = invoke(stypy.reporting.localization.Localization(__file__, 121, 23), format_call_6152, *[callable_entity_6153, arg_types_6154, kwargs_types_6155], **kwargs_6156)
        
        # Assigning a type to the variable 'str_call' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'str_call', format_call_call_result_6157)
        
        # Call to TypeError(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'localization' (line 122)
        localization_6159 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'localization', False)
        
        # Call to format(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'str_call' (line 122)
        str_call_6162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'str_call', False)
        
        # Call to str(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'ex' (line 122)
        ex_6164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 75), 'ex', False)
        # Processing the call keyword arguments (line 122)
        kwargs_6165 = {}
        # Getting the type of 'str' (line 122)
        str_6163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 71), 'str', False)
        # Calling str(args, kwargs) (line 122)
        str_call_result_6166 = invoke(stypy.reporting.localization.Localization(__file__, 122, 71), str_6163, *[ex_6164], **kwargs_6165)
        
        # Processing the call keyword arguments (line 122)
        kwargs_6167 = {}
        str_6160 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 43), 'str', '{0}: {1}')
        # Obtaining the member 'format' of a type (line 122)
        format_6161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 43), str_6160, 'format')
        # Calling format(args, kwargs) (line 122)
        format_call_result_6168 = invoke(stypy.reporting.localization.Localization(__file__, 122, 43), format_6161, *[str_call_6162, str_call_result_6166], **kwargs_6167)
        
        # Processing the call keyword arguments (line 122)
        kwargs_6169 = {}
        # Getting the type of 'TypeError' (line 122)
        TypeError_6158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 122)
        TypeError_call_result_6170 = invoke(stypy.reporting.localization.Localization(__file__, 122, 19), TypeError_6158, *[localization_6159, format_call_result_6168], **kwargs_6169)
        
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'stypy_return_type', TypeError_call_result_6170)
        # SSA join for try-except statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_6171 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_6171)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_6171


# Assigning a type to the variable 'FakeParamValuesCallHandler' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'FakeParamValuesCallHandler', FakeParamValuesCallHandler)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
