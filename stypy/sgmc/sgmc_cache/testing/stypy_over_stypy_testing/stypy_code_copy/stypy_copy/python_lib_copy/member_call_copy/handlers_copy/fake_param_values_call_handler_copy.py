
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: import types
3: 
4: from stypy_copy.errors_copy.type_error_copy import TypeError
5: from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy import get_type_sample_value
6: from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
7: from call_handler_copy import CallHandler
8: from stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy import format_call
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

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5682 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_5682) is not StypyTypeError):

    if (import_5682 != 'pyd_module'):
        __import__(import_5682)
        sys_modules_5683 = sys.modules[import_5682]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_5683.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_5683, sys_modules_5683.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'stypy_copy.errors_copy.type_error_copy', import_5682)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy import get_type_sample_value' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5684 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy')

if (type(import_5684) is not StypyTypeError):

    if (import_5684 != 'pyd_module'):
        __import__(import_5684)
        sys_modules_5685 = sys.modules[import_5684]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy', sys_modules_5685.module_type_store, module_type_store, ['get_type_sample_value'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_5685, sys_modules_5685.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy import get_type_sample_value

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy', None, module_type_store, ['get_type_sample_value'], [get_type_sample_value])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.instantiation_copy.type_instantiation_copy', import_5684)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5686 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy')

if (type(import_5686) is not StypyTypeError):

    if (import_5686 != 'pyd_module'):
        __import__(import_5686)
        sys_modules_5687 = sys.modules[import_5686]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', sys_modules_5687.module_type_store, module_type_store, ['Type'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_5687, sys_modules_5687.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', None, module_type_store, ['Type'], [Type])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_copy', import_5686)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 7, 0))

# 'from call_handler_copy import CallHandler' statement (line 7)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5688 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_handler_copy')

if (type(import_5688) is not StypyTypeError):

    if (import_5688 != 'pyd_module'):
        __import__(import_5688)
        sys_modules_5689 = sys.modules[import_5688]
        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_handler_copy', sys_modules_5689.module_type_store, module_type_store, ['CallHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 7, 0), __file__, sys_modules_5689, sys_modules_5689.module_type_store, module_type_store)
    else:
        from call_handler_copy import CallHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_handler_copy', None, module_type_store, ['CallHandler'], [CallHandler])

else:
    # Assigning a type to the variable 'call_handler_copy' (line 7)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 7, 0), 'call_handler_copy', import_5688)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy import format_call' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_5690 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy')

if (type(import_5690) is not StypyTypeError):

    if (import_5690 != 'pyd_module'):
        __import__(import_5690)
        sys_modules_5691 = sys.modules[import_5690]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy', sys_modules_5691.module_type_store, module_type_store, ['format_call'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_5691, sys_modules_5691.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy import format_call

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy', None, module_type_store, ['format_call'], [format_call])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.member_call_copy.call_handlers_helper_methods_copy', import_5690)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

# Declaration of the 'FakeParamValuesCallHandler' class
# Getting the type of 'CallHandler' (line 11)
CallHandler_5692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 33), 'CallHandler')

class FakeParamValuesCallHandler(CallHandler_5692, ):
    str_5693 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 18, (-1)), 'str', '\n    This handler simulated the call to a callable entity by creating fake values of the passed parameters, actually\n    call the callable entity and returned the type of the result. It is used when no other call handler can be\n    used to call this entity, meaning that this is third-party code or Python library modules with no source code\n    available to be transformed (and therefore using the UserCallablesCallHandler) or this module has no type rule\n    file associated (which can be done in the future, and stypy will use the TypeRuleCallHandler instead).\n    ')

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

        str_5694 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 26, (-1)), 'str', '\n        Obtain a fake value for the type represented by arg\n        :param arg: Type\n        :return: Value for that type\n        ')
        
        # Call to isinstance(...): (line 27)
        # Processing the call arguments (line 27)
        # Getting the type of 'arg' (line 27)
        arg_5696 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 22), 'arg', False)
        # Getting the type of 'Type' (line 27)
        Type_5697 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 27), 'Type', False)
        # Processing the call keyword arguments (line 27)
        kwargs_5698 = {}
        # Getting the type of 'isinstance' (line 27)
        isinstance_5695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 11), 'isinstance', False)
        # Calling isinstance(args, kwargs) (line 27)
        isinstance_call_result_5699 = invoke(stypy.reporting.localization.Localization(__file__, 27, 11), isinstance_5695, *[arg_5696, Type_5697], **kwargs_5698)
        
        # Testing if the type of an if condition is none (line 27)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 27, 8), isinstance_call_result_5699):
            pass
        else:
            
            # Testing the type of an if condition (line 27)
            if_condition_5700 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 27, 8), isinstance_call_result_5699)
            # Assigning a type to the variable 'if_condition_5700' (line 27)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 27, 8), 'if_condition_5700', if_condition_5700)
            # SSA begins for if statement (line 27)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 29):
            
            # Call to get_instance(...): (line 29)
            # Processing the call keyword arguments (line 29)
            kwargs_5703 = {}
            # Getting the type of 'arg' (line 29)
            arg_5701 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 29, 23), 'arg', False)
            # Obtaining the member 'get_instance' of a type (line 29)
            get_instance_5702 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 29, 23), arg_5701, 'get_instance')
            # Calling get_instance(args, kwargs) (line 29)
            get_instance_call_result_5704 = invoke(stypy.reporting.localization.Localization(__file__, 29, 23), get_instance_5702, *[], **kwargs_5703)
            
            # Assigning a type to the variable 'instance' (line 29)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 29, 12), 'instance', get_instance_call_result_5704)
            
            # Type idiom detected: calculating its left and rigth part (line 30)
            # Getting the type of 'instance' (line 30)
            instance_5705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 12), 'instance')
            # Getting the type of 'None' (line 30)
            None_5706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 30, 31), 'None')
            
            (may_be_5707, more_types_in_union_5708) = may_not_be_none(instance_5705, None_5706)

            if may_be_5707:

                if more_types_in_union_5708:
                    # Runtime conditional SSA (line 30)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                else:
                    module_type_store = module_type_store

                # Getting the type of 'instance' (line 31)
                instance_5709 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 23), 'instance')
                # Assigning a type to the variable 'stypy_return_type' (line 31)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 16), 'stypy_return_type', instance_5709)

                if more_types_in_union_5708:
                    # Runtime conditional SSA for else branch (line 30)
                    module_type_store.open_ssa_branch('idiom else')



            if ((not may_be_5707) or more_types_in_union_5708):
                
                # Type idiom detected: calculating its left and rigth part (line 34)
                str_5710 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 32), 'str', 'has_value')
                # Getting the type of 'arg' (line 34)
                arg_5711 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 27), 'arg')
                
                (may_be_5712, more_types_in_union_5713) = may_provide_member(str_5710, arg_5711)

                if may_be_5712:

                    if more_types_in_union_5713:
                        # Runtime conditional SSA (line 34)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'arg' (line 34)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'arg', remove_not_member_provider_from_union(arg_5711, 'has_value'))
                    
                    # Call to has_value(...): (line 35)
                    # Processing the call keyword arguments (line 35)
                    kwargs_5716 = {}
                    # Getting the type of 'arg' (line 35)
                    arg_5714 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'arg', False)
                    # Obtaining the member 'has_value' of a type (line 35)
                    has_value_5715 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), arg_5714, 'has_value')
                    # Calling has_value(args, kwargs) (line 35)
                    has_value_call_result_5717 = invoke(stypy.reporting.localization.Localization(__file__, 35, 23), has_value_5715, *[], **kwargs_5716)
                    
                    # Testing if the type of an if condition is none (line 35)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 35, 20), has_value_call_result_5717):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 35)
                        if_condition_5718 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 35, 20), has_value_call_result_5717)
                        # Assigning a type to the variable 'if_condition_5718' (line 35)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 20), 'if_condition_5718', if_condition_5718)
                        # SSA begins for if statement (line 35)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to get_value(...): (line 36)
                        # Processing the call keyword arguments (line 36)
                        kwargs_5721 = {}
                        # Getting the type of 'arg' (line 36)
                        arg_5719 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 31), 'arg', False)
                        # Obtaining the member 'get_value' of a type (line 36)
                        get_value_5720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 31), arg_5719, 'get_value')
                        # Calling get_value(args, kwargs) (line 36)
                        get_value_call_result_5722 = invoke(stypy.reporting.localization.Localization(__file__, 36, 31), get_value_5720, *[], **kwargs_5721)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 36)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 36, 24), 'stypy_return_type', get_value_call_result_5722)
                        # SSA join for if statement (line 35)
                        module_type_store = module_type_store.join_ssa_context()
                        


                    if more_types_in_union_5713:
                        # SSA join for if statement (line 34)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Call to get_type_sample_value(...): (line 39)
                # Processing the call arguments (line 39)
                
                # Call to get_python_type(...): (line 39)
                # Processing the call keyword arguments (line 39)
                kwargs_5726 = {}
                # Getting the type of 'arg' (line 39)
                arg_5724 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 45), 'arg', False)
                # Obtaining the member 'get_python_type' of a type (line 39)
                get_python_type_5725 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 45), arg_5724, 'get_python_type')
                # Calling get_python_type(args, kwargs) (line 39)
                get_python_type_call_result_5727 = invoke(stypy.reporting.localization.Localization(__file__, 39, 45), get_python_type_5725, *[], **kwargs_5726)
                
                # Processing the call keyword arguments (line 39)
                kwargs_5728 = {}
                # Getting the type of 'get_type_sample_value' (line 39)
                get_type_sample_value_5723 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 23), 'get_type_sample_value', False)
                # Calling get_type_sample_value(args, kwargs) (line 39)
                get_type_sample_value_call_result_5729 = invoke(stypy.reporting.localization.Localization(__file__, 39, 23), get_type_sample_value_5723, *[get_python_type_call_result_5727], **kwargs_5728)
                
                # Assigning a type to the variable 'stypy_return_type' (line 39)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 39, 16), 'stypy_return_type', get_type_sample_value_call_result_5729)

                if (may_be_5707 and more_types_in_union_5708):
                    # SSA join for if statement (line 30)
                    module_type_store = module_type_store.join_ssa_context()


            
            # SSA join for if statement (line 27)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to get_type_sample_value(...): (line 42)
        # Processing the call arguments (line 42)
        # Getting the type of 'arg' (line 42)
        arg_5731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 37), 'arg', False)
        # Processing the call keyword arguments (line 42)
        kwargs_5732 = {}
        # Getting the type of 'get_type_sample_value' (line 42)
        get_type_sample_value_5730 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 42, 15), 'get_type_sample_value', False)
        # Calling get_type_sample_value(args, kwargs) (line 42)
        get_type_sample_value_call_result_5733 = invoke(stypy.reporting.localization.Localization(__file__, 42, 15), get_type_sample_value_5730, *[arg_5731], **kwargs_5732)
        
        # Assigning a type to the variable 'stypy_return_type' (line 42)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 42, 8), 'stypy_return_type', get_type_sample_value_call_result_5733)
        
        # ################# End of '__get_type_instance(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_type_instance' in the type store
        # Getting the type of 'stypy_return_type' (line 20)
        stypy_return_type_5734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5734)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_type_instance'
        return stypy_return_type_5734


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

        str_5735 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 50, (-1)), 'str', '\n        Obtain a fake value for all the types passed in a list\n        :param arg_types: List of types\n        :return: List of values\n        ')
        
        # Call to map(...): (line 51)
        # Processing the call arguments (line 51)

        @norecursion
        def _stypy_temp_lambda_12(localization, *varargs, **kwargs):
            global module_type_store
            # Assign values to the parameters with defaults
            defaults = []
            # Create a new context for function '_stypy_temp_lambda_12'
            module_type_store = module_type_store.open_function_context('_stypy_temp_lambda_12', 51, 19, True)
            # Passed parameters checking function
            _stypy_temp_lambda_12.stypy_localization = localization
            _stypy_temp_lambda_12.stypy_type_of_self = None
            _stypy_temp_lambda_12.stypy_type_store = module_type_store
            _stypy_temp_lambda_12.stypy_function_name = '_stypy_temp_lambda_12'
            _stypy_temp_lambda_12.stypy_param_names_list = ['arg']
            _stypy_temp_lambda_12.stypy_varargs_param_name = None
            _stypy_temp_lambda_12.stypy_kwargs_param_name = None
            _stypy_temp_lambda_12.stypy_call_defaults = defaults
            _stypy_temp_lambda_12.stypy_call_varargs = varargs
            _stypy_temp_lambda_12.stypy_call_kwargs = kwargs
            arguments = process_argument_values(localization, None, module_type_store, '_stypy_temp_lambda_12', ['arg'], None, None, defaults, varargs, kwargs)

            if is_error_type(arguments):
                # Destroy the current context
                module_type_store = module_type_store.close_function_context()
                return arguments

            # Stacktrace push for error reporting
            localization.set_stack_trace('_stypy_temp_lambda_12', ['arg'], arguments)
            # Default return type storage variable (SSA)
            # Assigning a type to the variable 'stypy_return_type'
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
            
            
            # ################# Begin of the lambda function code ##################

            
            # Call to __get_type_instance(...): (line 51)
            # Processing the call arguments (line 51)
            # Getting the type of 'arg' (line 51)
            arg_5739 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 78), 'arg', False)
            # Processing the call keyword arguments (line 51)
            kwargs_5740 = {}
            # Getting the type of 'FakeParamValuesCallHandler' (line 51)
            FakeParamValuesCallHandler_5737 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 31), 'FakeParamValuesCallHandler', False)
            # Obtaining the member '__get_type_instance' of a type (line 51)
            get_type_instance_5738 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 51, 31), FakeParamValuesCallHandler_5737, '__get_type_instance')
            # Calling __get_type_instance(args, kwargs) (line 51)
            get_type_instance_call_result_5741 = invoke(stypy.reporting.localization.Localization(__file__, 51, 31), get_type_instance_5738, *[arg_5739], **kwargs_5740)
            
            # Assigning the return type of the lambda function
            # Assigning a type to the variable 'stypy_return_type' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'stypy_return_type', get_type_instance_call_result_5741)
            
            # ################# End of the lambda function code ##################

            # Stacktrace pop (error reporting)
            localization.unset_stack_trace()
            
            # Storing the return type of function '_stypy_temp_lambda_12' in the type store
            # Getting the type of 'stypy_return_type' (line 51)
            stypy_return_type_5742 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), 'stypy_return_type')
            module_type_store.store_return_type_of_current_context(stypy_return_type_5742)
            
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            
            # Return type of the function '_stypy_temp_lambda_12'
            return stypy_return_type_5742

        # Assigning a type to the variable '_stypy_temp_lambda_12' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), '_stypy_temp_lambda_12', _stypy_temp_lambda_12)
        # Getting the type of '_stypy_temp_lambda_12' (line 51)
        _stypy_temp_lambda_12_5743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 19), '_stypy_temp_lambda_12')
        # Getting the type of 'arg_types' (line 51)
        arg_types_5744 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 84), 'arg_types', False)
        # Processing the call keyword arguments (line 51)
        kwargs_5745 = {}
        # Getting the type of 'map' (line 51)
        map_5736 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 15), 'map', False)
        # Calling map(args, kwargs) (line 51)
        map_call_result_5746 = invoke(stypy.reporting.localization.Localization(__file__, 51, 15), map_5736, *[_stypy_temp_lambda_12_5743, arg_types_5744], **kwargs_5745)
        
        # Assigning a type to the variable 'stypy_return_type' (line 51)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 8), 'stypy_return_type', map_call_result_5746)
        
        # ################# End of '__get_arg_sample_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_arg_sample_values' in the type store
        # Getting the type of 'stypy_return_type' (line 44)
        stypy_return_type_5747 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5747)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_arg_sample_values'
        return stypy_return_type_5747


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

        str_5748 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, (-1)), 'str', '\n        Obtain a fake value for all the types passed on a dict. This is used for keyword arguments\n        :param kwargs_types: Dict of types\n        :return: Dict of values\n        ')
        
        # Assigning a Dict to a Name (line 60):
        
        # Obtaining an instance of the builtin type 'dict' (line 60)
        dict_5749 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 24), 'dict')
        # Adding type elements to the builtin type 'dict' instance (line 60)
        
        # Assigning a type to the variable 'kwargs_values' (line 60)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 8), 'kwargs_values', dict_5749)
        
        # Getting the type of 'kwargs_types' (line 61)
        kwargs_types_5750 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 21), 'kwargs_types')
        # Assigning a type to the variable 'kwargs_types_5750' (line 61)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'kwargs_types_5750', kwargs_types_5750)
        # Testing if the for loop is going to be iterated (line 61)
        # Testing the type of a for loop iterable (line 61)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 61, 8), kwargs_types_5750)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 61, 8), kwargs_types_5750):
            # Getting the type of the for loop variable (line 61)
            for_loop_var_5751 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 61, 8), kwargs_types_5750)
            # Assigning a type to the variable 'value' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 8), 'value', for_loop_var_5751)
            # SSA begins for a for statement (line 61)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Subscript (line 62):
            
            # Call to __get_type_instance(...): (line 62)
            # Processing the call arguments (line 62)
            
            # Obtaining the type of the subscript
            # Getting the type of 'value' (line 62)
            value_5754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 95), 'value', False)
            # Getting the type of 'kwargs_types' (line 62)
            kwargs_types_5755 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 82), 'kwargs_types', False)
            # Obtaining the member '__getitem__' of a type (line 62)
            getitem___5756 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 82), kwargs_types_5755, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 62)
            subscript_call_result_5757 = invoke(stypy.reporting.localization.Localization(__file__, 62, 82), getitem___5756, value_5754)
            
            # Processing the call keyword arguments (line 62)
            kwargs_5758 = {}
            # Getting the type of 'FakeParamValuesCallHandler' (line 62)
            FakeParamValuesCallHandler_5752 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 35), 'FakeParamValuesCallHandler', False)
            # Obtaining the member '__get_type_instance' of a type (line 62)
            get_type_instance_5753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 35), FakeParamValuesCallHandler_5752, '__get_type_instance')
            # Calling __get_type_instance(args, kwargs) (line 62)
            get_type_instance_call_result_5759 = invoke(stypy.reporting.localization.Localization(__file__, 62, 35), get_type_instance_5753, *[subscript_call_result_5757], **kwargs_5758)
            
            # Getting the type of 'kwargs_values' (line 62)
            kwargs_values_5760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 12), 'kwargs_values')
            # Getting the type of 'value' (line 62)
            value_5761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 26), 'value')
            # Storing an element on a container (line 62)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 62, 12), kwargs_values_5760, (value_5761, get_type_instance_call_result_5759))
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'kwargs_values' (line 64)
        kwargs_values_5762 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 15), 'kwargs_values')
        # Assigning a type to the variable 'stypy_return_type' (line 64)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'stypy_return_type', kwargs_values_5762)
        
        # ################# End of '__get_kwarg_sample_values(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__get_kwarg_sample_values' in the type store
        # Getting the type of 'stypy_return_type' (line 53)
        stypy_return_type_5763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 53, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5763)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__get_kwarg_sample_values'
        return stypy_return_type_5763


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        # Getting the type of 'None' (line 66)
        None_5764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 33), 'None')
        defaults = [None_5764]
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
        self_5767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 29), 'self', False)
        # Processing the call keyword arguments (line 67)
        kwargs_5768 = {}
        # Getting the type of 'CallHandler' (line 67)
        CallHandler_5765 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 67, 8), 'CallHandler', False)
        # Obtaining the member '__init__' of a type (line 67)
        init___5766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 67, 8), CallHandler_5765, '__init__')
        # Calling __init__(args, kwargs) (line 67)
        init___call_result_5769 = invoke(stypy.reporting.localization.Localization(__file__, 67, 8), init___5766, *[self_5767], **kwargs_5768)
        
        
        # Assigning a Name to a Attribute (line 68):
        # Getting the type of 'fake_self' (line 68)
        fake_self_5770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 25), 'fake_self')
        # Getting the type of 'self' (line 68)
        self_5771 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'self')
        # Setting the type of the member 'fake_self' of a type (line 68)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 8), self_5771, 'fake_self', fake_self_5770)
        
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

        str_5772 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, (-1)), 'str', '\n        This method determines if this call handler is able to respond to a call to callable_entity. The call handler\n        respond to any callable code, as it is the last to be used.\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param callable_entity: Callable entity\n        :return: Always True\n        ')
        # Getting the type of 'True' (line 78)
        True_5773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 15), 'True')
        # Assigning a type to the variable 'stypy_return_type' (line 78)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 78, 8), 'stypy_return_type', True_5773)
        
        # ################# End of 'applies_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'applies_to' in the type store
        # Getting the type of 'stypy_return_type' (line 70)
        stypy_return_type_5774 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5774)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'applies_to'
        return stypy_return_type_5774


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

        str_5775 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 92, (-1)), 'str', '\n        This call handler substitutes the param types by fake values and perform a call to the real python callable\n        entity, returning the type of the return type if the called entity is not a class (in that case it returns the\n        created instance)\n\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param localization: Caller information\n        :param callable_entity: Callable entity\n        :param arg_types: Arguments\n        :param kwargs_types: Keyword arguments\n        :return: Return type of the call\n        ')
        
        # Assigning a Call to a Name (line 95):
        
        # Call to __get_arg_sample_values(...): (line 95)
        # Processing the call arguments (line 95)
        # Getting the type of 'arg_types' (line 95)
        arg_types_5778 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 72), 'arg_types', False)
        # Processing the call keyword arguments (line 95)
        kwargs_5779 = {}
        # Getting the type of 'FakeParamValuesCallHandler' (line 95)
        FakeParamValuesCallHandler_5776 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 21), 'FakeParamValuesCallHandler', False)
        # Obtaining the member '__get_arg_sample_values' of a type (line 95)
        get_arg_sample_values_5777 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 21), FakeParamValuesCallHandler_5776, '__get_arg_sample_values')
        # Calling __get_arg_sample_values(args, kwargs) (line 95)
        get_arg_sample_values_call_result_5780 = invoke(stypy.reporting.localization.Localization(__file__, 95, 21), get_arg_sample_values_5777, *[arg_types_5778], **kwargs_5779)
        
        # Assigning a type to the variable 'arg_values' (line 95)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 8), 'arg_values', get_arg_sample_values_call_result_5780)
        
        # Assigning a Call to a Name (line 96):
        
        # Call to __get_kwarg_sample_values(...): (line 96)
        # Processing the call arguments (line 96)
        # Getting the type of 'kwargs_types' (line 96)
        kwargs_types_5783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 77), 'kwargs_types', False)
        # Processing the call keyword arguments (line 96)
        kwargs_5784 = {}
        # Getting the type of 'FakeParamValuesCallHandler' (line 96)
        FakeParamValuesCallHandler_5781 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 24), 'FakeParamValuesCallHandler', False)
        # Obtaining the member '__get_kwarg_sample_values' of a type (line 96)
        get_kwarg_sample_values_5782 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 24), FakeParamValuesCallHandler_5781, '__get_kwarg_sample_values')
        # Calling __get_kwarg_sample_values(args, kwargs) (line 96)
        get_kwarg_sample_values_call_result_5785 = invoke(stypy.reporting.localization.Localization(__file__, 96, 24), get_kwarg_sample_values_5782, *[kwargs_types_5783], **kwargs_5784)
        
        # Assigning a type to the variable 'kwargs_values' (line 96)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 96, 8), 'kwargs_values', get_kwarg_sample_values_call_result_5785)
        
        # Assigning a Name to a Name (line 98):
        # Getting the type of 'callable_entity' (line 98)
        callable_entity_5786 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 98, 33), 'callable_entity')
        # Assigning a type to the variable 'callable_python_entity' (line 98)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 98, 8), 'callable_python_entity', callable_entity_5786)
        
        
        # SSA begins for try-except statement (line 99)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Evaluating a boolean operation
        
        # Getting the type of 'self' (line 100)
        self_5787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 16), 'self')
        # Obtaining the member 'fake_self' of a type (line 100)
        fake_self_5788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 16), self_5787, 'fake_self')
        # Getting the type of 'None' (line 100)
        None_5789 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 38), 'None')
        # Applying the binary operator 'isnot' (line 100)
        result_is_not_5790 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 16), 'isnot', fake_self_5788, None_5789)
        
        
        # Evaluating a boolean operation
        
        
        # Call to hasattr(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'callable_python_entity' (line 100)
        callable_python_entity_5792 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 62), 'callable_python_entity', False)
        str_5793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 86), 'str', '__self__')
        # Processing the call keyword arguments (line 100)
        kwargs_5794 = {}
        # Getting the type of 'hasattr' (line 100)
        hasattr_5791 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 54), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 100)
        hasattr_call_result_5795 = invoke(stypy.reporting.localization.Localization(__file__, 100, 54), hasattr_5791, *[callable_python_entity_5792, str_5793], **kwargs_5794)
        
        # Applying the 'not' unary operator (line 100)
        result_not__5796 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 50), 'not', hasattr_call_result_5795)
        
        
        # Evaluating a boolean operation
        
        # Call to hasattr(...): (line 101)
        # Processing the call arguments (line 101)
        # Getting the type of 'callable_python_entity' (line 101)
        callable_python_entity_5798 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 32), 'callable_python_entity', False)
        str_5799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 56), 'str', '__self__')
        # Processing the call keyword arguments (line 101)
        kwargs_5800 = {}
        # Getting the type of 'hasattr' (line 101)
        hasattr_5797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 24), 'hasattr', False)
        # Calling hasattr(args, kwargs) (line 101)
        hasattr_call_result_5801 = invoke(stypy.reporting.localization.Localization(__file__, 101, 24), hasattr_5797, *[callable_python_entity_5798, str_5799], **kwargs_5800)
        
        
        # Getting the type of 'callable_python_entity' (line 101)
        callable_python_entity_5802 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 72), 'callable_python_entity')
        # Obtaining the member '__self__' of a type (line 101)
        self___5803 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 72), callable_python_entity_5802, '__self__')
        # Getting the type of 'None' (line 101)
        None_5804 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 107), 'None')
        # Applying the binary operator 'is' (line 101)
        result_is__5805 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 72), 'is', self___5803, None_5804)
        
        # Applying the binary operator 'and' (line 101)
        result_and_keyword_5806 = python_operator(stypy.reporting.localization.Localization(__file__, 101, 24), 'and', hasattr_call_result_5801, result_is__5805)
        
        # Applying the binary operator 'or' (line 100)
        result_or_keyword_5807 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 49), 'or', result_not__5796, result_and_keyword_5806)
        
        # Applying the binary operator 'and' (line 100)
        result_and_keyword_5808 = python_operator(stypy.reporting.localization.Localization(__file__, 100, 15), 'and', result_is_not_5790, result_or_keyword_5807)
        
        # Testing if the type of an if condition is none (line 100)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 100, 12), result_and_keyword_5808):
            pass
        else:
            
            # Testing the type of an if condition (line 100)
            if_condition_5809 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 100, 12), result_and_keyword_5808)
            # Assigning a type to the variable 'if_condition_5809' (line 100)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 12), 'if_condition_5809', if_condition_5809)
            # SSA begins for if statement (line 100)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 102):
            
            # Obtaining an instance of the builtin type 'list' (line 102)
            list_5810 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 29), 'list')
            # Adding type elements to the builtin type 'list' instance (line 102)
            # Adding element type (line 102)
            # Getting the type of 'self' (line 102)
            self_5811 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 30), 'self')
            # Obtaining the member 'fake_self' of a type (line 102)
            fake_self_5812 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 30), self_5811, 'fake_self')
            add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 102, 29), list_5810, fake_self_5812)
            
            # Getting the type of 'arg_values' (line 102)
            arg_values_5813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 48), 'arg_values')
            # Applying the binary operator '+' (line 102)
            result_add_5814 = python_operator(stypy.reporting.localization.Localization(__file__, 102, 29), '+', list_5810, arg_values_5813)
            
            # Assigning a type to the variable 'arg_values' (line 102)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 16), 'arg_values', result_add_5814)
            # SSA join for if statement (line 100)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 105):
        
        # Call to callable_python_entity(...): (line 105)
        # Getting the type of 'arg_values' (line 105)
        arg_values_5816 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 50), 'arg_values', False)
        # Processing the call keyword arguments (line 105)
        # Getting the type of 'kwargs_values' (line 105)
        kwargs_values_5817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 64), 'kwargs_values', False)
        kwargs_5818 = {'kwargs_values_5817': kwargs_values_5817}
        # Getting the type of 'callable_python_entity' (line 105)
        callable_python_entity_5815 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 26), 'callable_python_entity', False)
        # Calling callable_python_entity(args, kwargs) (line 105)
        callable_python_entity_call_result_5819 = invoke(stypy.reporting.localization.Localization(__file__, 105, 26), callable_python_entity_5815, *[arg_values_5816], **kwargs_5818)
        
        # Assigning a type to the variable 'call_result' (line 105)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 12), 'call_result', callable_python_entity_call_result_5819)
        
        # Type idiom detected: calculating its left and rigth part (line 108)
        # Getting the type of 'call_result' (line 108)
        call_result_5820 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 12), 'call_result')
        # Getting the type of 'None' (line 108)
        None_5821 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 108, 34), 'None')
        
        (may_be_5822, more_types_in_union_5823) = may_not_be_none(call_result_5820, None_5821)

        if may_be_5822:

            if more_types_in_union_5823:
                # Runtime conditional SSA (line 108)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to isclass(...): (line 109)
            # Processing the call arguments (line 109)
            # Getting the type of 'callable_entity' (line 109)
            callable_entity_5826 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 39), 'callable_entity', False)
            # Processing the call keyword arguments (line 109)
            kwargs_5827 = {}
            # Getting the type of 'inspect' (line 109)
            inspect_5824 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 109, 23), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 109)
            isclass_5825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 109, 23), inspect_5824, 'isclass')
            # Calling isclass(args, kwargs) (line 109)
            isclass_call_result_5828 = invoke(stypy.reporting.localization.Localization(__file__, 109, 23), isclass_5825, *[callable_entity_5826], **kwargs_5827)
            
            # Applying the 'not' unary operator (line 109)
            result_not__5829 = python_operator(stypy.reporting.localization.Localization(__file__, 109, 19), 'not', isclass_call_result_5828)
            
            # Testing if the type of an if condition is none (line 109)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 109, 16), result_not__5829):
                pass
            else:
                
                # Testing the type of an if condition (line 109)
                if_condition_5830 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 109, 16), result_not__5829)
                # Assigning a type to the variable 'if_condition_5830' (line 109)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 109, 16), 'if_condition_5830', if_condition_5830)
                # SSA begins for if statement (line 109)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Call to type(...): (line 110)
                # Processing the call arguments (line 110)
                # Getting the type of 'call_result' (line 110)
                call_result_5832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 32), 'call_result', False)
                # Processing the call keyword arguments (line 110)
                kwargs_5833 = {}
                # Getting the type of 'type' (line 110)
                type_5831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'type', False)
                # Calling type(args, kwargs) (line 110)
                type_call_result_5834 = invoke(stypy.reporting.localization.Localization(__file__, 110, 27), type_5831, *[call_result_5832], **kwargs_5833)
                
                # Assigning a type to the variable 'stypy_return_type' (line 110)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 20), 'stypy_return_type', type_call_result_5834)
                # SSA join for if statement (line 109)
                module_type_store = module_type_store.join_ssa_context()
                

            
            # Call to isinstance(...): (line 112)
            # Processing the call arguments (line 112)
            
            # Call to type(...): (line 112)
            # Processing the call arguments (line 112)
            # Getting the type of 'call_result' (line 112)
            call_result_5837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 35), 'call_result', False)
            # Processing the call keyword arguments (line 112)
            kwargs_5838 = {}
            # Getting the type of 'type' (line 112)
            type_5836 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 30), 'type', False)
            # Calling type(args, kwargs) (line 112)
            type_call_result_5839 = invoke(stypy.reporting.localization.Localization(__file__, 112, 30), type_5836, *[call_result_5837], **kwargs_5838)
            
            # Obtaining the member '__dict__' of a type (line 112)
            dict___5840 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 30), type_call_result_5839, '__dict__')
            # Getting the type of 'types' (line 112)
            types_5841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 58), 'types', False)
            # Obtaining the member 'DictProxyType' of a type (line 112)
            DictProxyType_5842 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 58), types_5841, 'DictProxyType')
            # Processing the call keyword arguments (line 112)
            kwargs_5843 = {}
            # Getting the type of 'isinstance' (line 112)
            isinstance_5835 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 19), 'isinstance', False)
            # Calling isinstance(args, kwargs) (line 112)
            isinstance_call_result_5844 = invoke(stypy.reporting.localization.Localization(__file__, 112, 19), isinstance_5835, *[dict___5840, DictProxyType_5842], **kwargs_5843)
            
            # Testing if the type of an if condition is none (line 112)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 112, 16), isinstance_call_result_5844):
                pass
            else:
                
                # Testing the type of an if condition (line 112)
                if_condition_5845 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 112, 16), isinstance_call_result_5844)
                # Assigning a type to the variable 'if_condition_5845' (line 112)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 16), 'if_condition_5845', if_condition_5845)
                # SSA begins for if statement (line 112)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Type idiom detected: calculating its left and rigth part (line 113)
                str_5846 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 113, 44), 'str', '__dict__')
                # Getting the type of 'call_result' (line 113)
                call_result_5847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 113, 31), 'call_result')
                
                (may_be_5848, more_types_in_union_5849) = may_provide_member(str_5846, call_result_5847)

                if may_be_5848:

                    if more_types_in_union_5849:
                        # Runtime conditional SSA (line 113)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
                    else:
                        module_type_store = module_type_store

                    # Assigning a type to the variable 'call_result' (line 113)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 113, 20), 'call_result', remove_not_member_provider_from_union(call_result_5847, '__dict__'))
                    
                    
                    # Call to isinstance(...): (line 114)
                    # Processing the call arguments (line 114)
                    # Getting the type of 'call_result' (line 114)
                    call_result_5851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 42), 'call_result', False)
                    # Obtaining the member '__dict__' of a type (line 114)
                    dict___5852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 42), call_result_5851, '__dict__')
                    # Getting the type of 'types' (line 114)
                    types_5853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 64), 'types', False)
                    # Obtaining the member 'DictProxyType' of a type (line 114)
                    DictProxyType_5854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 64), types_5853, 'DictProxyType')
                    # Processing the call keyword arguments (line 114)
                    kwargs_5855 = {}
                    # Getting the type of 'isinstance' (line 114)
                    isinstance_5850 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 31), 'isinstance', False)
                    # Calling isinstance(args, kwargs) (line 114)
                    isinstance_call_result_5856 = invoke(stypy.reporting.localization.Localization(__file__, 114, 31), isinstance_5850, *[dict___5852, DictProxyType_5854], **kwargs_5855)
                    
                    # Applying the 'not' unary operator (line 114)
                    result_not__5857 = python_operator(stypy.reporting.localization.Localization(__file__, 114, 27), 'not', isinstance_call_result_5856)
                    
                    # Testing if the type of an if condition is none (line 114)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 114, 24), result_not__5857):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 114)
                        if_condition_5858 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 114, 24), result_not__5857)
                        # Assigning a type to the variable 'if_condition_5858' (line 114)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 114, 24), 'if_condition_5858', if_condition_5858)
                        # SSA begins for if statement (line 114)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        # Getting the type of 'call_result' (line 115)
                        call_result_5859 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 35), 'call_result')
                        # Assigning a type to the variable 'stypy_return_type' (line 115)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 28), 'stypy_return_type', call_result_5859)
                        # SSA join for if statement (line 114)
                        module_type_store = module_type_store.join_ssa_context()
                        


                    if more_types_in_union_5849:
                        # SSA join for if statement (line 113)
                        module_type_store = module_type_store.join_ssa_context()


                
                
                # Call to type(...): (line 117)
                # Processing the call arguments (line 117)
                # Getting the type of 'call_result' (line 117)
                call_result_5861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 32), 'call_result', False)
                # Processing the call keyword arguments (line 117)
                kwargs_5862 = {}
                # Getting the type of 'type' (line 117)
                type_5860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 117, 27), 'type', False)
                # Calling type(args, kwargs) (line 117)
                type_call_result_5863 = invoke(stypy.reporting.localization.Localization(__file__, 117, 27), type_5860, *[call_result_5861], **kwargs_5862)
                
                # Assigning a type to the variable 'stypy_return_type' (line 117)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 117, 20), 'stypy_return_type', type_call_result_5863)
                # SSA join for if statement (line 112)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_5823:
                # SSA join for if statement (line 108)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'call_result' (line 119)
        call_result_5864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 19), 'call_result')
        # Assigning a type to the variable 'stypy_return_type' (line 119)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 12), 'stypy_return_type', call_result_5864)
        # SSA branch for the except part of a try statement (line 99)
        # SSA branch for the except 'Exception' branch of a try statement (line 99)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 120)
        Exception_5865 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 15), 'Exception')
        # Assigning a type to the variable 'ex' (line 120)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 120, 8), 'ex', Exception_5865)
        
        # Assigning a Call to a Name (line 121):
        
        # Call to format_call(...): (line 121)
        # Processing the call arguments (line 121)
        # Getting the type of 'callable_entity' (line 121)
        callable_entity_5867 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 35), 'callable_entity', False)
        # Getting the type of 'arg_types' (line 121)
        arg_types_5868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 52), 'arg_types', False)
        # Getting the type of 'kwargs_types' (line 121)
        kwargs_types_5869 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 63), 'kwargs_types', False)
        # Processing the call keyword arguments (line 121)
        kwargs_5870 = {}
        # Getting the type of 'format_call' (line 121)
        format_call_5866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 121, 23), 'format_call', False)
        # Calling format_call(args, kwargs) (line 121)
        format_call_call_result_5871 = invoke(stypy.reporting.localization.Localization(__file__, 121, 23), format_call_5866, *[callable_entity_5867, arg_types_5868, kwargs_types_5869], **kwargs_5870)
        
        # Assigning a type to the variable 'str_call' (line 121)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 121, 12), 'str_call', format_call_call_result_5871)
        
        # Call to TypeError(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'localization' (line 122)
        localization_5873 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 29), 'localization', False)
        
        # Call to format(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'str_call' (line 122)
        str_call_5876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 61), 'str_call', False)
        
        # Call to str(...): (line 122)
        # Processing the call arguments (line 122)
        # Getting the type of 'ex' (line 122)
        ex_5878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 75), 'ex', False)
        # Processing the call keyword arguments (line 122)
        kwargs_5879 = {}
        # Getting the type of 'str' (line 122)
        str_5877 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 71), 'str', False)
        # Calling str(args, kwargs) (line 122)
        str_call_result_5880 = invoke(stypy.reporting.localization.Localization(__file__, 122, 71), str_5877, *[ex_5878], **kwargs_5879)
        
        # Processing the call keyword arguments (line 122)
        kwargs_5881 = {}
        str_5874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 122, 43), 'str', '{0}: {1}')
        # Obtaining the member 'format' of a type (line 122)
        format_5875 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 122, 43), str_5874, 'format')
        # Calling format(args, kwargs) (line 122)
        format_call_result_5882 = invoke(stypy.reporting.localization.Localization(__file__, 122, 43), format_5875, *[str_call_5876, str_call_result_5880], **kwargs_5881)
        
        # Processing the call keyword arguments (line 122)
        kwargs_5883 = {}
        # Getting the type of 'TypeError' (line 122)
        TypeError_5872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 122, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 122)
        TypeError_call_result_5884 = invoke(stypy.reporting.localization.Localization(__file__, 122, 19), TypeError_5872, *[localization_5873, format_call_result_5882], **kwargs_5883)
        
        # Assigning a type to the variable 'stypy_return_type' (line 122)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 122, 12), 'stypy_return_type', TypeError_call_result_5884)
        # SSA join for try-except statement (line 99)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 80)
        stypy_return_type_5885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_5885)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_5885


# Assigning a type to the variable 'FakeParamValuesCallHandler' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'FakeParamValuesCallHandler', FakeParamValuesCallHandler)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
