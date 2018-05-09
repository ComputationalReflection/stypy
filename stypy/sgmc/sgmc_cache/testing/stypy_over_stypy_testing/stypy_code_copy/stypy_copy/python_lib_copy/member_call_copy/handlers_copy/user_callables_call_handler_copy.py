
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: 
3: from stypy_copy.errors_copy.type_error_copy import TypeError
4: from call_handler_copy import CallHandler
5: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import no_recursion_copy
6: from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy import is_user_defined_class, \
7:     is_user_defined_module
8: from stypy_copy.python_lib_copy.python_types_copy import type_inference_copy
9: 
10: 
11: class UserCallablesCallHandler(CallHandler):
12:     '''
13:     This class handles calls to code that have been obtained from available .py source files that the program uses,
14:     generated to perform type inference of the original callable code. Remember that all functions in type inference
15:     programs are transformed to the following form:
16: 
17:     def function_name(*args, **kwargs):
18:         ...
19: 
20:     This handler just use this convention to call the code and return the result. Transformed code can handle
21:      UnionTypes and other stypy constructs.
22:     '''
23: 
24:     def applies_to(self, proxy_obj, callable_entity):
25:         '''
26:         This method determines if this call handler is able to respond to a call to callable_entity.
27:         :param proxy_obj: TypeInferenceProxy that hold the callable entity
28:         :param callable_entity: Callable entity
29:         :return: bool
30:         '''
31:         return is_user_defined_class(callable_entity) or proxy_obj.parent_proxy.python_entity == no_recursion_copy or \
32:                (inspect.ismethod(callable_entity) and is_user_defined_class(proxy_obj.parent_proxy.python_entity)) or \
33:                (inspect.isfunction(callable_entity) and is_user_defined_class(proxy_obj.parent_proxy.python_entity)) or \
34:                (inspect.isfunction(callable_entity) and is_user_defined_module(callable_entity.__module__))
35: 
36:     def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
37:         '''
38:         Perform the call and return the call result
39:         :param proxy_obj: TypeInferenceProxy that hold the callable entity
40:         :param localization: Caller information
41:         :param callable_entity: Callable entity
42:         :param arg_types: Arguments
43:         :param kwargs_types: Keyword arguments
44:         :return: Return type of the call
45:         '''
46:         callable_python_entity = callable_entity
47:         try:
48:             pre_errors = TypeError.get_error_msgs()
49:             # None localizations are used to indicate that the callable entity do not use this parameter
50:             if localization is None:
51:                 call_result = callable_python_entity(*arg_types, **kwargs_types)
52:             else:
53:                 # static method
54:                 if inspect.isfunction(proxy_obj.python_entity) and inspect.isclass(
55:                         proxy_obj.parent_proxy.python_entity):
56:                     call_result = callable_python_entity(proxy_obj.parent_proxy.python_entity,
57:                                                          localization, *arg_types, **kwargs_types)
58:                 else:
59:                     # Explicit call to __init__ functions (super)
60:                     if inspect.ismethod(proxy_obj.python_entity) and ".__init__" in proxy_obj.name and inspect.isclass(
61:                         proxy_obj.parent_proxy.python_entity):
62:                         call_result = callable_python_entity(arg_types[0].python_entity,
63:                                      localization, *arg_types, **kwargs_types)
64:                     else:
65:                         # instance method
66:                         call_result = callable_python_entity(localization, *arg_types, **kwargs_types)
67: 
68:             # Return type
69:             if call_result is not None:
70:                 if not inspect.isclass(callable_entity):
71:                     return call_result
72:                 else:
73:                     # Are we creating an instance of a classs?
74:                     if isinstance(call_result, callable_entity):
75:                         post_errors = TypeError.get_error_msgs()
76:                         if not len(pre_errors) == len(post_errors):
77:                             return TypeError(localization, "Could not create an instance of ({0}) due to constructor"
78:                                                            " call errors".format(callable_entity), False)
79: 
80:                     instance = call_result
81:                     type_ = callable_entity
82:                     return type_inference_copy.type_inference_proxy.TypeInferenceProxy.instance(type_, instance=instance)
83:             return call_result
84:         except Exception as ex:
85:             return TypeError(localization,
86:                              "The attempted call to '{0}' cannot be possible with parameter types {1}: {2}".format(
87:                                  callable_entity, list(arg_types) + list(kwargs_types.values()), str(ex)))
88: 

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

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'from stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7035 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy')

if (type(import_7035) is not StypyTypeError):

    if (import_7035 != 'pyd_module'):
        __import__(import_7035)
        sys_modules_7036 = sys.modules[import_7035]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', sys_modules_7036.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_7036, sys_modules_7036.module_type_store, module_type_store)
    else:
        from stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'stypy_copy.errors_copy.type_error_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'stypy_copy.errors_copy.type_error_copy', import_7035)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from call_handler_copy import CallHandler' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7037 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'call_handler_copy')

if (type(import_7037) is not StypyTypeError):

    if (import_7037 != 'pyd_module'):
        __import__(import_7037)
        sys_modules_7038 = sys.modules[import_7037]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'call_handler_copy', sys_modules_7038.module_type_store, module_type_store, ['CallHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_7038, sys_modules_7038.module_type_store, module_type_store)
    else:
        from call_handler_copy import CallHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'call_handler_copy', None, module_type_store, ['CallHandler'], [CallHandler])

else:
    # Assigning a type to the variable 'call_handler_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'call_handler_copy', import_7037)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import no_recursion_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7039 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_7039) is not StypyTypeError):

    if (import_7039 != 'pyd_module'):
        __import__(import_7039)
        sys_modules_7040 = sys.modules[import_7039]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_7040.module_type_store, module_type_store, ['no_recursion_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_7040, sys_modules_7040.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import no_recursion_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['no_recursion_copy'], [no_recursion_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_7039)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy import is_user_defined_class, is_user_defined_module' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7041 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy')

if (type(import_7041) is not StypyTypeError):

    if (import_7041 != 'pyd_module'):
        __import__(import_7041)
        sys_modules_7042 = sys.modules[import_7041]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy', sys_modules_7042.module_type_store, module_type_store, ['is_user_defined_class', 'is_user_defined_module'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_7042, sys_modules_7042.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy import is_user_defined_class, is_user_defined_module

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy', None, module_type_store, ['is_user_defined_class', 'is_user_defined_module'], [is_user_defined_class, is_user_defined_module])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy', import_7041)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from stypy_copy.python_lib_copy.python_types_copy import type_inference_copy' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7043 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy')

if (type(import_7043) is not StypyTypeError):

    if (import_7043 != 'pyd_module'):
        __import__(import_7043)
        sys_modules_7044 = sys.modules[import_7043]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy', sys_modules_7044.module_type_store, module_type_store, ['type_inference_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_7044, sys_modules_7044.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.python_types_copy import type_inference_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy', None, module_type_store, ['type_inference_copy'], [type_inference_copy])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.python_types_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'stypy_copy.python_lib_copy.python_types_copy', import_7043)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

# Declaration of the 'UserCallablesCallHandler' class
# Getting the type of 'CallHandler' (line 11)
CallHandler_7045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 31), 'CallHandler')

class UserCallablesCallHandler(CallHandler_7045, ):
    str_7046 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n    This class handles calls to code that have been obtained from available .py source files that the program uses,\n    generated to perform type inference of the original callable code. Remember that all functions in type inference\n    programs are transformed to the following form:\n\n    def function_name(*args, **kwargs):\n        ...\n\n    This handler just use this convention to call the code and return the result. Transformed code can handle\n     UnionTypes and other stypy constructs.\n    ')

    @norecursion
    def applies_to(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'applies_to'
        module_type_store = module_type_store.open_function_context('applies_to', 24, 4, False)
        # Assigning a type to the variable 'self' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_localization', localization)
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_type_store', module_type_store)
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_function_name', 'UserCallablesCallHandler.applies_to')
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'callable_entity'])
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_varargs_param_name', None)
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_kwargs_param_name', None)
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_call_defaults', defaults)
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_call_varargs', varargs)
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UserCallablesCallHandler.applies_to.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UserCallablesCallHandler.applies_to', ['proxy_obj', 'callable_entity'], None, None, defaults, varargs, kwargs)

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

        str_7047 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\n        This method determines if this call handler is able to respond to a call to callable_entity.\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param callable_entity: Callable entity\n        :return: bool\n        ')
        
        # Evaluating a boolean operation
        
        # Call to is_user_defined_class(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'callable_entity' (line 31)
        callable_entity_7049 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'callable_entity', False)
        # Processing the call keyword arguments (line 31)
        kwargs_7050 = {}
        # Getting the type of 'is_user_defined_class' (line 31)
        is_user_defined_class_7048 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'is_user_defined_class', False)
        # Calling is_user_defined_class(args, kwargs) (line 31)
        is_user_defined_class_call_result_7051 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), is_user_defined_class_7048, *[callable_entity_7049], **kwargs_7050)
        
        
        # Getting the type of 'proxy_obj' (line 31)
        proxy_obj_7052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 57), 'proxy_obj')
        # Obtaining the member 'parent_proxy' of a type (line 31)
        parent_proxy_7053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 57), proxy_obj_7052, 'parent_proxy')
        # Obtaining the member 'python_entity' of a type (line 31)
        python_entity_7054 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 57), parent_proxy_7053, 'python_entity')
        # Getting the type of 'no_recursion_copy' (line 31)
        no_recursion_copy_7055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 97), 'no_recursion_copy')
        # Applying the binary operator '==' (line 31)
        result_eq_7056 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 57), '==', python_entity_7054, no_recursion_copy_7055)
        
        # Applying the binary operator 'or' (line 31)
        result_or_keyword_7057 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'or', is_user_defined_class_call_result_7051, result_eq_7056)
        
        # Evaluating a boolean operation
        
        # Call to ismethod(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'callable_entity' (line 32)
        callable_entity_7060 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'callable_entity', False)
        # Processing the call keyword arguments (line 32)
        kwargs_7061 = {}
        # Getting the type of 'inspect' (line 32)
        inspect_7058 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'inspect', False)
        # Obtaining the member 'ismethod' of a type (line 32)
        ismethod_7059 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), inspect_7058, 'ismethod')
        # Calling ismethod(args, kwargs) (line 32)
        ismethod_call_result_7062 = invoke(stypy.reporting.localization.Localization(__file__, 32, 16), ismethod_7059, *[callable_entity_7060], **kwargs_7061)
        
        
        # Call to is_user_defined_class(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'proxy_obj' (line 32)
        proxy_obj_7064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 76), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 32)
        parent_proxy_7065 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 76), proxy_obj_7064, 'parent_proxy')
        # Obtaining the member 'python_entity' of a type (line 32)
        python_entity_7066 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 76), parent_proxy_7065, 'python_entity')
        # Processing the call keyword arguments (line 32)
        kwargs_7067 = {}
        # Getting the type of 'is_user_defined_class' (line 32)
        is_user_defined_class_7063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 54), 'is_user_defined_class', False)
        # Calling is_user_defined_class(args, kwargs) (line 32)
        is_user_defined_class_call_result_7068 = invoke(stypy.reporting.localization.Localization(__file__, 32, 54), is_user_defined_class_7063, *[python_entity_7066], **kwargs_7067)
        
        # Applying the binary operator 'and' (line 32)
        result_and_keyword_7069 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 16), 'and', ismethod_call_result_7062, is_user_defined_class_call_result_7068)
        
        # Applying the binary operator 'or' (line 31)
        result_or_keyword_7070 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'or', result_or_keyword_7057, result_and_keyword_7069)
        
        # Evaluating a boolean operation
        
        # Call to isfunction(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'callable_entity' (line 33)
        callable_entity_7073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 35), 'callable_entity', False)
        # Processing the call keyword arguments (line 33)
        kwargs_7074 = {}
        # Getting the type of 'inspect' (line 33)
        inspect_7071 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 33)
        isfunction_7072 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), inspect_7071, 'isfunction')
        # Calling isfunction(args, kwargs) (line 33)
        isfunction_call_result_7075 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), isfunction_7072, *[callable_entity_7073], **kwargs_7074)
        
        
        # Call to is_user_defined_class(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'proxy_obj' (line 33)
        proxy_obj_7077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 78), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 33)
        parent_proxy_7078 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 78), proxy_obj_7077, 'parent_proxy')
        # Obtaining the member 'python_entity' of a type (line 33)
        python_entity_7079 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 78), parent_proxy_7078, 'python_entity')
        # Processing the call keyword arguments (line 33)
        kwargs_7080 = {}
        # Getting the type of 'is_user_defined_class' (line 33)
        is_user_defined_class_7076 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 56), 'is_user_defined_class', False)
        # Calling is_user_defined_class(args, kwargs) (line 33)
        is_user_defined_class_call_result_7081 = invoke(stypy.reporting.localization.Localization(__file__, 33, 56), is_user_defined_class_7076, *[python_entity_7079], **kwargs_7080)
        
        # Applying the binary operator 'and' (line 33)
        result_and_keyword_7082 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 16), 'and', isfunction_call_result_7075, is_user_defined_class_call_result_7081)
        
        # Applying the binary operator 'or' (line 31)
        result_or_keyword_7083 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'or', result_or_keyword_7070, result_and_keyword_7082)
        
        # Evaluating a boolean operation
        
        # Call to isfunction(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'callable_entity' (line 34)
        callable_entity_7086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'callable_entity', False)
        # Processing the call keyword arguments (line 34)
        kwargs_7087 = {}
        # Getting the type of 'inspect' (line 34)
        inspect_7084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 34)
        isfunction_7085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), inspect_7084, 'isfunction')
        # Calling isfunction(args, kwargs) (line 34)
        isfunction_call_result_7088 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), isfunction_7085, *[callable_entity_7086], **kwargs_7087)
        
        
        # Call to is_user_defined_module(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'callable_entity' (line 34)
        callable_entity_7090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 79), 'callable_entity', False)
        # Obtaining the member '__module__' of a type (line 34)
        module___7091 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 79), callable_entity_7090, '__module__')
        # Processing the call keyword arguments (line 34)
        kwargs_7092 = {}
        # Getting the type of 'is_user_defined_module' (line 34)
        is_user_defined_module_7089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 56), 'is_user_defined_module', False)
        # Calling is_user_defined_module(args, kwargs) (line 34)
        is_user_defined_module_call_result_7093 = invoke(stypy.reporting.localization.Localization(__file__, 34, 56), is_user_defined_module_7089, *[module___7091], **kwargs_7092)
        
        # Applying the binary operator 'and' (line 34)
        result_and_keyword_7094 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 16), 'and', isfunction_call_result_7088, is_user_defined_module_call_result_7093)
        
        # Applying the binary operator 'or' (line 31)
        result_or_keyword_7095 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'or', result_or_keyword_7083, result_and_keyword_7094)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', result_or_keyword_7095)
        
        # ################# End of 'applies_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'applies_to' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_7096 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7096)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'applies_to'
        return stypy_return_type_7096


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 36, 4, False)
        # Assigning a type to the variable 'self' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_localization', localization)
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_function_name', 'UserCallablesCallHandler.__call__')
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'localization', 'callable_entity'])
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'arg_types')
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs_types')
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        UserCallablesCallHandler.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'UserCallablesCallHandler.__call__', ['proxy_obj', 'localization', 'callable_entity'], 'arg_types', 'kwargs_types', defaults, varargs, kwargs)

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

        str_7097 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', '\n        Perform the call and return the call result\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param localization: Caller information\n        :param callable_entity: Callable entity\n        :param arg_types: Arguments\n        :param kwargs_types: Keyword arguments\n        :return: Return type of the call\n        ')
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'callable_entity' (line 46)
        callable_entity_7098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'callable_entity')
        # Assigning a type to the variable 'callable_python_entity' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'callable_python_entity', callable_entity_7098)
        
        
        # SSA begins for try-except statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 48):
        
        # Call to get_error_msgs(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_7101 = {}
        # Getting the type of 'TypeError' (line 48)
        TypeError_7099 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'TypeError', False)
        # Obtaining the member 'get_error_msgs' of a type (line 48)
        get_error_msgs_7100 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 25), TypeError_7099, 'get_error_msgs')
        # Calling get_error_msgs(args, kwargs) (line 48)
        get_error_msgs_call_result_7102 = invoke(stypy.reporting.localization.Localization(__file__, 48, 25), get_error_msgs_7100, *[], **kwargs_7101)
        
        # Assigning a type to the variable 'pre_errors' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'pre_errors', get_error_msgs_call_result_7102)
        
        # Type idiom detected: calculating its left and rigth part (line 50)
        # Getting the type of 'localization' (line 50)
        localization_7103 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'localization')
        # Getting the type of 'None' (line 50)
        None_7104 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'None')
        
        (may_be_7105, more_types_in_union_7106) = may_be_none(localization_7103, None_7104)

        if may_be_7105:

            if more_types_in_union_7106:
                # Runtime conditional SSA (line 50)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 51):
            
            # Call to callable_python_entity(...): (line 51)
            # Getting the type of 'arg_types' (line 51)
            arg_types_7108 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 54), 'arg_types', False)
            # Processing the call keyword arguments (line 51)
            # Getting the type of 'kwargs_types' (line 51)
            kwargs_types_7109 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 67), 'kwargs_types', False)
            kwargs_7110 = {'kwargs_types_7109': kwargs_types_7109}
            # Getting the type of 'callable_python_entity' (line 51)
            callable_python_entity_7107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'callable_python_entity', False)
            # Calling callable_python_entity(args, kwargs) (line 51)
            callable_python_entity_call_result_7111 = invoke(stypy.reporting.localization.Localization(__file__, 51, 30), callable_python_entity_7107, *[arg_types_7108], **kwargs_7110)
            
            # Assigning a type to the variable 'call_result' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'call_result', callable_python_entity_call_result_7111)

            if more_types_in_union_7106:
                # Runtime conditional SSA for else branch (line 50)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_7105) or more_types_in_union_7106):
            
            # Evaluating a boolean operation
            
            # Call to isfunction(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'proxy_obj' (line 54)
            proxy_obj_7114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 54)
            python_entity_7115 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 38), proxy_obj_7114, 'python_entity')
            # Processing the call keyword arguments (line 54)
            kwargs_7116 = {}
            # Getting the type of 'inspect' (line 54)
            inspect_7112 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 54)
            isfunction_7113 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 19), inspect_7112, 'isfunction')
            # Calling isfunction(args, kwargs) (line 54)
            isfunction_call_result_7117 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), isfunction_7113, *[python_entity_7115], **kwargs_7116)
            
            
            # Call to isclass(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'proxy_obj' (line 55)
            proxy_obj_7120 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 55)
            parent_proxy_7121 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), proxy_obj_7120, 'parent_proxy')
            # Obtaining the member 'python_entity' of a type (line 55)
            python_entity_7122 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), parent_proxy_7121, 'python_entity')
            # Processing the call keyword arguments (line 54)
            kwargs_7123 = {}
            # Getting the type of 'inspect' (line 54)
            inspect_7118 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 67), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 54)
            isclass_7119 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 67), inspect_7118, 'isclass')
            # Calling isclass(args, kwargs) (line 54)
            isclass_call_result_7124 = invoke(stypy.reporting.localization.Localization(__file__, 54, 67), isclass_7119, *[python_entity_7122], **kwargs_7123)
            
            # Applying the binary operator 'and' (line 54)
            result_and_keyword_7125 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 19), 'and', isfunction_call_result_7117, isclass_call_result_7124)
            
            # Testing if the type of an if condition is none (line 54)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 16), result_and_keyword_7125):
                
                # Evaluating a boolean operation
                
                # Call to ismethod(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'proxy_obj' (line 60)
                proxy_obj_7138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 60)
                python_entity_7139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 40), proxy_obj_7138, 'python_entity')
                # Processing the call keyword arguments (line 60)
                kwargs_7140 = {}
                # Getting the type of 'inspect' (line 60)
                inspect_7136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'inspect', False)
                # Obtaining the member 'ismethod' of a type (line 60)
                ismethod_7137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), inspect_7136, 'ismethod')
                # Calling ismethod(args, kwargs) (line 60)
                ismethod_call_result_7141 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), ismethod_7137, *[python_entity_7139], **kwargs_7140)
                
                
                str_7142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 69), 'str', '.__init__')
                # Getting the type of 'proxy_obj' (line 60)
                proxy_obj_7143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 84), 'proxy_obj')
                # Obtaining the member 'name' of a type (line 60)
                name_7144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 84), proxy_obj_7143, 'name')
                # Applying the binary operator 'in' (line 60)
                result_contains_7145 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 69), 'in', str_7142, name_7144)
                
                # Applying the binary operator 'and' (line 60)
                result_and_keyword_7146 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), 'and', ismethod_call_result_7141, result_contains_7145)
                
                # Call to isclass(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'proxy_obj' (line 61)
                proxy_obj_7149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 61)
                parent_proxy_7150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), proxy_obj_7149, 'parent_proxy')
                # Obtaining the member 'python_entity' of a type (line 61)
                python_entity_7151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), parent_proxy_7150, 'python_entity')
                # Processing the call keyword arguments (line 60)
                kwargs_7152 = {}
                # Getting the type of 'inspect' (line 60)
                inspect_7147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 103), 'inspect', False)
                # Obtaining the member 'isclass' of a type (line 60)
                isclass_7148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 103), inspect_7147, 'isclass')
                # Calling isclass(args, kwargs) (line 60)
                isclass_call_result_7153 = invoke(stypy.reporting.localization.Localization(__file__, 60, 103), isclass_7148, *[python_entity_7151], **kwargs_7152)
                
                # Applying the binary operator 'and' (line 60)
                result_and_keyword_7154 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), 'and', result_and_keyword_7146, isclass_call_result_7153)
                
                # Testing if the type of an if condition is none (line 60)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 20), result_and_keyword_7154):
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to callable_python_entity(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'localization' (line 66)
                    localization_7168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'localization', False)
                    # Getting the type of 'arg_types' (line 66)
                    arg_types_7169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 76), 'arg_types', False)
                    # Processing the call keyword arguments (line 66)
                    # Getting the type of 'kwargs_types' (line 66)
                    kwargs_types_7170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 89), 'kwargs_types', False)
                    kwargs_7171 = {'kwargs_types_7170': kwargs_types_7170}
                    # Getting the type of 'callable_python_entity' (line 66)
                    callable_python_entity_7167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 66)
                    callable_python_entity_call_result_7172 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), callable_python_entity_7167, *[localization_7168, arg_types_7169], **kwargs_7171)
                    
                    # Assigning a type to the variable 'call_result' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'call_result', callable_python_entity_call_result_7172)
                else:
                    
                    # Testing the type of an if condition (line 60)
                    if_condition_7155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 20), result_and_keyword_7154)
                    # Assigning a type to the variable 'if_condition_7155' (line 60)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'if_condition_7155', if_condition_7155)
                    # SSA begins for if statement (line 60)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 62):
                    
                    # Call to callable_python_entity(...): (line 62)
                    # Processing the call arguments (line 62)
                    
                    # Obtaining the type of the subscript
                    int_7157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 71), 'int')
                    # Getting the type of 'arg_types' (line 62)
                    arg_types_7158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 61), 'arg_types', False)
                    # Obtaining the member '__getitem__' of a type (line 62)
                    getitem___7159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 61), arg_types_7158, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
                    subscript_call_result_7160 = invoke(stypy.reporting.localization.Localization(__file__, 62, 61), getitem___7159, int_7157)
                    
                    # Obtaining the member 'python_entity' of a type (line 62)
                    python_entity_7161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 61), subscript_call_result_7160, 'python_entity')
                    # Getting the type of 'localization' (line 63)
                    localization_7162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'localization', False)
                    # Getting the type of 'arg_types' (line 63)
                    arg_types_7163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 52), 'arg_types', False)
                    # Processing the call keyword arguments (line 62)
                    # Getting the type of 'kwargs_types' (line 63)
                    kwargs_types_7164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 65), 'kwargs_types', False)
                    kwargs_7165 = {'kwargs_types_7164': kwargs_types_7164}
                    # Getting the type of 'callable_python_entity' (line 62)
                    callable_python_entity_7156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 62)
                    callable_python_entity_call_result_7166 = invoke(stypy.reporting.localization.Localization(__file__, 62, 38), callable_python_entity_7156, *[python_entity_7161, localization_7162, arg_types_7163], **kwargs_7165)
                    
                    # Assigning a type to the variable 'call_result' (line 62)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'call_result', callable_python_entity_call_result_7166)
                    # SSA branch for the else part of an if statement (line 60)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to callable_python_entity(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'localization' (line 66)
                    localization_7168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'localization', False)
                    # Getting the type of 'arg_types' (line 66)
                    arg_types_7169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 76), 'arg_types', False)
                    # Processing the call keyword arguments (line 66)
                    # Getting the type of 'kwargs_types' (line 66)
                    kwargs_types_7170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 89), 'kwargs_types', False)
                    kwargs_7171 = {'kwargs_types_7170': kwargs_types_7170}
                    # Getting the type of 'callable_python_entity' (line 66)
                    callable_python_entity_7167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 66)
                    callable_python_entity_call_result_7172 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), callable_python_entity_7167, *[localization_7168, arg_types_7169], **kwargs_7171)
                    
                    # Assigning a type to the variable 'call_result' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'call_result', callable_python_entity_call_result_7172)
                    # SSA join for if statement (line 60)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 54)
                if_condition_7126 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 16), result_and_keyword_7125)
                # Assigning a type to the variable 'if_condition_7126' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'if_condition_7126', if_condition_7126)
                # SSA begins for if statement (line 54)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 56):
                
                # Call to callable_python_entity(...): (line 56)
                # Processing the call arguments (line 56)
                # Getting the type of 'proxy_obj' (line 56)
                proxy_obj_7128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 57), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 56)
                parent_proxy_7129 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 57), proxy_obj_7128, 'parent_proxy')
                # Obtaining the member 'python_entity' of a type (line 56)
                python_entity_7130 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 57), parent_proxy_7129, 'python_entity')
                # Getting the type of 'localization' (line 57)
                localization_7131 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 57), 'localization', False)
                # Getting the type of 'arg_types' (line 57)
                arg_types_7132 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 72), 'arg_types', False)
                # Processing the call keyword arguments (line 56)
                # Getting the type of 'kwargs_types' (line 57)
                kwargs_types_7133 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 85), 'kwargs_types', False)
                kwargs_7134 = {'kwargs_types_7133': kwargs_types_7133}
                # Getting the type of 'callable_python_entity' (line 56)
                callable_python_entity_7127 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'callable_python_entity', False)
                # Calling callable_python_entity(args, kwargs) (line 56)
                callable_python_entity_call_result_7135 = invoke(stypy.reporting.localization.Localization(__file__, 56, 34), callable_python_entity_7127, *[python_entity_7130, localization_7131, arg_types_7132], **kwargs_7134)
                
                # Assigning a type to the variable 'call_result' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'call_result', callable_python_entity_call_result_7135)
                # SSA branch for the else part of an if statement (line 54)
                module_type_store.open_ssa_branch('else')
                
                # Evaluating a boolean operation
                
                # Call to ismethod(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'proxy_obj' (line 60)
                proxy_obj_7138 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 60)
                python_entity_7139 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 40), proxy_obj_7138, 'python_entity')
                # Processing the call keyword arguments (line 60)
                kwargs_7140 = {}
                # Getting the type of 'inspect' (line 60)
                inspect_7136 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'inspect', False)
                # Obtaining the member 'ismethod' of a type (line 60)
                ismethod_7137 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), inspect_7136, 'ismethod')
                # Calling ismethod(args, kwargs) (line 60)
                ismethod_call_result_7141 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), ismethod_7137, *[python_entity_7139], **kwargs_7140)
                
                
                str_7142 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 69), 'str', '.__init__')
                # Getting the type of 'proxy_obj' (line 60)
                proxy_obj_7143 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 84), 'proxy_obj')
                # Obtaining the member 'name' of a type (line 60)
                name_7144 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 84), proxy_obj_7143, 'name')
                # Applying the binary operator 'in' (line 60)
                result_contains_7145 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 69), 'in', str_7142, name_7144)
                
                # Applying the binary operator 'and' (line 60)
                result_and_keyword_7146 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), 'and', ismethod_call_result_7141, result_contains_7145)
                
                # Call to isclass(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'proxy_obj' (line 61)
                proxy_obj_7149 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 61)
                parent_proxy_7150 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), proxy_obj_7149, 'parent_proxy')
                # Obtaining the member 'python_entity' of a type (line 61)
                python_entity_7151 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), parent_proxy_7150, 'python_entity')
                # Processing the call keyword arguments (line 60)
                kwargs_7152 = {}
                # Getting the type of 'inspect' (line 60)
                inspect_7147 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 103), 'inspect', False)
                # Obtaining the member 'isclass' of a type (line 60)
                isclass_7148 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 103), inspect_7147, 'isclass')
                # Calling isclass(args, kwargs) (line 60)
                isclass_call_result_7153 = invoke(stypy.reporting.localization.Localization(__file__, 60, 103), isclass_7148, *[python_entity_7151], **kwargs_7152)
                
                # Applying the binary operator 'and' (line 60)
                result_and_keyword_7154 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), 'and', result_and_keyword_7146, isclass_call_result_7153)
                
                # Testing if the type of an if condition is none (line 60)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 20), result_and_keyword_7154):
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to callable_python_entity(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'localization' (line 66)
                    localization_7168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'localization', False)
                    # Getting the type of 'arg_types' (line 66)
                    arg_types_7169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 76), 'arg_types', False)
                    # Processing the call keyword arguments (line 66)
                    # Getting the type of 'kwargs_types' (line 66)
                    kwargs_types_7170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 89), 'kwargs_types', False)
                    kwargs_7171 = {'kwargs_types_7170': kwargs_types_7170}
                    # Getting the type of 'callable_python_entity' (line 66)
                    callable_python_entity_7167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 66)
                    callable_python_entity_call_result_7172 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), callable_python_entity_7167, *[localization_7168, arg_types_7169], **kwargs_7171)
                    
                    # Assigning a type to the variable 'call_result' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'call_result', callable_python_entity_call_result_7172)
                else:
                    
                    # Testing the type of an if condition (line 60)
                    if_condition_7155 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 20), result_and_keyword_7154)
                    # Assigning a type to the variable 'if_condition_7155' (line 60)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'if_condition_7155', if_condition_7155)
                    # SSA begins for if statement (line 60)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 62):
                    
                    # Call to callable_python_entity(...): (line 62)
                    # Processing the call arguments (line 62)
                    
                    # Obtaining the type of the subscript
                    int_7157 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 71), 'int')
                    # Getting the type of 'arg_types' (line 62)
                    arg_types_7158 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 61), 'arg_types', False)
                    # Obtaining the member '__getitem__' of a type (line 62)
                    getitem___7159 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 61), arg_types_7158, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
                    subscript_call_result_7160 = invoke(stypy.reporting.localization.Localization(__file__, 62, 61), getitem___7159, int_7157)
                    
                    # Obtaining the member 'python_entity' of a type (line 62)
                    python_entity_7161 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 61), subscript_call_result_7160, 'python_entity')
                    # Getting the type of 'localization' (line 63)
                    localization_7162 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'localization', False)
                    # Getting the type of 'arg_types' (line 63)
                    arg_types_7163 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 52), 'arg_types', False)
                    # Processing the call keyword arguments (line 62)
                    # Getting the type of 'kwargs_types' (line 63)
                    kwargs_types_7164 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 65), 'kwargs_types', False)
                    kwargs_7165 = {'kwargs_types_7164': kwargs_types_7164}
                    # Getting the type of 'callable_python_entity' (line 62)
                    callable_python_entity_7156 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 62)
                    callable_python_entity_call_result_7166 = invoke(stypy.reporting.localization.Localization(__file__, 62, 38), callable_python_entity_7156, *[python_entity_7161, localization_7162, arg_types_7163], **kwargs_7165)
                    
                    # Assigning a type to the variable 'call_result' (line 62)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'call_result', callable_python_entity_call_result_7166)
                    # SSA branch for the else part of an if statement (line 60)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to callable_python_entity(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'localization' (line 66)
                    localization_7168 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'localization', False)
                    # Getting the type of 'arg_types' (line 66)
                    arg_types_7169 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 76), 'arg_types', False)
                    # Processing the call keyword arguments (line 66)
                    # Getting the type of 'kwargs_types' (line 66)
                    kwargs_types_7170 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 89), 'kwargs_types', False)
                    kwargs_7171 = {'kwargs_types_7170': kwargs_types_7170}
                    # Getting the type of 'callable_python_entity' (line 66)
                    callable_python_entity_7167 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 66)
                    callable_python_entity_call_result_7172 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), callable_python_entity_7167, *[localization_7168, arg_types_7169], **kwargs_7171)
                    
                    # Assigning a type to the variable 'call_result' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'call_result', callable_python_entity_call_result_7172)
                    # SSA join for if statement (line 60)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 54)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_7105 and more_types_in_union_7106):
                # SSA join for if statement (line 50)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 69)
        # Getting the type of 'call_result' (line 69)
        call_result_7173 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'call_result')
        # Getting the type of 'None' (line 69)
        None_7174 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'None')
        
        (may_be_7175, more_types_in_union_7176) = may_not_be_none(call_result_7173, None_7174)

        if may_be_7175:

            if more_types_in_union_7176:
                # Runtime conditional SSA (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to isclass(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'callable_entity' (line 70)
            callable_entity_7179 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 39), 'callable_entity', False)
            # Processing the call keyword arguments (line 70)
            kwargs_7180 = {}
            # Getting the type of 'inspect' (line 70)
            inspect_7177 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 70)
            isclass_7178 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 23), inspect_7177, 'isclass')
            # Calling isclass(args, kwargs) (line 70)
            isclass_call_result_7181 = invoke(stypy.reporting.localization.Localization(__file__, 70, 23), isclass_7178, *[callable_entity_7179], **kwargs_7180)
            
            # Applying the 'not' unary operator (line 70)
            result_not__7182 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), 'not', isclass_call_result_7181)
            
            # Testing if the type of an if condition is none (line 70)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 16), result_not__7182):
                
                # Call to isinstance(...): (line 74)
                # Processing the call arguments (line 74)
                # Getting the type of 'call_result' (line 74)
                call_result_7186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'call_result', False)
                # Getting the type of 'callable_entity' (line 74)
                callable_entity_7187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 47), 'callable_entity', False)
                # Processing the call keyword arguments (line 74)
                kwargs_7188 = {}
                # Getting the type of 'isinstance' (line 74)
                isinstance_7185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 74)
                isinstance_call_result_7189 = invoke(stypy.reporting.localization.Localization(__file__, 74, 23), isinstance_7185, *[call_result_7186, callable_entity_7187], **kwargs_7188)
                
                # Testing if the type of an if condition is none (line 74)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 20), isinstance_call_result_7189):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 74)
                    if_condition_7190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 20), isinstance_call_result_7189)
                    # Assigning a type to the variable 'if_condition_7190' (line 74)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'if_condition_7190', if_condition_7190)
                    # SSA begins for if statement (line 74)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 75):
                    
                    # Call to get_error_msgs(...): (line 75)
                    # Processing the call keyword arguments (line 75)
                    kwargs_7193 = {}
                    # Getting the type of 'TypeError' (line 75)
                    TypeError_7191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'TypeError', False)
                    # Obtaining the member 'get_error_msgs' of a type (line 75)
                    get_error_msgs_7192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 38), TypeError_7191, 'get_error_msgs')
                    # Calling get_error_msgs(args, kwargs) (line 75)
                    get_error_msgs_call_result_7194 = invoke(stypy.reporting.localization.Localization(__file__, 75, 38), get_error_msgs_7192, *[], **kwargs_7193)
                    
                    # Assigning a type to the variable 'post_errors' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'post_errors', get_error_msgs_call_result_7194)
                    
                    
                    
                    # Call to len(...): (line 76)
                    # Processing the call arguments (line 76)
                    # Getting the type of 'pre_errors' (line 76)
                    pre_errors_7196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 35), 'pre_errors', False)
                    # Processing the call keyword arguments (line 76)
                    kwargs_7197 = {}
                    # Getting the type of 'len' (line 76)
                    len_7195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'len', False)
                    # Calling len(args, kwargs) (line 76)
                    len_call_result_7198 = invoke(stypy.reporting.localization.Localization(__file__, 76, 31), len_7195, *[pre_errors_7196], **kwargs_7197)
                    
                    
                    # Call to len(...): (line 76)
                    # Processing the call arguments (line 76)
                    # Getting the type of 'post_errors' (line 76)
                    post_errors_7200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 54), 'post_errors', False)
                    # Processing the call keyword arguments (line 76)
                    kwargs_7201 = {}
                    # Getting the type of 'len' (line 76)
                    len_7199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 50), 'len', False)
                    # Calling len(args, kwargs) (line 76)
                    len_call_result_7202 = invoke(stypy.reporting.localization.Localization(__file__, 76, 50), len_7199, *[post_errors_7200], **kwargs_7201)
                    
                    # Applying the binary operator '==' (line 76)
                    result_eq_7203 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 31), '==', len_call_result_7198, len_call_result_7202)
                    
                    # Applying the 'not' unary operator (line 76)
                    result_not__7204 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 27), 'not', result_eq_7203)
                    
                    # Testing if the type of an if condition is none (line 76)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 24), result_not__7204):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 76)
                        if_condition_7205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 24), result_not__7204)
                        # Assigning a type to the variable 'if_condition_7205' (line 76)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'if_condition_7205', if_condition_7205)
                        # SSA begins for if statement (line 76)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to TypeError(...): (line 77)
                        # Processing the call arguments (line 77)
                        # Getting the type of 'localization' (line 77)
                        localization_7207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'localization', False)
                        
                        # Call to format(...): (line 77)
                        # Processing the call arguments (line 77)
                        # Getting the type of 'callable_entity' (line 78)
                        callable_entity_7210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 81), 'callable_entity', False)
                        # Processing the call keyword arguments (line 77)
                        kwargs_7211 = {}
                        str_7208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 59), 'str', 'Could not create an instance of ({0}) due to constructor call errors')
                        # Obtaining the member 'format' of a type (line 77)
                        format_7209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 59), str_7208, 'format')
                        # Calling format(args, kwargs) (line 77)
                        format_call_result_7212 = invoke(stypy.reporting.localization.Localization(__file__, 77, 59), format_7209, *[callable_entity_7210], **kwargs_7211)
                        
                        # Getting the type of 'False' (line 78)
                        False_7213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 99), 'False', False)
                        # Processing the call keyword arguments (line 77)
                        kwargs_7214 = {}
                        # Getting the type of 'TypeError' (line 77)
                        TypeError_7206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 77)
                        TypeError_call_result_7215 = invoke(stypy.reporting.localization.Localization(__file__, 77, 35), TypeError_7206, *[localization_7207, format_call_result_7212, False_7213], **kwargs_7214)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 77)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'stypy_return_type', TypeError_call_result_7215)
                        # SSA join for if statement (line 76)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 74)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Name (line 80):
                # Getting the type of 'call_result' (line 80)
                call_result_7216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'call_result')
                # Assigning a type to the variable 'instance' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'instance', call_result_7216)
                
                # Assigning a Name to a Name (line 81):
                # Getting the type of 'callable_entity' (line 81)
                callable_entity_7217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'callable_entity')
                # Assigning a type to the variable 'type_' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'type_', callable_entity_7217)
                
                # Call to instance(...): (line 82)
                # Processing the call arguments (line 82)
                # Getting the type of 'type_' (line 82)
                type__7222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 96), 'type_', False)
                # Processing the call keyword arguments (line 82)
                # Getting the type of 'instance' (line 82)
                instance_7223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 112), 'instance', False)
                keyword_7224 = instance_7223
                kwargs_7225 = {'instance': keyword_7224}
                # Getting the type of 'type_inference_copy' (line 82)
                type_inference_copy_7218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'type_inference_copy', False)
                # Obtaining the member 'type_inference_proxy' of a type (line 82)
                type_inference_proxy_7219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), type_inference_copy_7218, 'type_inference_proxy')
                # Obtaining the member 'TypeInferenceProxy' of a type (line 82)
                TypeInferenceProxy_7220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), type_inference_proxy_7219, 'TypeInferenceProxy')
                # Obtaining the member 'instance' of a type (line 82)
                instance_7221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), TypeInferenceProxy_7220, 'instance')
                # Calling instance(args, kwargs) (line 82)
                instance_call_result_7226 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), instance_7221, *[type__7222], **kwargs_7225)
                
                # Assigning a type to the variable 'stypy_return_type' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'stypy_return_type', instance_call_result_7226)
            else:
                
                # Testing the type of an if condition (line 70)
                if_condition_7183 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 16), result_not__7182)
                # Assigning a type to the variable 'if_condition_7183' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'if_condition_7183', if_condition_7183)
                # SSA begins for if statement (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'call_result' (line 71)
                call_result_7184 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'call_result')
                # Assigning a type to the variable 'stypy_return_type' (line 71)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'stypy_return_type', call_result_7184)
                # SSA branch for the else part of an if statement (line 70)
                module_type_store.open_ssa_branch('else')
                
                # Call to isinstance(...): (line 74)
                # Processing the call arguments (line 74)
                # Getting the type of 'call_result' (line 74)
                call_result_7186 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'call_result', False)
                # Getting the type of 'callable_entity' (line 74)
                callable_entity_7187 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 47), 'callable_entity', False)
                # Processing the call keyword arguments (line 74)
                kwargs_7188 = {}
                # Getting the type of 'isinstance' (line 74)
                isinstance_7185 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 74)
                isinstance_call_result_7189 = invoke(stypy.reporting.localization.Localization(__file__, 74, 23), isinstance_7185, *[call_result_7186, callable_entity_7187], **kwargs_7188)
                
                # Testing if the type of an if condition is none (line 74)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 20), isinstance_call_result_7189):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 74)
                    if_condition_7190 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 20), isinstance_call_result_7189)
                    # Assigning a type to the variable 'if_condition_7190' (line 74)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'if_condition_7190', if_condition_7190)
                    # SSA begins for if statement (line 74)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 75):
                    
                    # Call to get_error_msgs(...): (line 75)
                    # Processing the call keyword arguments (line 75)
                    kwargs_7193 = {}
                    # Getting the type of 'TypeError' (line 75)
                    TypeError_7191 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'TypeError', False)
                    # Obtaining the member 'get_error_msgs' of a type (line 75)
                    get_error_msgs_7192 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 38), TypeError_7191, 'get_error_msgs')
                    # Calling get_error_msgs(args, kwargs) (line 75)
                    get_error_msgs_call_result_7194 = invoke(stypy.reporting.localization.Localization(__file__, 75, 38), get_error_msgs_7192, *[], **kwargs_7193)
                    
                    # Assigning a type to the variable 'post_errors' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'post_errors', get_error_msgs_call_result_7194)
                    
                    
                    
                    # Call to len(...): (line 76)
                    # Processing the call arguments (line 76)
                    # Getting the type of 'pre_errors' (line 76)
                    pre_errors_7196 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 35), 'pre_errors', False)
                    # Processing the call keyword arguments (line 76)
                    kwargs_7197 = {}
                    # Getting the type of 'len' (line 76)
                    len_7195 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'len', False)
                    # Calling len(args, kwargs) (line 76)
                    len_call_result_7198 = invoke(stypy.reporting.localization.Localization(__file__, 76, 31), len_7195, *[pre_errors_7196], **kwargs_7197)
                    
                    
                    # Call to len(...): (line 76)
                    # Processing the call arguments (line 76)
                    # Getting the type of 'post_errors' (line 76)
                    post_errors_7200 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 54), 'post_errors', False)
                    # Processing the call keyword arguments (line 76)
                    kwargs_7201 = {}
                    # Getting the type of 'len' (line 76)
                    len_7199 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 50), 'len', False)
                    # Calling len(args, kwargs) (line 76)
                    len_call_result_7202 = invoke(stypy.reporting.localization.Localization(__file__, 76, 50), len_7199, *[post_errors_7200], **kwargs_7201)
                    
                    # Applying the binary operator '==' (line 76)
                    result_eq_7203 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 31), '==', len_call_result_7198, len_call_result_7202)
                    
                    # Applying the 'not' unary operator (line 76)
                    result_not__7204 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 27), 'not', result_eq_7203)
                    
                    # Testing if the type of an if condition is none (line 76)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 24), result_not__7204):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 76)
                        if_condition_7205 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 24), result_not__7204)
                        # Assigning a type to the variable 'if_condition_7205' (line 76)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'if_condition_7205', if_condition_7205)
                        # SSA begins for if statement (line 76)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to TypeError(...): (line 77)
                        # Processing the call arguments (line 77)
                        # Getting the type of 'localization' (line 77)
                        localization_7207 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'localization', False)
                        
                        # Call to format(...): (line 77)
                        # Processing the call arguments (line 77)
                        # Getting the type of 'callable_entity' (line 78)
                        callable_entity_7210 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 81), 'callable_entity', False)
                        # Processing the call keyword arguments (line 77)
                        kwargs_7211 = {}
                        str_7208 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 59), 'str', 'Could not create an instance of ({0}) due to constructor call errors')
                        # Obtaining the member 'format' of a type (line 77)
                        format_7209 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 59), str_7208, 'format')
                        # Calling format(args, kwargs) (line 77)
                        format_call_result_7212 = invoke(stypy.reporting.localization.Localization(__file__, 77, 59), format_7209, *[callable_entity_7210], **kwargs_7211)
                        
                        # Getting the type of 'False' (line 78)
                        False_7213 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 99), 'False', False)
                        # Processing the call keyword arguments (line 77)
                        kwargs_7214 = {}
                        # Getting the type of 'TypeError' (line 77)
                        TypeError_7206 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 77)
                        TypeError_call_result_7215 = invoke(stypy.reporting.localization.Localization(__file__, 77, 35), TypeError_7206, *[localization_7207, format_call_result_7212, False_7213], **kwargs_7214)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 77)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'stypy_return_type', TypeError_call_result_7215)
                        # SSA join for if statement (line 76)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 74)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Name (line 80):
                # Getting the type of 'call_result' (line 80)
                call_result_7216 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'call_result')
                # Assigning a type to the variable 'instance' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'instance', call_result_7216)
                
                # Assigning a Name to a Name (line 81):
                # Getting the type of 'callable_entity' (line 81)
                callable_entity_7217 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'callable_entity')
                # Assigning a type to the variable 'type_' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'type_', callable_entity_7217)
                
                # Call to instance(...): (line 82)
                # Processing the call arguments (line 82)
                # Getting the type of 'type_' (line 82)
                type__7222 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 96), 'type_', False)
                # Processing the call keyword arguments (line 82)
                # Getting the type of 'instance' (line 82)
                instance_7223 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 112), 'instance', False)
                keyword_7224 = instance_7223
                kwargs_7225 = {'instance': keyword_7224}
                # Getting the type of 'type_inference_copy' (line 82)
                type_inference_copy_7218 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'type_inference_copy', False)
                # Obtaining the member 'type_inference_proxy' of a type (line 82)
                type_inference_proxy_7219 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), type_inference_copy_7218, 'type_inference_proxy')
                # Obtaining the member 'TypeInferenceProxy' of a type (line 82)
                TypeInferenceProxy_7220 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), type_inference_proxy_7219, 'TypeInferenceProxy')
                # Obtaining the member 'instance' of a type (line 82)
                instance_7221 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), TypeInferenceProxy_7220, 'instance')
                # Calling instance(args, kwargs) (line 82)
                instance_call_result_7226 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), instance_7221, *[type__7222], **kwargs_7225)
                
                # Assigning a type to the variable 'stypy_return_type' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'stypy_return_type', instance_call_result_7226)
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_7176:
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'call_result' (line 83)
        call_result_7227 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'call_result')
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'stypy_return_type', call_result_7227)
        # SSA branch for the except part of a try statement (line 47)
        # SSA branch for the except 'Exception' branch of a try statement (line 47)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 84)
        Exception_7228 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'Exception')
        # Assigning a type to the variable 'ex' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'ex', Exception_7228)
        
        # Call to TypeError(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'localization' (line 85)
        localization_7230 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'localization', False)
        
        # Call to format(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'callable_entity' (line 87)
        callable_entity_7233 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'callable_entity', False)
        
        # Call to list(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'arg_types' (line 87)
        arg_types_7235 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 55), 'arg_types', False)
        # Processing the call keyword arguments (line 87)
        kwargs_7236 = {}
        # Getting the type of 'list' (line 87)
        list_7234 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 50), 'list', False)
        # Calling list(args, kwargs) (line 87)
        list_call_result_7237 = invoke(stypy.reporting.localization.Localization(__file__, 87, 50), list_7234, *[arg_types_7235], **kwargs_7236)
        
        
        # Call to list(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to values(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_7241 = {}
        # Getting the type of 'kwargs_types' (line 87)
        kwargs_types_7239 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 73), 'kwargs_types', False)
        # Obtaining the member 'values' of a type (line 87)
        values_7240 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 73), kwargs_types_7239, 'values')
        # Calling values(args, kwargs) (line 87)
        values_call_result_7242 = invoke(stypy.reporting.localization.Localization(__file__, 87, 73), values_7240, *[], **kwargs_7241)
        
        # Processing the call keyword arguments (line 87)
        kwargs_7243 = {}
        # Getting the type of 'list' (line 87)
        list_7238 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 68), 'list', False)
        # Calling list(args, kwargs) (line 87)
        list_call_result_7244 = invoke(stypy.reporting.localization.Localization(__file__, 87, 68), list_7238, *[values_call_result_7242], **kwargs_7243)
        
        # Applying the binary operator '+' (line 87)
        result_add_7245 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 50), '+', list_call_result_7237, list_call_result_7244)
        
        
        # Call to str(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'ex' (line 87)
        ex_7247 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 101), 'ex', False)
        # Processing the call keyword arguments (line 87)
        kwargs_7248 = {}
        # Getting the type of 'str' (line 87)
        str_7246 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 97), 'str', False)
        # Calling str(args, kwargs) (line 87)
        str_call_result_7249 = invoke(stypy.reporting.localization.Localization(__file__, 87, 97), str_7246, *[ex_7247], **kwargs_7248)
        
        # Processing the call keyword arguments (line 86)
        kwargs_7250 = {}
        str_7231 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'str', "The attempted call to '{0}' cannot be possible with parameter types {1}: {2}")
        # Obtaining the member 'format' of a type (line 86)
        format_7232 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 29), str_7231, 'format')
        # Calling format(args, kwargs) (line 86)
        format_call_result_7251 = invoke(stypy.reporting.localization.Localization(__file__, 86, 29), format_7232, *[callable_entity_7233, result_add_7245, str_call_result_7249], **kwargs_7250)
        
        # Processing the call keyword arguments (line 85)
        kwargs_7252 = {}
        # Getting the type of 'TypeError' (line 85)
        TypeError_7229 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 85)
        TypeError_call_result_7253 = invoke(stypy.reporting.localization.Localization(__file__, 85, 19), TypeError_7229, *[localization_7230, format_call_result_7251], **kwargs_7252)
        
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'stypy_return_type', TypeError_call_result_7253)
        # SSA join for try-except statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_7254 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7254)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_7254


# Assigning a type to the variable 'UserCallablesCallHandler' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'UserCallablesCallHandler', UserCallablesCallHandler)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
