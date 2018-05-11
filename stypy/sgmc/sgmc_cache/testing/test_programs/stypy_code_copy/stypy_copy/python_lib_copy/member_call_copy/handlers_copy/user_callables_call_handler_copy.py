
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import inspect
2: 
3: from ....errors_copy.type_error_copy import TypeError
4: from call_handler_copy import CallHandler
5: from ....python_lib_copy.python_types_copy.type_inference_copy import no_recursion_copy
6: from ....python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy import is_user_defined_class, \
7:     is_user_defined_module
8: from ....python_lib_copy.python_types_copy import type_inference_copy
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

# 'from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError' statement (line 3)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7321 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy')

if (type(import_7321) is not StypyTypeError):

    if (import_7321 != 'pyd_module'):
        __import__(import_7321)
        sys_modules_7322 = sys.modules[import_7321]
        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', sys_modules_7322.module_type_store, module_type_store, ['TypeError'])
        nest_module(stypy.reporting.localization.Localization(__file__, 3, 0), __file__, sys_modules_7322, sys_modules_7322.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy import TypeError

        import_from_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', None, module_type_store, ['TypeError'], [TypeError])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy' (line 3)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 3, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.errors_copy.type_error_copy', import_7321)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 4, 0))

# 'from call_handler_copy import CallHandler' statement (line 4)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7323 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'call_handler_copy')

if (type(import_7323) is not StypyTypeError):

    if (import_7323 != 'pyd_module'):
        __import__(import_7323)
        sys_modules_7324 = sys.modules[import_7323]
        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'call_handler_copy', sys_modules_7324.module_type_store, module_type_store, ['CallHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 4, 0), __file__, sys_modules_7324, sys_modules_7324.module_type_store, module_type_store)
    else:
        from call_handler_copy import CallHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 4, 0), 'call_handler_copy', None, module_type_store, ['CallHandler'], [CallHandler])

else:
    # Assigning a type to the variable 'call_handler_copy' (line 4)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'call_handler_copy', import_7323)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import no_recursion_copy' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7325 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy')

if (type(import_7325) is not StypyTypeError):

    if (import_7325 != 'pyd_module'):
        __import__(import_7325)
        sys_modules_7326 = sys.modules[import_7325]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', sys_modules_7326.module_type_store, module_type_store, ['no_recursion_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_7326, sys_modules_7326.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import no_recursion_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', None, module_type_store, ['no_recursion_copy'], [no_recursion_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy', import_7325)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy import is_user_defined_class, is_user_defined_module' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7327 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy')

if (type(import_7327) is not StypyTypeError):

    if (import_7327 != 'pyd_module'):
        __import__(import_7327)
        sys_modules_7328 = sys.modules[import_7327]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy', sys_modules_7328.module_type_store, module_type_store, ['is_user_defined_class', 'is_user_defined_module'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_7328, sys_modules_7328.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy import is_user_defined_class, is_user_defined_module

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy', None, module_type_store, ['is_user_defined_class', 'is_user_defined_module'], [is_user_defined_class, is_user_defined_module])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy.type_inference_copy.type_inference_proxy_management_copy', import_7327)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 8, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy import type_inference_copy' statement (line 8)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')
import_7329 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy')

if (type(import_7329) is not StypyTypeError):

    if (import_7329 != 'pyd_module'):
        __import__(import_7329)
        sys_modules_7330 = sys.modules[import_7329]
        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', sys_modules_7330.module_type_store, module_type_store, ['type_inference_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 8, 0), __file__, sys_modules_7330, sys_modules_7330.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy import type_inference_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', None, module_type_store, ['type_inference_copy'], [type_inference_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy' (line 8)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 8, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.python_types_copy', import_7329)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/handlers_copy/')

# Declaration of the 'UserCallablesCallHandler' class
# Getting the type of 'CallHandler' (line 11)
CallHandler_7331 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 31), 'CallHandler')

class UserCallablesCallHandler(CallHandler_7331, ):
    str_7332 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 22, (-1)), 'str', '\n    This class handles calls to code that have been obtained from available .py source files that the program uses,\n    generated to perform type inference of the original callable code. Remember that all functions in type inference\n    programs are transformed to the following form:\n\n    def function_name(*args, **kwargs):\n        ...\n\n    This handler just use this convention to call the code and return the result. Transformed code can handle\n     UnionTypes and other stypy constructs.\n    ')

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

        str_7333 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 30, (-1)), 'str', '\n        This method determines if this call handler is able to respond to a call to callable_entity.\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param callable_entity: Callable entity\n        :return: bool\n        ')
        
        # Evaluating a boolean operation
        
        # Call to is_user_defined_class(...): (line 31)
        # Processing the call arguments (line 31)
        # Getting the type of 'callable_entity' (line 31)
        callable_entity_7335 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 37), 'callable_entity', False)
        # Processing the call keyword arguments (line 31)
        kwargs_7336 = {}
        # Getting the type of 'is_user_defined_class' (line 31)
        is_user_defined_class_7334 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 15), 'is_user_defined_class', False)
        # Calling is_user_defined_class(args, kwargs) (line 31)
        is_user_defined_class_call_result_7337 = invoke(stypy.reporting.localization.Localization(__file__, 31, 15), is_user_defined_class_7334, *[callable_entity_7335], **kwargs_7336)
        
        
        # Getting the type of 'proxy_obj' (line 31)
        proxy_obj_7338 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 57), 'proxy_obj')
        # Obtaining the member 'parent_proxy' of a type (line 31)
        parent_proxy_7339 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 57), proxy_obj_7338, 'parent_proxy')
        # Obtaining the member 'python_entity' of a type (line 31)
        python_entity_7340 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 31, 57), parent_proxy_7339, 'python_entity')
        # Getting the type of 'no_recursion_copy' (line 31)
        no_recursion_copy_7341 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 31, 97), 'no_recursion_copy')
        # Applying the binary operator '==' (line 31)
        result_eq_7342 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 57), '==', python_entity_7340, no_recursion_copy_7341)
        
        # Applying the binary operator 'or' (line 31)
        result_or_keyword_7343 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'or', is_user_defined_class_call_result_7337, result_eq_7342)
        
        # Evaluating a boolean operation
        
        # Call to ismethod(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'callable_entity' (line 32)
        callable_entity_7346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 33), 'callable_entity', False)
        # Processing the call keyword arguments (line 32)
        kwargs_7347 = {}
        # Getting the type of 'inspect' (line 32)
        inspect_7344 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 16), 'inspect', False)
        # Obtaining the member 'ismethod' of a type (line 32)
        ismethod_7345 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 16), inspect_7344, 'ismethod')
        # Calling ismethod(args, kwargs) (line 32)
        ismethod_call_result_7348 = invoke(stypy.reporting.localization.Localization(__file__, 32, 16), ismethod_7345, *[callable_entity_7346], **kwargs_7347)
        
        
        # Call to is_user_defined_class(...): (line 32)
        # Processing the call arguments (line 32)
        # Getting the type of 'proxy_obj' (line 32)
        proxy_obj_7350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 76), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 32)
        parent_proxy_7351 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 76), proxy_obj_7350, 'parent_proxy')
        # Obtaining the member 'python_entity' of a type (line 32)
        python_entity_7352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 32, 76), parent_proxy_7351, 'python_entity')
        # Processing the call keyword arguments (line 32)
        kwargs_7353 = {}
        # Getting the type of 'is_user_defined_class' (line 32)
        is_user_defined_class_7349 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 32, 54), 'is_user_defined_class', False)
        # Calling is_user_defined_class(args, kwargs) (line 32)
        is_user_defined_class_call_result_7354 = invoke(stypy.reporting.localization.Localization(__file__, 32, 54), is_user_defined_class_7349, *[python_entity_7352], **kwargs_7353)
        
        # Applying the binary operator 'and' (line 32)
        result_and_keyword_7355 = python_operator(stypy.reporting.localization.Localization(__file__, 32, 16), 'and', ismethod_call_result_7348, is_user_defined_class_call_result_7354)
        
        # Applying the binary operator 'or' (line 31)
        result_or_keyword_7356 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'or', result_or_keyword_7343, result_and_keyword_7355)
        
        # Evaluating a boolean operation
        
        # Call to isfunction(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'callable_entity' (line 33)
        callable_entity_7359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 35), 'callable_entity', False)
        # Processing the call keyword arguments (line 33)
        kwargs_7360 = {}
        # Getting the type of 'inspect' (line 33)
        inspect_7357 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 16), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 33)
        isfunction_7358 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 16), inspect_7357, 'isfunction')
        # Calling isfunction(args, kwargs) (line 33)
        isfunction_call_result_7361 = invoke(stypy.reporting.localization.Localization(__file__, 33, 16), isfunction_7358, *[callable_entity_7359], **kwargs_7360)
        
        
        # Call to is_user_defined_class(...): (line 33)
        # Processing the call arguments (line 33)
        # Getting the type of 'proxy_obj' (line 33)
        proxy_obj_7363 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 78), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 33)
        parent_proxy_7364 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 78), proxy_obj_7363, 'parent_proxy')
        # Obtaining the member 'python_entity' of a type (line 33)
        python_entity_7365 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 33, 78), parent_proxy_7364, 'python_entity')
        # Processing the call keyword arguments (line 33)
        kwargs_7366 = {}
        # Getting the type of 'is_user_defined_class' (line 33)
        is_user_defined_class_7362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 33, 56), 'is_user_defined_class', False)
        # Calling is_user_defined_class(args, kwargs) (line 33)
        is_user_defined_class_call_result_7367 = invoke(stypy.reporting.localization.Localization(__file__, 33, 56), is_user_defined_class_7362, *[python_entity_7365], **kwargs_7366)
        
        # Applying the binary operator 'and' (line 33)
        result_and_keyword_7368 = python_operator(stypy.reporting.localization.Localization(__file__, 33, 16), 'and', isfunction_call_result_7361, is_user_defined_class_call_result_7367)
        
        # Applying the binary operator 'or' (line 31)
        result_or_keyword_7369 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'or', result_or_keyword_7356, result_and_keyword_7368)
        
        # Evaluating a boolean operation
        
        # Call to isfunction(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'callable_entity' (line 34)
        callable_entity_7372 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 35), 'callable_entity', False)
        # Processing the call keyword arguments (line 34)
        kwargs_7373 = {}
        # Getting the type of 'inspect' (line 34)
        inspect_7370 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 16), 'inspect', False)
        # Obtaining the member 'isfunction' of a type (line 34)
        isfunction_7371 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 16), inspect_7370, 'isfunction')
        # Calling isfunction(args, kwargs) (line 34)
        isfunction_call_result_7374 = invoke(stypy.reporting.localization.Localization(__file__, 34, 16), isfunction_7371, *[callable_entity_7372], **kwargs_7373)
        
        
        # Call to is_user_defined_module(...): (line 34)
        # Processing the call arguments (line 34)
        # Getting the type of 'callable_entity' (line 34)
        callable_entity_7376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 79), 'callable_entity', False)
        # Obtaining the member '__module__' of a type (line 34)
        module___7377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 79), callable_entity_7376, '__module__')
        # Processing the call keyword arguments (line 34)
        kwargs_7378 = {}
        # Getting the type of 'is_user_defined_module' (line 34)
        is_user_defined_module_7375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 56), 'is_user_defined_module', False)
        # Calling is_user_defined_module(args, kwargs) (line 34)
        is_user_defined_module_call_result_7379 = invoke(stypy.reporting.localization.Localization(__file__, 34, 56), is_user_defined_module_7375, *[module___7377], **kwargs_7378)
        
        # Applying the binary operator 'and' (line 34)
        result_and_keyword_7380 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 16), 'and', isfunction_call_result_7374, is_user_defined_module_call_result_7379)
        
        # Applying the binary operator 'or' (line 31)
        result_or_keyword_7381 = python_operator(stypy.reporting.localization.Localization(__file__, 31, 15), 'or', result_or_keyword_7369, result_and_keyword_7380)
        
        # Assigning a type to the variable 'stypy_return_type' (line 31)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 31, 8), 'stypy_return_type', result_or_keyword_7381)
        
        # ################# End of 'applies_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'applies_to' in the type store
        # Getting the type of 'stypy_return_type' (line 24)
        stypy_return_type_7382 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 24, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7382)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'applies_to'
        return stypy_return_type_7382


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

        str_7383 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, (-1)), 'str', '\n        Perform the call and return the call result\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param localization: Caller information\n        :param callable_entity: Callable entity\n        :param arg_types: Arguments\n        :param kwargs_types: Keyword arguments\n        :return: Return type of the call\n        ')
        
        # Assigning a Name to a Name (line 46):
        # Getting the type of 'callable_entity' (line 46)
        callable_entity_7384 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 46, 33), 'callable_entity')
        # Assigning a type to the variable 'callable_python_entity' (line 46)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 46, 8), 'callable_python_entity', callable_entity_7384)
        
        
        # SSA begins for try-except statement (line 47)
        module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
        
        # Assigning a Call to a Name (line 48):
        
        # Call to get_error_msgs(...): (line 48)
        # Processing the call keyword arguments (line 48)
        kwargs_7387 = {}
        # Getting the type of 'TypeError' (line 48)
        TypeError_7385 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 48, 25), 'TypeError', False)
        # Obtaining the member 'get_error_msgs' of a type (line 48)
        get_error_msgs_7386 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 48, 25), TypeError_7385, 'get_error_msgs')
        # Calling get_error_msgs(args, kwargs) (line 48)
        get_error_msgs_call_result_7388 = invoke(stypy.reporting.localization.Localization(__file__, 48, 25), get_error_msgs_7386, *[], **kwargs_7387)
        
        # Assigning a type to the variable 'pre_errors' (line 48)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 48, 12), 'pre_errors', get_error_msgs_call_result_7388)
        
        # Type idiom detected: calculating its left and rigth part (line 50)
        # Getting the type of 'localization' (line 50)
        localization_7389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 15), 'localization')
        # Getting the type of 'None' (line 50)
        None_7390 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 50, 31), 'None')
        
        (may_be_7391, more_types_in_union_7392) = may_be_none(localization_7389, None_7390)

        if may_be_7391:

            if more_types_in_union_7392:
                # Runtime conditional SSA (line 50)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            # Assigning a Call to a Name (line 51):
            
            # Call to callable_python_entity(...): (line 51)
            # Getting the type of 'arg_types' (line 51)
            arg_types_7394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 54), 'arg_types', False)
            # Processing the call keyword arguments (line 51)
            # Getting the type of 'kwargs_types' (line 51)
            kwargs_types_7395 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 67), 'kwargs_types', False)
            kwargs_7396 = {'kwargs_types_7395': kwargs_types_7395}
            # Getting the type of 'callable_python_entity' (line 51)
            callable_python_entity_7393 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 51, 30), 'callable_python_entity', False)
            # Calling callable_python_entity(args, kwargs) (line 51)
            callable_python_entity_call_result_7397 = invoke(stypy.reporting.localization.Localization(__file__, 51, 30), callable_python_entity_7393, *[arg_types_7394], **kwargs_7396)
            
            # Assigning a type to the variable 'call_result' (line 51)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 51, 16), 'call_result', callable_python_entity_call_result_7397)

            if more_types_in_union_7392:
                # Runtime conditional SSA for else branch (line 50)
                module_type_store.open_ssa_branch('idiom else')



        if ((not may_be_7391) or more_types_in_union_7392):
            
            # Evaluating a boolean operation
            
            # Call to isfunction(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'proxy_obj' (line 54)
            proxy_obj_7400 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 38), 'proxy_obj', False)
            # Obtaining the member 'python_entity' of a type (line 54)
            python_entity_7401 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 38), proxy_obj_7400, 'python_entity')
            # Processing the call keyword arguments (line 54)
            kwargs_7402 = {}
            # Getting the type of 'inspect' (line 54)
            inspect_7398 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 19), 'inspect', False)
            # Obtaining the member 'isfunction' of a type (line 54)
            isfunction_7399 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 19), inspect_7398, 'isfunction')
            # Calling isfunction(args, kwargs) (line 54)
            isfunction_call_result_7403 = invoke(stypy.reporting.localization.Localization(__file__, 54, 19), isfunction_7399, *[python_entity_7401], **kwargs_7402)
            
            
            # Call to isclass(...): (line 54)
            # Processing the call arguments (line 54)
            # Getting the type of 'proxy_obj' (line 55)
            proxy_obj_7406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 55, 24), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 55)
            parent_proxy_7407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), proxy_obj_7406, 'parent_proxy')
            # Obtaining the member 'python_entity' of a type (line 55)
            python_entity_7408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 55, 24), parent_proxy_7407, 'python_entity')
            # Processing the call keyword arguments (line 54)
            kwargs_7409 = {}
            # Getting the type of 'inspect' (line 54)
            inspect_7404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 54, 67), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 54)
            isclass_7405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 54, 67), inspect_7404, 'isclass')
            # Calling isclass(args, kwargs) (line 54)
            isclass_call_result_7410 = invoke(stypy.reporting.localization.Localization(__file__, 54, 67), isclass_7405, *[python_entity_7408], **kwargs_7409)
            
            # Applying the binary operator 'and' (line 54)
            result_and_keyword_7411 = python_operator(stypy.reporting.localization.Localization(__file__, 54, 19), 'and', isfunction_call_result_7403, isclass_call_result_7410)
            
            # Testing if the type of an if condition is none (line 54)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 54, 16), result_and_keyword_7411):
                
                # Evaluating a boolean operation
                
                # Call to ismethod(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'proxy_obj' (line 60)
                proxy_obj_7424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 60)
                python_entity_7425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 40), proxy_obj_7424, 'python_entity')
                # Processing the call keyword arguments (line 60)
                kwargs_7426 = {}
                # Getting the type of 'inspect' (line 60)
                inspect_7422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'inspect', False)
                # Obtaining the member 'ismethod' of a type (line 60)
                ismethod_7423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), inspect_7422, 'ismethod')
                # Calling ismethod(args, kwargs) (line 60)
                ismethod_call_result_7427 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), ismethod_7423, *[python_entity_7425], **kwargs_7426)
                
                
                str_7428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 69), 'str', '.__init__')
                # Getting the type of 'proxy_obj' (line 60)
                proxy_obj_7429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 84), 'proxy_obj')
                # Obtaining the member 'name' of a type (line 60)
                name_7430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 84), proxy_obj_7429, 'name')
                # Applying the binary operator 'in' (line 60)
                result_contains_7431 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 69), 'in', str_7428, name_7430)
                
                # Applying the binary operator 'and' (line 60)
                result_and_keyword_7432 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), 'and', ismethod_call_result_7427, result_contains_7431)
                
                # Call to isclass(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'proxy_obj' (line 61)
                proxy_obj_7435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 61)
                parent_proxy_7436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), proxy_obj_7435, 'parent_proxy')
                # Obtaining the member 'python_entity' of a type (line 61)
                python_entity_7437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), parent_proxy_7436, 'python_entity')
                # Processing the call keyword arguments (line 60)
                kwargs_7438 = {}
                # Getting the type of 'inspect' (line 60)
                inspect_7433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 103), 'inspect', False)
                # Obtaining the member 'isclass' of a type (line 60)
                isclass_7434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 103), inspect_7433, 'isclass')
                # Calling isclass(args, kwargs) (line 60)
                isclass_call_result_7439 = invoke(stypy.reporting.localization.Localization(__file__, 60, 103), isclass_7434, *[python_entity_7437], **kwargs_7438)
                
                # Applying the binary operator 'and' (line 60)
                result_and_keyword_7440 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), 'and', result_and_keyword_7432, isclass_call_result_7439)
                
                # Testing if the type of an if condition is none (line 60)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 20), result_and_keyword_7440):
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to callable_python_entity(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'localization' (line 66)
                    localization_7454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'localization', False)
                    # Getting the type of 'arg_types' (line 66)
                    arg_types_7455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 76), 'arg_types', False)
                    # Processing the call keyword arguments (line 66)
                    # Getting the type of 'kwargs_types' (line 66)
                    kwargs_types_7456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 89), 'kwargs_types', False)
                    kwargs_7457 = {'kwargs_types_7456': kwargs_types_7456}
                    # Getting the type of 'callable_python_entity' (line 66)
                    callable_python_entity_7453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 66)
                    callable_python_entity_call_result_7458 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), callable_python_entity_7453, *[localization_7454, arg_types_7455], **kwargs_7457)
                    
                    # Assigning a type to the variable 'call_result' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'call_result', callable_python_entity_call_result_7458)
                else:
                    
                    # Testing the type of an if condition (line 60)
                    if_condition_7441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 20), result_and_keyword_7440)
                    # Assigning a type to the variable 'if_condition_7441' (line 60)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'if_condition_7441', if_condition_7441)
                    # SSA begins for if statement (line 60)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 62):
                    
                    # Call to callable_python_entity(...): (line 62)
                    # Processing the call arguments (line 62)
                    
                    # Obtaining the type of the subscript
                    int_7443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 71), 'int')
                    # Getting the type of 'arg_types' (line 62)
                    arg_types_7444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 61), 'arg_types', False)
                    # Obtaining the member '__getitem__' of a type (line 62)
                    getitem___7445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 61), arg_types_7444, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
                    subscript_call_result_7446 = invoke(stypy.reporting.localization.Localization(__file__, 62, 61), getitem___7445, int_7443)
                    
                    # Obtaining the member 'python_entity' of a type (line 62)
                    python_entity_7447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 61), subscript_call_result_7446, 'python_entity')
                    # Getting the type of 'localization' (line 63)
                    localization_7448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'localization', False)
                    # Getting the type of 'arg_types' (line 63)
                    arg_types_7449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 52), 'arg_types', False)
                    # Processing the call keyword arguments (line 62)
                    # Getting the type of 'kwargs_types' (line 63)
                    kwargs_types_7450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 65), 'kwargs_types', False)
                    kwargs_7451 = {'kwargs_types_7450': kwargs_types_7450}
                    # Getting the type of 'callable_python_entity' (line 62)
                    callable_python_entity_7442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 62)
                    callable_python_entity_call_result_7452 = invoke(stypy.reporting.localization.Localization(__file__, 62, 38), callable_python_entity_7442, *[python_entity_7447, localization_7448, arg_types_7449], **kwargs_7451)
                    
                    # Assigning a type to the variable 'call_result' (line 62)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'call_result', callable_python_entity_call_result_7452)
                    # SSA branch for the else part of an if statement (line 60)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to callable_python_entity(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'localization' (line 66)
                    localization_7454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'localization', False)
                    # Getting the type of 'arg_types' (line 66)
                    arg_types_7455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 76), 'arg_types', False)
                    # Processing the call keyword arguments (line 66)
                    # Getting the type of 'kwargs_types' (line 66)
                    kwargs_types_7456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 89), 'kwargs_types', False)
                    kwargs_7457 = {'kwargs_types_7456': kwargs_types_7456}
                    # Getting the type of 'callable_python_entity' (line 66)
                    callable_python_entity_7453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 66)
                    callable_python_entity_call_result_7458 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), callable_python_entity_7453, *[localization_7454, arg_types_7455], **kwargs_7457)
                    
                    # Assigning a type to the variable 'call_result' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'call_result', callable_python_entity_call_result_7458)
                    # SSA join for if statement (line 60)
                    module_type_store = module_type_store.join_ssa_context()
                    

            else:
                
                # Testing the type of an if condition (line 54)
                if_condition_7412 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 54, 16), result_and_keyword_7411)
                # Assigning a type to the variable 'if_condition_7412' (line 54)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 54, 16), 'if_condition_7412', if_condition_7412)
                # SSA begins for if statement (line 54)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Name (line 56):
                
                # Call to callable_python_entity(...): (line 56)
                # Processing the call arguments (line 56)
                # Getting the type of 'proxy_obj' (line 56)
                proxy_obj_7414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 57), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 56)
                parent_proxy_7415 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 57), proxy_obj_7414, 'parent_proxy')
                # Obtaining the member 'python_entity' of a type (line 56)
                python_entity_7416 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 56, 57), parent_proxy_7415, 'python_entity')
                # Getting the type of 'localization' (line 57)
                localization_7417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 57), 'localization', False)
                # Getting the type of 'arg_types' (line 57)
                arg_types_7418 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 72), 'arg_types', False)
                # Processing the call keyword arguments (line 56)
                # Getting the type of 'kwargs_types' (line 57)
                kwargs_types_7419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 57, 85), 'kwargs_types', False)
                kwargs_7420 = {'kwargs_types_7419': kwargs_types_7419}
                # Getting the type of 'callable_python_entity' (line 56)
                callable_python_entity_7413 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 56, 34), 'callable_python_entity', False)
                # Calling callable_python_entity(args, kwargs) (line 56)
                callable_python_entity_call_result_7421 = invoke(stypy.reporting.localization.Localization(__file__, 56, 34), callable_python_entity_7413, *[python_entity_7416, localization_7417, arg_types_7418], **kwargs_7420)
                
                # Assigning a type to the variable 'call_result' (line 56)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 56, 20), 'call_result', callable_python_entity_call_result_7421)
                # SSA branch for the else part of an if statement (line 54)
                module_type_store.open_ssa_branch('else')
                
                # Evaluating a boolean operation
                
                # Call to ismethod(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'proxy_obj' (line 60)
                proxy_obj_7424 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 40), 'proxy_obj', False)
                # Obtaining the member 'python_entity' of a type (line 60)
                python_entity_7425 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 40), proxy_obj_7424, 'python_entity')
                # Processing the call keyword arguments (line 60)
                kwargs_7426 = {}
                # Getting the type of 'inspect' (line 60)
                inspect_7422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 23), 'inspect', False)
                # Obtaining the member 'ismethod' of a type (line 60)
                ismethod_7423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 23), inspect_7422, 'ismethod')
                # Calling ismethod(args, kwargs) (line 60)
                ismethod_call_result_7427 = invoke(stypy.reporting.localization.Localization(__file__, 60, 23), ismethod_7423, *[python_entity_7425], **kwargs_7426)
                
                
                str_7428 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 60, 69), 'str', '.__init__')
                # Getting the type of 'proxy_obj' (line 60)
                proxy_obj_7429 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 84), 'proxy_obj')
                # Obtaining the member 'name' of a type (line 60)
                name_7430 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 84), proxy_obj_7429, 'name')
                # Applying the binary operator 'in' (line 60)
                result_contains_7431 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 69), 'in', str_7428, name_7430)
                
                # Applying the binary operator 'and' (line 60)
                result_and_keyword_7432 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), 'and', ismethod_call_result_7427, result_contains_7431)
                
                # Call to isclass(...): (line 60)
                # Processing the call arguments (line 60)
                # Getting the type of 'proxy_obj' (line 61)
                proxy_obj_7435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 61)
                parent_proxy_7436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), proxy_obj_7435, 'parent_proxy')
                # Obtaining the member 'python_entity' of a type (line 61)
                python_entity_7437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 24), parent_proxy_7436, 'python_entity')
                # Processing the call keyword arguments (line 60)
                kwargs_7438 = {}
                # Getting the type of 'inspect' (line 60)
                inspect_7433 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 60, 103), 'inspect', False)
                # Obtaining the member 'isclass' of a type (line 60)
                isclass_7434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 60, 103), inspect_7433, 'isclass')
                # Calling isclass(args, kwargs) (line 60)
                isclass_call_result_7439 = invoke(stypy.reporting.localization.Localization(__file__, 60, 103), isclass_7434, *[python_entity_7437], **kwargs_7438)
                
                # Applying the binary operator 'and' (line 60)
                result_and_keyword_7440 = python_operator(stypy.reporting.localization.Localization(__file__, 60, 23), 'and', result_and_keyword_7432, isclass_call_result_7439)
                
                # Testing if the type of an if condition is none (line 60)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 60, 20), result_and_keyword_7440):
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to callable_python_entity(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'localization' (line 66)
                    localization_7454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'localization', False)
                    # Getting the type of 'arg_types' (line 66)
                    arg_types_7455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 76), 'arg_types', False)
                    # Processing the call keyword arguments (line 66)
                    # Getting the type of 'kwargs_types' (line 66)
                    kwargs_types_7456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 89), 'kwargs_types', False)
                    kwargs_7457 = {'kwargs_types_7456': kwargs_types_7456}
                    # Getting the type of 'callable_python_entity' (line 66)
                    callable_python_entity_7453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 66)
                    callable_python_entity_call_result_7458 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), callable_python_entity_7453, *[localization_7454, arg_types_7455], **kwargs_7457)
                    
                    # Assigning a type to the variable 'call_result' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'call_result', callable_python_entity_call_result_7458)
                else:
                    
                    # Testing the type of an if condition (line 60)
                    if_condition_7441 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 60, 20), result_and_keyword_7440)
                    # Assigning a type to the variable 'if_condition_7441' (line 60)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 60, 20), 'if_condition_7441', if_condition_7441)
                    # SSA begins for if statement (line 60)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 62):
                    
                    # Call to callable_python_entity(...): (line 62)
                    # Processing the call arguments (line 62)
                    
                    # Obtaining the type of the subscript
                    int_7443 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 62, 71), 'int')
                    # Getting the type of 'arg_types' (line 62)
                    arg_types_7444 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 61), 'arg_types', False)
                    # Obtaining the member '__getitem__' of a type (line 62)
                    getitem___7445 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 61), arg_types_7444, '__getitem__')
                    # Calling the subscript (__getitem__) to obtain the elements type (line 62)
                    subscript_call_result_7446 = invoke(stypy.reporting.localization.Localization(__file__, 62, 61), getitem___7445, int_7443)
                    
                    # Obtaining the member 'python_entity' of a type (line 62)
                    python_entity_7447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 62, 61), subscript_call_result_7446, 'python_entity')
                    # Getting the type of 'localization' (line 63)
                    localization_7448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 37), 'localization', False)
                    # Getting the type of 'arg_types' (line 63)
                    arg_types_7449 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 52), 'arg_types', False)
                    # Processing the call keyword arguments (line 62)
                    # Getting the type of 'kwargs_types' (line 63)
                    kwargs_types_7450 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 63, 65), 'kwargs_types', False)
                    kwargs_7451 = {'kwargs_types_7450': kwargs_types_7450}
                    # Getting the type of 'callable_python_entity' (line 62)
                    callable_python_entity_7442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 62, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 62)
                    callable_python_entity_call_result_7452 = invoke(stypy.reporting.localization.Localization(__file__, 62, 38), callable_python_entity_7442, *[python_entity_7447, localization_7448, arg_types_7449], **kwargs_7451)
                    
                    # Assigning a type to the variable 'call_result' (line 62)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 62, 24), 'call_result', callable_python_entity_call_result_7452)
                    # SSA branch for the else part of an if statement (line 60)
                    module_type_store.open_ssa_branch('else')
                    
                    # Assigning a Call to a Name (line 66):
                    
                    # Call to callable_python_entity(...): (line 66)
                    # Processing the call arguments (line 66)
                    # Getting the type of 'localization' (line 66)
                    localization_7454 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 61), 'localization', False)
                    # Getting the type of 'arg_types' (line 66)
                    arg_types_7455 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 76), 'arg_types', False)
                    # Processing the call keyword arguments (line 66)
                    # Getting the type of 'kwargs_types' (line 66)
                    kwargs_types_7456 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 89), 'kwargs_types', False)
                    kwargs_7457 = {'kwargs_types_7456': kwargs_types_7456}
                    # Getting the type of 'callable_python_entity' (line 66)
                    callable_python_entity_7453 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 66, 38), 'callable_python_entity', False)
                    # Calling callable_python_entity(args, kwargs) (line 66)
                    callable_python_entity_call_result_7458 = invoke(stypy.reporting.localization.Localization(__file__, 66, 38), callable_python_entity_7453, *[localization_7454, arg_types_7455], **kwargs_7457)
                    
                    # Assigning a type to the variable 'call_result' (line 66)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 66, 24), 'call_result', callable_python_entity_call_result_7458)
                    # SSA join for if statement (line 60)
                    module_type_store = module_type_store.join_ssa_context()
                    

                # SSA join for if statement (line 54)
                module_type_store = module_type_store.join_ssa_context()
                


            if (may_be_7391 and more_types_in_union_7392):
                # SSA join for if statement (line 50)
                module_type_store = module_type_store.join_ssa_context()


        
        
        # Type idiom detected: calculating its left and rigth part (line 69)
        # Getting the type of 'call_result' (line 69)
        call_result_7459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'call_result')
        # Getting the type of 'None' (line 69)
        None_7460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 34), 'None')
        
        (may_be_7461, more_types_in_union_7462) = may_not_be_none(call_result_7459, None_7460)

        if may_be_7461:

            if more_types_in_union_7462:
                # Runtime conditional SSA (line 69)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'idiom if')
            else:
                module_type_store = module_type_store

            
            
            # Call to isclass(...): (line 70)
            # Processing the call arguments (line 70)
            # Getting the type of 'callable_entity' (line 70)
            callable_entity_7465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 39), 'callable_entity', False)
            # Processing the call keyword arguments (line 70)
            kwargs_7466 = {}
            # Getting the type of 'inspect' (line 70)
            inspect_7463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 70, 23), 'inspect', False)
            # Obtaining the member 'isclass' of a type (line 70)
            isclass_7464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 70, 23), inspect_7463, 'isclass')
            # Calling isclass(args, kwargs) (line 70)
            isclass_call_result_7467 = invoke(stypy.reporting.localization.Localization(__file__, 70, 23), isclass_7464, *[callable_entity_7465], **kwargs_7466)
            
            # Applying the 'not' unary operator (line 70)
            result_not__7468 = python_operator(stypy.reporting.localization.Localization(__file__, 70, 19), 'not', isclass_call_result_7467)
            
            # Testing if the type of an if condition is none (line 70)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 70, 16), result_not__7468):
                
                # Call to isinstance(...): (line 74)
                # Processing the call arguments (line 74)
                # Getting the type of 'call_result' (line 74)
                call_result_7472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'call_result', False)
                # Getting the type of 'callable_entity' (line 74)
                callable_entity_7473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 47), 'callable_entity', False)
                # Processing the call keyword arguments (line 74)
                kwargs_7474 = {}
                # Getting the type of 'isinstance' (line 74)
                isinstance_7471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 74)
                isinstance_call_result_7475 = invoke(stypy.reporting.localization.Localization(__file__, 74, 23), isinstance_7471, *[call_result_7472, callable_entity_7473], **kwargs_7474)
                
                # Testing if the type of an if condition is none (line 74)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 20), isinstance_call_result_7475):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 74)
                    if_condition_7476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 20), isinstance_call_result_7475)
                    # Assigning a type to the variable 'if_condition_7476' (line 74)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'if_condition_7476', if_condition_7476)
                    # SSA begins for if statement (line 74)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 75):
                    
                    # Call to get_error_msgs(...): (line 75)
                    # Processing the call keyword arguments (line 75)
                    kwargs_7479 = {}
                    # Getting the type of 'TypeError' (line 75)
                    TypeError_7477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'TypeError', False)
                    # Obtaining the member 'get_error_msgs' of a type (line 75)
                    get_error_msgs_7478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 38), TypeError_7477, 'get_error_msgs')
                    # Calling get_error_msgs(args, kwargs) (line 75)
                    get_error_msgs_call_result_7480 = invoke(stypy.reporting.localization.Localization(__file__, 75, 38), get_error_msgs_7478, *[], **kwargs_7479)
                    
                    # Assigning a type to the variable 'post_errors' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'post_errors', get_error_msgs_call_result_7480)
                    
                    
                    
                    # Call to len(...): (line 76)
                    # Processing the call arguments (line 76)
                    # Getting the type of 'pre_errors' (line 76)
                    pre_errors_7482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 35), 'pre_errors', False)
                    # Processing the call keyword arguments (line 76)
                    kwargs_7483 = {}
                    # Getting the type of 'len' (line 76)
                    len_7481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'len', False)
                    # Calling len(args, kwargs) (line 76)
                    len_call_result_7484 = invoke(stypy.reporting.localization.Localization(__file__, 76, 31), len_7481, *[pre_errors_7482], **kwargs_7483)
                    
                    
                    # Call to len(...): (line 76)
                    # Processing the call arguments (line 76)
                    # Getting the type of 'post_errors' (line 76)
                    post_errors_7486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 54), 'post_errors', False)
                    # Processing the call keyword arguments (line 76)
                    kwargs_7487 = {}
                    # Getting the type of 'len' (line 76)
                    len_7485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 50), 'len', False)
                    # Calling len(args, kwargs) (line 76)
                    len_call_result_7488 = invoke(stypy.reporting.localization.Localization(__file__, 76, 50), len_7485, *[post_errors_7486], **kwargs_7487)
                    
                    # Applying the binary operator '==' (line 76)
                    result_eq_7489 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 31), '==', len_call_result_7484, len_call_result_7488)
                    
                    # Applying the 'not' unary operator (line 76)
                    result_not__7490 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 27), 'not', result_eq_7489)
                    
                    # Testing if the type of an if condition is none (line 76)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 24), result_not__7490):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 76)
                        if_condition_7491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 24), result_not__7490)
                        # Assigning a type to the variable 'if_condition_7491' (line 76)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'if_condition_7491', if_condition_7491)
                        # SSA begins for if statement (line 76)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to TypeError(...): (line 77)
                        # Processing the call arguments (line 77)
                        # Getting the type of 'localization' (line 77)
                        localization_7493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'localization', False)
                        
                        # Call to format(...): (line 77)
                        # Processing the call arguments (line 77)
                        # Getting the type of 'callable_entity' (line 78)
                        callable_entity_7496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 81), 'callable_entity', False)
                        # Processing the call keyword arguments (line 77)
                        kwargs_7497 = {}
                        str_7494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 59), 'str', 'Could not create an instance of ({0}) due to constructor call errors')
                        # Obtaining the member 'format' of a type (line 77)
                        format_7495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 59), str_7494, 'format')
                        # Calling format(args, kwargs) (line 77)
                        format_call_result_7498 = invoke(stypy.reporting.localization.Localization(__file__, 77, 59), format_7495, *[callable_entity_7496], **kwargs_7497)
                        
                        # Getting the type of 'False' (line 78)
                        False_7499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 99), 'False', False)
                        # Processing the call keyword arguments (line 77)
                        kwargs_7500 = {}
                        # Getting the type of 'TypeError' (line 77)
                        TypeError_7492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 77)
                        TypeError_call_result_7501 = invoke(stypy.reporting.localization.Localization(__file__, 77, 35), TypeError_7492, *[localization_7493, format_call_result_7498, False_7499], **kwargs_7500)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 77)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'stypy_return_type', TypeError_call_result_7501)
                        # SSA join for if statement (line 76)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 74)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Name (line 80):
                # Getting the type of 'call_result' (line 80)
                call_result_7502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'call_result')
                # Assigning a type to the variable 'instance' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'instance', call_result_7502)
                
                # Assigning a Name to a Name (line 81):
                # Getting the type of 'callable_entity' (line 81)
                callable_entity_7503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'callable_entity')
                # Assigning a type to the variable 'type_' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'type_', callable_entity_7503)
                
                # Call to instance(...): (line 82)
                # Processing the call arguments (line 82)
                # Getting the type of 'type_' (line 82)
                type__7508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 96), 'type_', False)
                # Processing the call keyword arguments (line 82)
                # Getting the type of 'instance' (line 82)
                instance_7509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 112), 'instance', False)
                keyword_7510 = instance_7509
                kwargs_7511 = {'instance': keyword_7510}
                # Getting the type of 'type_inference_copy' (line 82)
                type_inference_copy_7504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'type_inference_copy', False)
                # Obtaining the member 'type_inference_proxy' of a type (line 82)
                type_inference_proxy_7505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), type_inference_copy_7504, 'type_inference_proxy')
                # Obtaining the member 'TypeInferenceProxy' of a type (line 82)
                TypeInferenceProxy_7506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), type_inference_proxy_7505, 'TypeInferenceProxy')
                # Obtaining the member 'instance' of a type (line 82)
                instance_7507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), TypeInferenceProxy_7506, 'instance')
                # Calling instance(args, kwargs) (line 82)
                instance_call_result_7512 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), instance_7507, *[type__7508], **kwargs_7511)
                
                # Assigning a type to the variable 'stypy_return_type' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'stypy_return_type', instance_call_result_7512)
            else:
                
                # Testing the type of an if condition (line 70)
                if_condition_7469 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 70, 16), result_not__7468)
                # Assigning a type to the variable 'if_condition_7469' (line 70)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 70, 16), 'if_condition_7469', if_condition_7469)
                # SSA begins for if statement (line 70)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'call_result' (line 71)
                call_result_7470 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 71, 27), 'call_result')
                # Assigning a type to the variable 'stypy_return_type' (line 71)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 71, 20), 'stypy_return_type', call_result_7470)
                # SSA branch for the else part of an if statement (line 70)
                module_type_store.open_ssa_branch('else')
                
                # Call to isinstance(...): (line 74)
                # Processing the call arguments (line 74)
                # Getting the type of 'call_result' (line 74)
                call_result_7472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 34), 'call_result', False)
                # Getting the type of 'callable_entity' (line 74)
                callable_entity_7473 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 47), 'callable_entity', False)
                # Processing the call keyword arguments (line 74)
                kwargs_7474 = {}
                # Getting the type of 'isinstance' (line 74)
                isinstance_7471 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'isinstance', False)
                # Calling isinstance(args, kwargs) (line 74)
                isinstance_call_result_7475 = invoke(stypy.reporting.localization.Localization(__file__, 74, 23), isinstance_7471, *[call_result_7472, callable_entity_7473], **kwargs_7474)
                
                # Testing if the type of an if condition is none (line 74)

                if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 74, 20), isinstance_call_result_7475):
                    pass
                else:
                    
                    # Testing the type of an if condition (line 74)
                    if_condition_7476 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 74, 20), isinstance_call_result_7475)
                    # Assigning a type to the variable 'if_condition_7476' (line 74)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 20), 'if_condition_7476', if_condition_7476)
                    # SSA begins for if statement (line 74)
                    module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                    
                    # Assigning a Call to a Name (line 75):
                    
                    # Call to get_error_msgs(...): (line 75)
                    # Processing the call keyword arguments (line 75)
                    kwargs_7479 = {}
                    # Getting the type of 'TypeError' (line 75)
                    TypeError_7477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 75, 38), 'TypeError', False)
                    # Obtaining the member 'get_error_msgs' of a type (line 75)
                    get_error_msgs_7478 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 75, 38), TypeError_7477, 'get_error_msgs')
                    # Calling get_error_msgs(args, kwargs) (line 75)
                    get_error_msgs_call_result_7480 = invoke(stypy.reporting.localization.Localization(__file__, 75, 38), get_error_msgs_7478, *[], **kwargs_7479)
                    
                    # Assigning a type to the variable 'post_errors' (line 75)
                    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 75, 24), 'post_errors', get_error_msgs_call_result_7480)
                    
                    
                    
                    # Call to len(...): (line 76)
                    # Processing the call arguments (line 76)
                    # Getting the type of 'pre_errors' (line 76)
                    pre_errors_7482 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 35), 'pre_errors', False)
                    # Processing the call keyword arguments (line 76)
                    kwargs_7483 = {}
                    # Getting the type of 'len' (line 76)
                    len_7481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 31), 'len', False)
                    # Calling len(args, kwargs) (line 76)
                    len_call_result_7484 = invoke(stypy.reporting.localization.Localization(__file__, 76, 31), len_7481, *[pre_errors_7482], **kwargs_7483)
                    
                    
                    # Call to len(...): (line 76)
                    # Processing the call arguments (line 76)
                    # Getting the type of 'post_errors' (line 76)
                    post_errors_7486 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 54), 'post_errors', False)
                    # Processing the call keyword arguments (line 76)
                    kwargs_7487 = {}
                    # Getting the type of 'len' (line 76)
                    len_7485 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 76, 50), 'len', False)
                    # Calling len(args, kwargs) (line 76)
                    len_call_result_7488 = invoke(stypy.reporting.localization.Localization(__file__, 76, 50), len_7485, *[post_errors_7486], **kwargs_7487)
                    
                    # Applying the binary operator '==' (line 76)
                    result_eq_7489 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 31), '==', len_call_result_7484, len_call_result_7488)
                    
                    # Applying the 'not' unary operator (line 76)
                    result_not__7490 = python_operator(stypy.reporting.localization.Localization(__file__, 76, 27), 'not', result_eq_7489)
                    
                    # Testing if the type of an if condition is none (line 76)

                    if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 76, 24), result_not__7490):
                        pass
                    else:
                        
                        # Testing the type of an if condition (line 76)
                        if_condition_7491 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 76, 24), result_not__7490)
                        # Assigning a type to the variable 'if_condition_7491' (line 76)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 76, 24), 'if_condition_7491', if_condition_7491)
                        # SSA begins for if statement (line 76)
                        module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                        
                        # Call to TypeError(...): (line 77)
                        # Processing the call arguments (line 77)
                        # Getting the type of 'localization' (line 77)
                        localization_7493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 45), 'localization', False)
                        
                        # Call to format(...): (line 77)
                        # Processing the call arguments (line 77)
                        # Getting the type of 'callable_entity' (line 78)
                        callable_entity_7496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 81), 'callable_entity', False)
                        # Processing the call keyword arguments (line 77)
                        kwargs_7497 = {}
                        str_7494 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 77, 59), 'str', 'Could not create an instance of ({0}) due to constructor call errors')
                        # Obtaining the member 'format' of a type (line 77)
                        format_7495 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 59), str_7494, 'format')
                        # Calling format(args, kwargs) (line 77)
                        format_call_result_7498 = invoke(stypy.reporting.localization.Localization(__file__, 77, 59), format_7495, *[callable_entity_7496], **kwargs_7497)
                        
                        # Getting the type of 'False' (line 78)
                        False_7499 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 99), 'False', False)
                        # Processing the call keyword arguments (line 77)
                        kwargs_7500 = {}
                        # Getting the type of 'TypeError' (line 77)
                        TypeError_7492 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 35), 'TypeError', False)
                        # Calling TypeError(args, kwargs) (line 77)
                        TypeError_call_result_7501 = invoke(stypy.reporting.localization.Localization(__file__, 77, 35), TypeError_7492, *[localization_7493, format_call_result_7498, False_7499], **kwargs_7500)
                        
                        # Assigning a type to the variable 'stypy_return_type' (line 77)
                        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'stypy_return_type', TypeError_call_result_7501)
                        # SSA join for if statement (line 76)
                        module_type_store = module_type_store.join_ssa_context()
                        

                    # SSA join for if statement (line 74)
                    module_type_store = module_type_store.join_ssa_context()
                    

                
                # Assigning a Name to a Name (line 80):
                # Getting the type of 'call_result' (line 80)
                call_result_7502 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 80, 31), 'call_result')
                # Assigning a type to the variable 'instance' (line 80)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 80, 20), 'instance', call_result_7502)
                
                # Assigning a Name to a Name (line 81):
                # Getting the type of 'callable_entity' (line 81)
                callable_entity_7503 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 28), 'callable_entity')
                # Assigning a type to the variable 'type_' (line 81)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 20), 'type_', callable_entity_7503)
                
                # Call to instance(...): (line 82)
                # Processing the call arguments (line 82)
                # Getting the type of 'type_' (line 82)
                type__7508 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 96), 'type_', False)
                # Processing the call keyword arguments (line 82)
                # Getting the type of 'instance' (line 82)
                instance_7509 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 112), 'instance', False)
                keyword_7510 = instance_7509
                kwargs_7511 = {'instance': keyword_7510}
                # Getting the type of 'type_inference_copy' (line 82)
                type_inference_copy_7504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 27), 'type_inference_copy', False)
                # Obtaining the member 'type_inference_proxy' of a type (line 82)
                type_inference_proxy_7505 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), type_inference_copy_7504, 'type_inference_proxy')
                # Obtaining the member 'TypeInferenceProxy' of a type (line 82)
                TypeInferenceProxy_7506 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), type_inference_proxy_7505, 'TypeInferenceProxy')
                # Obtaining the member 'instance' of a type (line 82)
                instance_7507 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 27), TypeInferenceProxy_7506, 'instance')
                # Calling instance(args, kwargs) (line 82)
                instance_call_result_7512 = invoke(stypy.reporting.localization.Localization(__file__, 82, 27), instance_7507, *[type__7508], **kwargs_7511)
                
                # Assigning a type to the variable 'stypy_return_type' (line 82)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'stypy_return_type', instance_call_result_7512)
                # SSA join for if statement (line 70)
                module_type_store = module_type_store.join_ssa_context()
                


            if more_types_in_union_7462:
                # SSA join for if statement (line 69)
                module_type_store = module_type_store.join_ssa_context()


        
        # Getting the type of 'call_result' (line 83)
        call_result_7513 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 19), 'call_result')
        # Assigning a type to the variable 'stypy_return_type' (line 83)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 83, 12), 'stypy_return_type', call_result_7513)
        # SSA branch for the except part of a try statement (line 47)
        # SSA branch for the except 'Exception' branch of a try statement (line 47)
        # Storing handler type
        module_type_store.open_ssa_branch('except')
        # Getting the type of 'Exception' (line 84)
        Exception_7514 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 84, 15), 'Exception')
        # Assigning a type to the variable 'ex' (line 84)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 84, 8), 'ex', Exception_7514)
        
        # Call to TypeError(...): (line 85)
        # Processing the call arguments (line 85)
        # Getting the type of 'localization' (line 85)
        localization_7516 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 29), 'localization', False)
        
        # Call to format(...): (line 86)
        # Processing the call arguments (line 86)
        # Getting the type of 'callable_entity' (line 87)
        callable_entity_7519 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 33), 'callable_entity', False)
        
        # Call to list(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'arg_types' (line 87)
        arg_types_7521 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 55), 'arg_types', False)
        # Processing the call keyword arguments (line 87)
        kwargs_7522 = {}
        # Getting the type of 'list' (line 87)
        list_7520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 50), 'list', False)
        # Calling list(args, kwargs) (line 87)
        list_call_result_7523 = invoke(stypy.reporting.localization.Localization(__file__, 87, 50), list_7520, *[arg_types_7521], **kwargs_7522)
        
        
        # Call to list(...): (line 87)
        # Processing the call arguments (line 87)
        
        # Call to values(...): (line 87)
        # Processing the call keyword arguments (line 87)
        kwargs_7527 = {}
        # Getting the type of 'kwargs_types' (line 87)
        kwargs_types_7525 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 73), 'kwargs_types', False)
        # Obtaining the member 'values' of a type (line 87)
        values_7526 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 73), kwargs_types_7525, 'values')
        # Calling values(args, kwargs) (line 87)
        values_call_result_7528 = invoke(stypy.reporting.localization.Localization(__file__, 87, 73), values_7526, *[], **kwargs_7527)
        
        # Processing the call keyword arguments (line 87)
        kwargs_7529 = {}
        # Getting the type of 'list' (line 87)
        list_7524 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 68), 'list', False)
        # Calling list(args, kwargs) (line 87)
        list_call_result_7530 = invoke(stypy.reporting.localization.Localization(__file__, 87, 68), list_7524, *[values_call_result_7528], **kwargs_7529)
        
        # Applying the binary operator '+' (line 87)
        result_add_7531 = python_operator(stypy.reporting.localization.Localization(__file__, 87, 50), '+', list_call_result_7523, list_call_result_7530)
        
        
        # Call to str(...): (line 87)
        # Processing the call arguments (line 87)
        # Getting the type of 'ex' (line 87)
        ex_7533 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 101), 'ex', False)
        # Processing the call keyword arguments (line 87)
        kwargs_7534 = {}
        # Getting the type of 'str' (line 87)
        str_7532 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 97), 'str', False)
        # Calling str(args, kwargs) (line 87)
        str_call_result_7535 = invoke(stypy.reporting.localization.Localization(__file__, 87, 97), str_7532, *[ex_7533], **kwargs_7534)
        
        # Processing the call keyword arguments (line 86)
        kwargs_7536 = {}
        str_7517 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 86, 29), 'str', "The attempted call to '{0}' cannot be possible with parameter types {1}: {2}")
        # Obtaining the member 'format' of a type (line 86)
        format_7518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 29), str_7517, 'format')
        # Calling format(args, kwargs) (line 86)
        format_call_result_7537 = invoke(stypy.reporting.localization.Localization(__file__, 86, 29), format_7518, *[callable_entity_7519, result_add_7531, str_call_result_7535], **kwargs_7536)
        
        # Processing the call keyword arguments (line 85)
        kwargs_7538 = {}
        # Getting the type of 'TypeError' (line 85)
        TypeError_7515 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 85, 19), 'TypeError', False)
        # Calling TypeError(args, kwargs) (line 85)
        TypeError_call_result_7539 = invoke(stypy.reporting.localization.Localization(__file__, 85, 19), TypeError_7515, *[localization_7516, format_call_result_7537], **kwargs_7538)
        
        # Assigning a type to the variable 'stypy_return_type' (line 85)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 85, 12), 'stypy_return_type', TypeError_call_result_7539)
        # SSA join for try-except statement (line 47)
        module_type_store = module_type_store.join_ssa_context()
        
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 36)
        stypy_return_type_7540 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7540)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_7540


# Assigning a type to the variable 'UserCallablesCallHandler' (line 11)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 11, 0), 'UserCallablesCallHandler', UserCallablesCallHandler)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
