
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: import sys
3: import inspect
4: 
5: from ....python_lib_copy.member_call_copy.handlers_copy.call_handler_copy import CallHandler
6: from .... import stypy_parameters_copy
7: 
8: 
9: class FileTypeModifier(CallHandler):
10:     '''
11:     Apart from type rules stored in python files, there are a second type of file called the type modifier file. This
12:     file contain functions whose name is identical to the member that they are attached to. In case a function for
13:     a member exist, this function is transferred the execution control once the member is called and a type rule is
14:     found to match with the call. Programming a type modifier is then a way to precisely control the return type of
15:      a member call, overriding the one specified by the type rule. Of course, not every member call have a type
16:      modifier associated, just those who need special treatment.
17:     '''
18: 
19:     # Cache of found type modifiers
20:     modifiers_cache = dict()
21: 
22:     # Cache of not found type modifiers
23:     unavailable_modifiers_cache = dict()
24: 
25:     @staticmethod
26:     def __modifier_files(parent_name, entity_name):
27:         '''
28:         For a call to parent_name.entity_name(...), compose the name of the type modifier file that will correspond to
29:         the entity or its parent, to look inside any of them for suitable modifiers to call
30:         :param parent_name: Parent entity (module/class) name
31:         :param entity_name: Callable entity (function/method) name
32:         :return: A tuple of (name of the rule file of the parent, name of the type rule of the entity)
33:         '''
34:         parent_modifier_file = stypy_parameters_copy.ROOT_PATH + stypy_parameters_copy.RULE_FILE_PATH + parent_name + "/" \
35:                                + parent_name + stypy_parameters_copy.type_modifier_file_postfix + ".py"
36: 
37:         own_modifier_file = stypy_parameters_copy.ROOT_PATH + stypy_parameters_copy.RULE_FILE_PATH + parent_name + "/" \
38:                             + entity_name.split('.')[-1] + "/" + entity_name.split('.')[
39:                                 -1] + stypy_parameters_copy.type_modifier_file_postfix + ".py"
40: 
41:         return parent_modifier_file, own_modifier_file
42: 
43:     def applies_to(self, proxy_obj, callable_entity):
44:         '''
45:         This method determines if this type modifier is able to respond to a call to callable_entity. The modifier
46:         respond to any callable code that has a modifier file associated. This method search the modifier file and,
47:         if found, loads and caches it for performance reasons. Cache also allows us to not to look for the same file on
48:         the hard disk over and over, saving much time. callable_entity modifier files have priority over the rule files
49:         of their parent entity should both exist.
50: 
51:         Code of this method is mostly identical to the code that searches for rule files on type_rule_call_handler
52: 
53:         :param proxy_obj: TypeInferenceProxy that hold the callable entity
54:         :param callable_entity: Callable entity
55:         :return: bool
56:         '''
57:         # We have a class, calling a class means instantiating it
58:         if inspect.isclass(callable_entity):
59:             cache_name = proxy_obj.name + ".__init__"
60:         else:
61:             cache_name = proxy_obj.name
62: 
63:         # No modifier file for this callable (from the cache)
64:         if self.unavailable_modifiers_cache.get(cache_name, False):
65:             return False
66: 
67:         # There are a modifier file for this callable (from the cache)
68:         if self.modifiers_cache.get(cache_name, False):
69:             return True
70: 
71:         # There are a modifier file for this callable parent entity (from the cache)
72:         if proxy_obj.parent_proxy is not None:
73:             if self.modifiers_cache.get(proxy_obj.parent_proxy.name, False):
74:                 return True
75: 
76:         # Obtain available rule files depending on the type of entity that is going to be called
77:         if inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity) or (
78:                     inspect.isbuiltin(callable_entity) and
79:                     (inspect.isclass(proxy_obj.parent_proxy.get_python_entity()))):
80:             try:
81:                 parent_type_rule_file, own_type_rule_file = self.__modifier_files(
82:                     callable_entity.__objclass__.__module__,
83:                     callable_entity.__objclass__.__name__,
84:                 )
85:             except:
86:                 if inspect.ismodule(proxy_obj.parent_proxy.get_python_entity()):
87:                     parent_type_rule_file, own_type_rule_file = self.__modifier_files(
88:                         proxy_obj.parent_proxy.name,
89:                         proxy_obj.parent_proxy.name)
90:                 else:
91:                     parent_type_rule_file, own_type_rule_file = self.__modifier_files(
92:                         proxy_obj.parent_proxy.parent_proxy.name,
93:                         proxy_obj.parent_proxy.name)
94:         else:
95:             parent_type_rule_file, own_type_rule_file = self.__modifier_files(proxy_obj.parent_proxy.name,
96:                                                                               proxy_obj.name)
97: 
98:         # Determine which modifier file to use
99:         parent_exist = os.path.isfile(parent_type_rule_file)
100:         own_exist = os.path.isfile(own_type_rule_file)
101:         file_path = ""
102: 
103:         if parent_exist:
104:             file_path = parent_type_rule_file
105: 
106:         if own_exist:
107:             file_path = own_type_rule_file
108: 
109:         # Load rule file
110:         if parent_exist or own_exist:
111:             dirname = os.path.dirname(file_path)
112:             file_ = file_path.split('/')[-1][0:-3]
113: 
114:             sys.path.append(dirname)
115:             module = __import__(file_, globals(), locals())
116:             entity_name = proxy_obj.name.split('.')[-1]
117:             try:
118:                 # Is there a modifier function for the specific called entity? Cache it if it is
119:                 method = getattr(module.TypeModifiers, entity_name)
120:                 self.modifiers_cache[cache_name] = method
121:             except:
122:                 # Not available: cache unavailability
123:                 self.unavailable_modifiers_cache[cache_name] = True
124:                 return False
125: 
126:         if not (parent_exist or own_exist):
127:             if proxy_obj.name not in self.unavailable_modifiers_cache:
128:                 # Not available: cache unavailability
129:                 self.unavailable_modifiers_cache[cache_name] = True
130: 
131:         return parent_exist or own_exist
132: 
133:     def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
134:         '''
135:         Calls the type modifier for callable entity to determine its return type.
136: 
137:         :param proxy_obj: TypeInferenceProxy that hold the callable entity
138:         :param localization: Caller information
139:         :param callable_entity: Callable entity
140:         :param arg_types: Arguments
141:         :param kwargs_types: Keyword arguments
142:         :return: Return type of the call
143:         '''
144:         if inspect.isclass(callable_entity):
145:             cache_name = proxy_obj.name + ".__init__"
146:         else:
147:             cache_name = proxy_obj.name
148: 
149:         modifier = self.modifiers_cache[cache_name]
150: 
151:         # Argument types passed for the call
152:         argument_types = tuple(list(arg_types) + kwargs_types.values())
153:         return modifier(localization, proxy_obj, argument_types)
154: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'import os' statement (line 1)
import os

import_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'os', os, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 2, 0))

# 'import sys' statement (line 2)
import sys

import_module(stypy.reporting.localization.Localization(__file__, 2, 0), 'sys', sys, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 3, 0))

# 'import inspect' statement (line 3)
import inspect

import_module(stypy.reporting.localization.Localization(__file__, 3, 0), 'inspect', inspect, module_type_store)

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 5, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy import CallHandler' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/type_modifiers_copy/')
import_7554 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy')

if (type(import_7554) is not StypyTypeError):

    if (import_7554 != 'pyd_module'):
        __import__(import_7554)
        sys_modules_7555 = sys.modules[import_7554]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy', sys_modules_7555.module_type_store, module_type_store, ['CallHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_7555, sys_modules_7555.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy import CallHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy', None, module_type_store, ['CallHandler'], [CallHandler])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'testing.test_programs.stypy_code_copy.stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy', import_7554)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/type_modifiers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/type_modifiers_copy/')
import_7556 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy')

if (type(import_7556) is not StypyTypeError):

    if (import_7556 != 'pyd_module'):
        __import__(import_7556)
        sys_modules_7557 = sys.modules[import_7556]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', sys_modules_7557.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_7557, sys_modules_7557.module_type_store, module_type_store)
    else:
        from testing.test_programs.stypy_code_copy.stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'testing.test_programs.stypy_code_copy.stypy_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'testing.test_programs.stypy_code_copy.stypy_copy', import_7556)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/type_modifiers_copy/')

# Declaration of the 'FileTypeModifier' class
# Getting the type of 'CallHandler' (line 9)
CallHandler_7558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'CallHandler')

class FileTypeModifier(CallHandler_7558, ):
    str_7559 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Apart from type rules stored in python files, there are a second type of file called the type modifier file. This\n    file contain functions whose name is identical to the member that they are attached to. In case a function for\n    a member exist, this function is transferred the execution control once the member is called and a type rule is\n    found to match with the call. Programming a type modifier is then a way to precisely control the return type of\n     a member call, overriding the one specified by the type rule. Of course, not every member call have a type\n     modifier associated, just those who need special treatment.\n    ')
    
    # Assigning a Call to a Name (line 20):
    
    # Assigning a Call to a Name (line 23):

    @staticmethod
    @norecursion
    def __modifier_files(localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__modifier_files'
        module_type_store = module_type_store.open_function_context('__modifier_files', 25, 4, False)
        
        # Passed parameters checking function
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_localization', localization)
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_type_of_self', None)
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_function_name', '__modifier_files')
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_param_names_list', ['parent_name', 'entity_name'])
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileTypeModifier.__modifier_files.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, None, module_type_store, '__modifier_files', ['parent_name', 'entity_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, '__modifier_files', localization, ['entity_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of '__modifier_files(...)' code ##################

        str_7560 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\n        For a call to parent_name.entity_name(...), compose the name of the type modifier file that will correspond to\n        the entity or its parent, to look inside any of them for suitable modifiers to call\n        :param parent_name: Parent entity (module/class) name\n        :param entity_name: Callable entity (function/method) name\n        :return: A tuple of (name of the rule file of the parent, name of the type rule of the entity)\n        ')
        
        # Assigning a BinOp to a Name (line 34):
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'stypy_parameters_copy' (line 34)
        stypy_parameters_copy_7561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'stypy_parameters_copy')
        # Obtaining the member 'ROOT_PATH' of a type (line 34)
        ROOT_PATH_7562 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 31), stypy_parameters_copy_7561, 'ROOT_PATH')
        # Getting the type of 'stypy_parameters_copy' (line 34)
        stypy_parameters_copy_7563 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 65), 'stypy_parameters_copy')
        # Obtaining the member 'RULE_FILE_PATH' of a type (line 34)
        RULE_FILE_PATH_7564 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 65), stypy_parameters_copy_7563, 'RULE_FILE_PATH')
        # Applying the binary operator '+' (line 34)
        result_add_7565 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 31), '+', ROOT_PATH_7562, RULE_FILE_PATH_7564)
        
        # Getting the type of 'parent_name' (line 34)
        parent_name_7566 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 104), 'parent_name')
        # Applying the binary operator '+' (line 34)
        result_add_7567 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 102), '+', result_add_7565, parent_name_7566)
        
        str_7568 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 118), 'str', '/')
        # Applying the binary operator '+' (line 34)
        result_add_7569 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 116), '+', result_add_7567, str_7568)
        
        # Getting the type of 'parent_name' (line 35)
        parent_name_7570 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'parent_name')
        # Applying the binary operator '+' (line 35)
        result_add_7571 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 31), '+', result_add_7569, parent_name_7570)
        
        # Getting the type of 'stypy_parameters_copy' (line 35)
        stypy_parameters_copy_7572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 47), 'stypy_parameters_copy')
        # Obtaining the member 'type_modifier_file_postfix' of a type (line 35)
        type_modifier_file_postfix_7573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 47), stypy_parameters_copy_7572, 'type_modifier_file_postfix')
        # Applying the binary operator '+' (line 35)
        result_add_7574 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 45), '+', result_add_7571, type_modifier_file_postfix_7573)
        
        str_7575 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 98), 'str', '.py')
        # Applying the binary operator '+' (line 35)
        result_add_7576 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 96), '+', result_add_7574, str_7575)
        
        # Assigning a type to the variable 'parent_modifier_file' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'parent_modifier_file', result_add_7576)
        
        # Assigning a BinOp to a Name (line 37):
        
        # Assigning a BinOp to a Name (line 37):
        # Getting the type of 'stypy_parameters_copy' (line 37)
        stypy_parameters_copy_7577 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 28), 'stypy_parameters_copy')
        # Obtaining the member 'ROOT_PATH' of a type (line 37)
        ROOT_PATH_7578 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 28), stypy_parameters_copy_7577, 'ROOT_PATH')
        # Getting the type of 'stypy_parameters_copy' (line 37)
        stypy_parameters_copy_7579 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 62), 'stypy_parameters_copy')
        # Obtaining the member 'RULE_FILE_PATH' of a type (line 37)
        RULE_FILE_PATH_7580 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 62), stypy_parameters_copy_7579, 'RULE_FILE_PATH')
        # Applying the binary operator '+' (line 37)
        result_add_7581 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 28), '+', ROOT_PATH_7578, RULE_FILE_PATH_7580)
        
        # Getting the type of 'parent_name' (line 37)
        parent_name_7582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 101), 'parent_name')
        # Applying the binary operator '+' (line 37)
        result_add_7583 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 99), '+', result_add_7581, parent_name_7582)
        
        str_7584 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 115), 'str', '/')
        # Applying the binary operator '+' (line 37)
        result_add_7585 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 113), '+', result_add_7583, str_7584)
        
        
        # Obtaining the type of the subscript
        int_7586 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 53), 'int')
        
        # Call to split(...): (line 38)
        # Processing the call arguments (line 38)
        str_7589 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 48), 'str', '.')
        # Processing the call keyword arguments (line 38)
        kwargs_7590 = {}
        # Getting the type of 'entity_name' (line 38)
        entity_name_7587 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'entity_name', False)
        # Obtaining the member 'split' of a type (line 38)
        split_7588 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 30), entity_name_7587, 'split')
        # Calling split(args, kwargs) (line 38)
        split_call_result_7591 = invoke(stypy.reporting.localization.Localization(__file__, 38, 30), split_7588, *[str_7589], **kwargs_7590)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___7592 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 30), split_call_result_7591, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_7593 = invoke(stypy.reporting.localization.Localization(__file__, 38, 30), getitem___7592, int_7586)
        
        # Applying the binary operator '+' (line 38)
        result_add_7594 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 28), '+', result_add_7585, subscript_call_result_7593)
        
        str_7595 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 59), 'str', '/')
        # Applying the binary operator '+' (line 38)
        result_add_7596 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 57), '+', result_add_7594, str_7595)
        
        
        # Obtaining the type of the subscript
        int_7597 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 32), 'int')
        
        # Call to split(...): (line 38)
        # Processing the call arguments (line 38)
        str_7600 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 83), 'str', '.')
        # Processing the call keyword arguments (line 38)
        kwargs_7601 = {}
        # Getting the type of 'entity_name' (line 38)
        entity_name_7598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 65), 'entity_name', False)
        # Obtaining the member 'split' of a type (line 38)
        split_7599 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 65), entity_name_7598, 'split')
        # Calling split(args, kwargs) (line 38)
        split_call_result_7602 = invoke(stypy.reporting.localization.Localization(__file__, 38, 65), split_7599, *[str_7600], **kwargs_7601)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___7603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 65), split_call_result_7602, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_7604 = invoke(stypy.reporting.localization.Localization(__file__, 38, 65), getitem___7603, int_7597)
        
        # Applying the binary operator '+' (line 38)
        result_add_7605 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 63), '+', result_add_7596, subscript_call_result_7604)
        
        # Getting the type of 'stypy_parameters_copy' (line 39)
        stypy_parameters_copy_7606 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 38), 'stypy_parameters_copy')
        # Obtaining the member 'type_modifier_file_postfix' of a type (line 39)
        type_modifier_file_postfix_7607 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 38), stypy_parameters_copy_7606, 'type_modifier_file_postfix')
        # Applying the binary operator '+' (line 39)
        result_add_7608 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 36), '+', result_add_7605, type_modifier_file_postfix_7607)
        
        str_7609 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 89), 'str', '.py')
        # Applying the binary operator '+' (line 39)
        result_add_7610 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 87), '+', result_add_7608, str_7609)
        
        # Assigning a type to the variable 'own_modifier_file' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'own_modifier_file', result_add_7610)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_7611 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'parent_modifier_file' (line 41)
        parent_modifier_file_7612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'parent_modifier_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_7611, parent_modifier_file_7612)
        # Adding element type (line 41)
        # Getting the type of 'own_modifier_file' (line 41)
        own_modifier_file_7613 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'own_modifier_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_7611, own_modifier_file_7613)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', tuple_7611)
        
        # ################# End of '__modifier_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__modifier_files' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_7614 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7614)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__modifier_files'
        return stypy_return_type_7614


    @norecursion
    def applies_to(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'applies_to'
        module_type_store = module_type_store.open_function_context('applies_to', 43, 4, False)
        # Assigning a type to the variable 'self' (line 44)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 44, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_localization', localization)
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_function_name', 'FileTypeModifier.applies_to')
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'callable_entity'])
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_varargs_param_name', None)
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_kwargs_param_name', None)
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileTypeModifier.applies_to.__dict__.__setitem__('stypy_declared_arg_number', 3)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileTypeModifier.applies_to', ['proxy_obj', 'callable_entity'], None, None, defaults, varargs, kwargs)

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

        str_7615 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n        This method determines if this type modifier is able to respond to a call to callable_entity. The modifier\n        respond to any callable code that has a modifier file associated. This method search the modifier file and,\n        if found, loads and caches it for performance reasons. Cache also allows us to not to look for the same file on\n        the hard disk over and over, saving much time. callable_entity modifier files have priority over the rule files\n        of their parent entity should both exist.\n\n        Code of this method is mostly identical to the code that searches for rule files on type_rule_call_handler\n\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param callable_entity: Callable entity\n        :return: bool\n        ')
        
        # Call to isclass(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'callable_entity' (line 58)
        callable_entity_7618 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'callable_entity', False)
        # Processing the call keyword arguments (line 58)
        kwargs_7619 = {}
        # Getting the type of 'inspect' (line 58)
        inspect_7616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 58)
        isclass_7617 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), inspect_7616, 'isclass')
        # Calling isclass(args, kwargs) (line 58)
        isclass_call_result_7620 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), isclass_7617, *[callable_entity_7618], **kwargs_7619)
        
        # Testing if the type of an if condition is none (line 58)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 8), isclass_call_result_7620):
            
            # Assigning a Attribute to a Name (line 61):
            
            # Assigning a Attribute to a Name (line 61):
            # Getting the type of 'proxy_obj' (line 61)
            proxy_obj_7626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 61)
            name_7627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), proxy_obj_7626, 'name')
            # Assigning a type to the variable 'cache_name' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'cache_name', name_7627)
        else:
            
            # Testing the type of an if condition (line 58)
            if_condition_7621 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), isclass_call_result_7620)
            # Assigning a type to the variable 'if_condition_7621' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_7621', if_condition_7621)
            # SSA begins for if statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 59):
            
            # Assigning a BinOp to a Name (line 59):
            # Getting the type of 'proxy_obj' (line 59)
            proxy_obj_7622 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 59)
            name_7623 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 25), proxy_obj_7622, 'name')
            str_7624 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 42), 'str', '.__init__')
            # Applying the binary operator '+' (line 59)
            result_add_7625 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 25), '+', name_7623, str_7624)
            
            # Assigning a type to the variable 'cache_name' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'cache_name', result_add_7625)
            # SSA branch for the else part of an if statement (line 58)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 61):
            
            # Assigning a Attribute to a Name (line 61):
            # Getting the type of 'proxy_obj' (line 61)
            proxy_obj_7626 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 61)
            name_7627 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), proxy_obj_7626, 'name')
            # Assigning a type to the variable 'cache_name' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'cache_name', name_7627)
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to get(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'cache_name' (line 64)
        cache_name_7631 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 48), 'cache_name', False)
        # Getting the type of 'False' (line 64)
        False_7632 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 60), 'False', False)
        # Processing the call keyword arguments (line 64)
        kwargs_7633 = {}
        # Getting the type of 'self' (line 64)
        self_7628 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'self', False)
        # Obtaining the member 'unavailable_modifiers_cache' of a type (line 64)
        unavailable_modifiers_cache_7629 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), self_7628, 'unavailable_modifiers_cache')
        # Obtaining the member 'get' of a type (line 64)
        get_7630 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), unavailable_modifiers_cache_7629, 'get')
        # Calling get(args, kwargs) (line 64)
        get_call_result_7634 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), get_7630, *[cache_name_7631, False_7632], **kwargs_7633)
        
        # Testing if the type of an if condition is none (line 64)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 8), get_call_result_7634):
            pass
        else:
            
            # Testing the type of an if condition (line 64)
            if_condition_7635 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), get_call_result_7634)
            # Assigning a type to the variable 'if_condition_7635' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_7635', if_condition_7635)
            # SSA begins for if statement (line 64)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 65)
            False_7636 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', False_7636)
            # SSA join for if statement (line 64)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to get(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'cache_name' (line 68)
        cache_name_7640 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'cache_name', False)
        # Getting the type of 'False' (line 68)
        False_7641 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 48), 'False', False)
        # Processing the call keyword arguments (line 68)
        kwargs_7642 = {}
        # Getting the type of 'self' (line 68)
        self_7637 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'self', False)
        # Obtaining the member 'modifiers_cache' of a type (line 68)
        modifiers_cache_7638 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), self_7637, 'modifiers_cache')
        # Obtaining the member 'get' of a type (line 68)
        get_7639 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), modifiers_cache_7638, 'get')
        # Calling get(args, kwargs) (line 68)
        get_call_result_7643 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), get_7639, *[cache_name_7640, False_7641], **kwargs_7642)
        
        # Testing if the type of an if condition is none (line 68)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 68, 8), get_call_result_7643):
            pass
        else:
            
            # Testing the type of an if condition (line 68)
            if_condition_7644 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 8), get_call_result_7643)
            # Assigning a type to the variable 'if_condition_7644' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'if_condition_7644', if_condition_7644)
            # SSA begins for if statement (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 69)
            True_7645 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'stypy_return_type', True_7645)
            # SSA join for if statement (line 68)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'proxy_obj' (line 72)
        proxy_obj_7646 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'proxy_obj')
        # Obtaining the member 'parent_proxy' of a type (line 72)
        parent_proxy_7647 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), proxy_obj_7646, 'parent_proxy')
        # Getting the type of 'None' (line 72)
        None_7648 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 41), 'None')
        # Applying the binary operator 'isnot' (line 72)
        result_is_not_7649 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), 'isnot', parent_proxy_7647, None_7648)
        
        # Testing if the type of an if condition is none (line 72)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 8), result_is_not_7649):
            pass
        else:
            
            # Testing the type of an if condition (line 72)
            if_condition_7650 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 8), result_is_not_7649)
            # Assigning a type to the variable 'if_condition_7650' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'if_condition_7650', if_condition_7650)
            # SSA begins for if statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to get(...): (line 73)
            # Processing the call arguments (line 73)
            # Getting the type of 'proxy_obj' (line 73)
            proxy_obj_7654 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 73)
            parent_proxy_7655 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 40), proxy_obj_7654, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 73)
            name_7656 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 40), parent_proxy_7655, 'name')
            # Getting the type of 'False' (line 73)
            False_7657 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 69), 'False', False)
            # Processing the call keyword arguments (line 73)
            kwargs_7658 = {}
            # Getting the type of 'self' (line 73)
            self_7651 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'self', False)
            # Obtaining the member 'modifiers_cache' of a type (line 73)
            modifiers_cache_7652 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), self_7651, 'modifiers_cache')
            # Obtaining the member 'get' of a type (line 73)
            get_7653 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), modifiers_cache_7652, 'get')
            # Calling get(args, kwargs) (line 73)
            get_call_result_7659 = invoke(stypy.reporting.localization.Localization(__file__, 73, 15), get_7653, *[name_7656, False_7657], **kwargs_7658)
            
            # Testing if the type of an if condition is none (line 73)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 12), get_call_result_7659):
                pass
            else:
                
                # Testing the type of an if condition (line 73)
                if_condition_7660 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 12), get_call_result_7659)
                # Assigning a type to the variable 'if_condition_7660' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'if_condition_7660', if_condition_7660)
                # SSA begins for if statement (line 73)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 74)
                True_7661 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'stypy_return_type', True_7661)
                # SSA join for if statement (line 73)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to ismethod(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'callable_entity' (line 77)
        callable_entity_7664 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'callable_entity', False)
        # Processing the call keyword arguments (line 77)
        kwargs_7665 = {}
        # Getting the type of 'inspect' (line 77)
        inspect_7662 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'inspect', False)
        # Obtaining the member 'ismethod' of a type (line 77)
        ismethod_7663 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), inspect_7662, 'ismethod')
        # Calling ismethod(args, kwargs) (line 77)
        ismethod_call_result_7666 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), ismethod_7663, *[callable_entity_7664], **kwargs_7665)
        
        
        # Call to ismethoddescriptor(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'callable_entity' (line 77)
        callable_entity_7669 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 75), 'callable_entity', False)
        # Processing the call keyword arguments (line 77)
        kwargs_7670 = {}
        # Getting the type of 'inspect' (line 77)
        inspect_7667 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 48), 'inspect', False)
        # Obtaining the member 'ismethoddescriptor' of a type (line 77)
        ismethoddescriptor_7668 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 48), inspect_7667, 'ismethoddescriptor')
        # Calling ismethoddescriptor(args, kwargs) (line 77)
        ismethoddescriptor_call_result_7671 = invoke(stypy.reporting.localization.Localization(__file__, 77, 48), ismethoddescriptor_7668, *[callable_entity_7669], **kwargs_7670)
        
        # Applying the binary operator 'or' (line 77)
        result_or_keyword_7672 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), 'or', ismethod_call_result_7666, ismethoddescriptor_call_result_7671)
        
        # Evaluating a boolean operation
        
        # Call to isbuiltin(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'callable_entity' (line 78)
        callable_entity_7675 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'callable_entity', False)
        # Processing the call keyword arguments (line 78)
        kwargs_7676 = {}
        # Getting the type of 'inspect' (line 78)
        inspect_7673 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'inspect', False)
        # Obtaining the member 'isbuiltin' of a type (line 78)
        isbuiltin_7674 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), inspect_7673, 'isbuiltin')
        # Calling isbuiltin(args, kwargs) (line 78)
        isbuiltin_call_result_7677 = invoke(stypy.reporting.localization.Localization(__file__, 78, 20), isbuiltin_7674, *[callable_entity_7675], **kwargs_7676)
        
        
        # Call to isclass(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to get_python_entity(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_7683 = {}
        # Getting the type of 'proxy_obj' (line 79)
        proxy_obj_7680 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 79)
        parent_proxy_7681 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 37), proxy_obj_7680, 'parent_proxy')
        # Obtaining the member 'get_python_entity' of a type (line 79)
        get_python_entity_7682 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 37), parent_proxy_7681, 'get_python_entity')
        # Calling get_python_entity(args, kwargs) (line 79)
        get_python_entity_call_result_7684 = invoke(stypy.reporting.localization.Localization(__file__, 79, 37), get_python_entity_7682, *[], **kwargs_7683)
        
        # Processing the call keyword arguments (line 79)
        kwargs_7685 = {}
        # Getting the type of 'inspect' (line 79)
        inspect_7678 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 79)
        isclass_7679 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 21), inspect_7678, 'isclass')
        # Calling isclass(args, kwargs) (line 79)
        isclass_call_result_7686 = invoke(stypy.reporting.localization.Localization(__file__, 79, 21), isclass_7679, *[get_python_entity_call_result_7684], **kwargs_7685)
        
        # Applying the binary operator 'and' (line 78)
        result_and_keyword_7687 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 20), 'and', isbuiltin_call_result_7677, isclass_call_result_7686)
        
        # Applying the binary operator 'or' (line 77)
        result_or_keyword_7688 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), 'or', result_or_keyword_7672, result_and_keyword_7687)
        
        # Testing if the type of an if condition is none (line 77)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 8), result_or_keyword_7688):
            
            # Assigning a Call to a Tuple (line 95):
            
            # Assigning a Call to a Name:
            
            # Call to __modifier_files(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'proxy_obj' (line 95)
            proxy_obj_7751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 78), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 95)
            parent_proxy_7752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 78), proxy_obj_7751, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 95)
            name_7753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 78), parent_proxy_7752, 'name')
            # Getting the type of 'proxy_obj' (line 96)
            proxy_obj_7754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 78), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 96)
            name_7755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 78), proxy_obj_7754, 'name')
            # Processing the call keyword arguments (line 95)
            kwargs_7756 = {}
            # Getting the type of 'self' (line 95)
            self_7749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 56), 'self', False)
            # Obtaining the member '__modifier_files' of a type (line 95)
            modifier_files_7750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 56), self_7749, '__modifier_files')
            # Calling __modifier_files(args, kwargs) (line 95)
            modifier_files_call_result_7757 = invoke(stypy.reporting.localization.Localization(__file__, 95, 56), modifier_files_7750, *[name_7753, name_7755], **kwargs_7756)
            
            # Assigning a type to the variable 'call_assignment_7551' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7551', modifier_files_call_result_7757)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7551' (line 95)
            call_assignment_7551_7758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7551', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7759 = stypy_get_value_from_tuple(call_assignment_7551_7758, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_7552' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7552', stypy_get_value_from_tuple_call_result_7759)
            
            # Assigning a Name to a Name (line 95):
            # Getting the type of 'call_assignment_7552' (line 95)
            call_assignment_7552_7760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7552')
            # Assigning a type to the variable 'parent_type_rule_file' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'parent_type_rule_file', call_assignment_7552_7760)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7551' (line 95)
            call_assignment_7551_7761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7551', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7762 = stypy_get_value_from_tuple(call_assignment_7551_7761, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_7553' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7553', stypy_get_value_from_tuple_call_result_7762)
            
            # Assigning a Name to a Name (line 95):
            # Getting the type of 'call_assignment_7553' (line 95)
            call_assignment_7553_7763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7553')
            # Assigning a type to the variable 'own_type_rule_file' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'own_type_rule_file', call_assignment_7553_7763)
        else:
            
            # Testing the type of an if condition (line 77)
            if_condition_7689 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), result_or_keyword_7688)
            # Assigning a type to the variable 'if_condition_7689' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_7689', if_condition_7689)
            # SSA begins for if statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # SSA begins for try-except statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Tuple (line 81):
            
            # Assigning a Call to a Name:
            
            # Call to __modifier_files(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'callable_entity' (line 82)
            callable_entity_7692 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'callable_entity', False)
            # Obtaining the member '__objclass__' of a type (line 82)
            objclass___7693 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 20), callable_entity_7692, '__objclass__')
            # Obtaining the member '__module__' of a type (line 82)
            module___7694 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 20), objclass___7693, '__module__')
            # Getting the type of 'callable_entity' (line 83)
            callable_entity_7695 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'callable_entity', False)
            # Obtaining the member '__objclass__' of a type (line 83)
            objclass___7696 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), callable_entity_7695, '__objclass__')
            # Obtaining the member '__name__' of a type (line 83)
            name___7697 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), objclass___7696, '__name__')
            # Processing the call keyword arguments (line 81)
            kwargs_7698 = {}
            # Getting the type of 'self' (line 81)
            self_7690 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 60), 'self', False)
            # Obtaining the member '__modifier_files' of a type (line 81)
            modifier_files_7691 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 60), self_7690, '__modifier_files')
            # Calling __modifier_files(args, kwargs) (line 81)
            modifier_files_call_result_7699 = invoke(stypy.reporting.localization.Localization(__file__, 81, 60), modifier_files_7691, *[module___7694, name___7697], **kwargs_7698)
            
            # Assigning a type to the variable 'call_assignment_7542' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7542', modifier_files_call_result_7699)
            
            # Assigning a Call to a Name (line 81):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7542' (line 81)
            call_assignment_7542_7700 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7542', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7701 = stypy_get_value_from_tuple(call_assignment_7542_7700, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_7543' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7543', stypy_get_value_from_tuple_call_result_7701)
            
            # Assigning a Name to a Name (line 81):
            # Getting the type of 'call_assignment_7543' (line 81)
            call_assignment_7543_7702 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7543')
            # Assigning a type to the variable 'parent_type_rule_file' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'parent_type_rule_file', call_assignment_7543_7702)
            
            # Assigning a Call to a Name (line 81):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7542' (line 81)
            call_assignment_7542_7703 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7542', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7704 = stypy_get_value_from_tuple(call_assignment_7542_7703, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_7544' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7544', stypy_get_value_from_tuple_call_result_7704)
            
            # Assigning a Name to a Name (line 81):
            # Getting the type of 'call_assignment_7544' (line 81)
            call_assignment_7544_7705 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7544')
            # Assigning a type to the variable 'own_type_rule_file' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 39), 'own_type_rule_file', call_assignment_7544_7705)
            # SSA branch for the except part of a try statement (line 80)
            # SSA branch for the except '<any exception>' branch of a try statement (line 80)
            module_type_store.open_ssa_branch('except')
            
            # Call to ismodule(...): (line 86)
            # Processing the call arguments (line 86)
            
            # Call to get_python_entity(...): (line 86)
            # Processing the call keyword arguments (line 86)
            kwargs_7711 = {}
            # Getting the type of 'proxy_obj' (line 86)
            proxy_obj_7708 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 86)
            parent_proxy_7709 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 36), proxy_obj_7708, 'parent_proxy')
            # Obtaining the member 'get_python_entity' of a type (line 86)
            get_python_entity_7710 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 36), parent_proxy_7709, 'get_python_entity')
            # Calling get_python_entity(args, kwargs) (line 86)
            get_python_entity_call_result_7712 = invoke(stypy.reporting.localization.Localization(__file__, 86, 36), get_python_entity_7710, *[], **kwargs_7711)
            
            # Processing the call keyword arguments (line 86)
            kwargs_7713 = {}
            # Getting the type of 'inspect' (line 86)
            inspect_7706 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'inspect', False)
            # Obtaining the member 'ismodule' of a type (line 86)
            ismodule_7707 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), inspect_7706, 'ismodule')
            # Calling ismodule(args, kwargs) (line 86)
            ismodule_call_result_7714 = invoke(stypy.reporting.localization.Localization(__file__, 86, 19), ismodule_7707, *[get_python_entity_call_result_7712], **kwargs_7713)
            
            # Testing if the type of an if condition is none (line 86)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 16), ismodule_call_result_7714):
                
                # Assigning a Call to a Tuple (line 91):
                
                # Assigning a Call to a Name:
                
                # Call to __modifier_files(...): (line 91)
                # Processing the call arguments (line 91)
                # Getting the type of 'proxy_obj' (line 92)
                proxy_obj_7734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 92)
                parent_proxy_7735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), proxy_obj_7734, 'parent_proxy')
                # Obtaining the member 'parent_proxy' of a type (line 92)
                parent_proxy_7736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), parent_proxy_7735, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 92)
                name_7737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), parent_proxy_7736, 'name')
                # Getting the type of 'proxy_obj' (line 93)
                proxy_obj_7738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 93)
                parent_proxy_7739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), proxy_obj_7738, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 93)
                name_7740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), parent_proxy_7739, 'name')
                # Processing the call keyword arguments (line 91)
                kwargs_7741 = {}
                # Getting the type of 'self' (line 91)
                self_7732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 64), 'self', False)
                # Obtaining the member '__modifier_files' of a type (line 91)
                modifier_files_7733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 64), self_7732, '__modifier_files')
                # Calling __modifier_files(args, kwargs) (line 91)
                modifier_files_call_result_7742 = invoke(stypy.reporting.localization.Localization(__file__, 91, 64), modifier_files_7733, *[name_7737, name_7740], **kwargs_7741)
                
                # Assigning a type to the variable 'call_assignment_7548' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7548', modifier_files_call_result_7742)
                
                # Assigning a Call to a Name (line 91):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7548' (line 91)
                call_assignment_7548_7743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7548', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7744 = stypy_get_value_from_tuple(call_assignment_7548_7743, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_7549' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7549', stypy_get_value_from_tuple_call_result_7744)
                
                # Assigning a Name to a Name (line 91):
                # Getting the type of 'call_assignment_7549' (line 91)
                call_assignment_7549_7745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7549')
                # Assigning a type to the variable 'parent_type_rule_file' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'parent_type_rule_file', call_assignment_7549_7745)
                
                # Assigning a Call to a Name (line 91):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7548' (line 91)
                call_assignment_7548_7746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7548', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7747 = stypy_get_value_from_tuple(call_assignment_7548_7746, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_7550' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7550', stypy_get_value_from_tuple_call_result_7747)
                
                # Assigning a Name to a Name (line 91):
                # Getting the type of 'call_assignment_7550' (line 91)
                call_assignment_7550_7748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7550')
                # Assigning a type to the variable 'own_type_rule_file' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'own_type_rule_file', call_assignment_7550_7748)
            else:
                
                # Testing the type of an if condition (line 86)
                if_condition_7715 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 16), ismodule_call_result_7714)
                # Assigning a type to the variable 'if_condition_7715' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'if_condition_7715', if_condition_7715)
                # SSA begins for if statement (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 87):
                
                # Assigning a Call to a Name:
                
                # Call to __modifier_files(...): (line 87)
                # Processing the call arguments (line 87)
                # Getting the type of 'proxy_obj' (line 88)
                proxy_obj_7718 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 88)
                parent_proxy_7719 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), proxy_obj_7718, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 88)
                name_7720 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), parent_proxy_7719, 'name')
                # Getting the type of 'proxy_obj' (line 89)
                proxy_obj_7721 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 89)
                parent_proxy_7722 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), proxy_obj_7721, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 89)
                name_7723 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), parent_proxy_7722, 'name')
                # Processing the call keyword arguments (line 87)
                kwargs_7724 = {}
                # Getting the type of 'self' (line 87)
                self_7716 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 64), 'self', False)
                # Obtaining the member '__modifier_files' of a type (line 87)
                modifier_files_7717 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 64), self_7716, '__modifier_files')
                # Calling __modifier_files(args, kwargs) (line 87)
                modifier_files_call_result_7725 = invoke(stypy.reporting.localization.Localization(__file__, 87, 64), modifier_files_7717, *[name_7720, name_7723], **kwargs_7724)
                
                # Assigning a type to the variable 'call_assignment_7545' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7545', modifier_files_call_result_7725)
                
                # Assigning a Call to a Name (line 87):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7545' (line 87)
                call_assignment_7545_7726 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7545', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7727 = stypy_get_value_from_tuple(call_assignment_7545_7726, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_7546' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7546', stypy_get_value_from_tuple_call_result_7727)
                
                # Assigning a Name to a Name (line 87):
                # Getting the type of 'call_assignment_7546' (line 87)
                call_assignment_7546_7728 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7546')
                # Assigning a type to the variable 'parent_type_rule_file' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'parent_type_rule_file', call_assignment_7546_7728)
                
                # Assigning a Call to a Name (line 87):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7545' (line 87)
                call_assignment_7545_7729 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7545', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7730 = stypy_get_value_from_tuple(call_assignment_7545_7729, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_7547' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7547', stypy_get_value_from_tuple_call_result_7730)
                
                # Assigning a Name to a Name (line 87):
                # Getting the type of 'call_assignment_7547' (line 87)
                call_assignment_7547_7731 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7547')
                # Assigning a type to the variable 'own_type_rule_file' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 43), 'own_type_rule_file', call_assignment_7547_7731)
                # SSA branch for the else part of an if statement (line 86)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Tuple (line 91):
                
                # Assigning a Call to a Name:
                
                # Call to __modifier_files(...): (line 91)
                # Processing the call arguments (line 91)
                # Getting the type of 'proxy_obj' (line 92)
                proxy_obj_7734 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 92)
                parent_proxy_7735 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), proxy_obj_7734, 'parent_proxy')
                # Obtaining the member 'parent_proxy' of a type (line 92)
                parent_proxy_7736 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), parent_proxy_7735, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 92)
                name_7737 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), parent_proxy_7736, 'name')
                # Getting the type of 'proxy_obj' (line 93)
                proxy_obj_7738 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 93)
                parent_proxy_7739 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), proxy_obj_7738, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 93)
                name_7740 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), parent_proxy_7739, 'name')
                # Processing the call keyword arguments (line 91)
                kwargs_7741 = {}
                # Getting the type of 'self' (line 91)
                self_7732 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 64), 'self', False)
                # Obtaining the member '__modifier_files' of a type (line 91)
                modifier_files_7733 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 64), self_7732, '__modifier_files')
                # Calling __modifier_files(args, kwargs) (line 91)
                modifier_files_call_result_7742 = invoke(stypy.reporting.localization.Localization(__file__, 91, 64), modifier_files_7733, *[name_7737, name_7740], **kwargs_7741)
                
                # Assigning a type to the variable 'call_assignment_7548' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7548', modifier_files_call_result_7742)
                
                # Assigning a Call to a Name (line 91):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7548' (line 91)
                call_assignment_7548_7743 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7548', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7744 = stypy_get_value_from_tuple(call_assignment_7548_7743, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_7549' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7549', stypy_get_value_from_tuple_call_result_7744)
                
                # Assigning a Name to a Name (line 91):
                # Getting the type of 'call_assignment_7549' (line 91)
                call_assignment_7549_7745 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7549')
                # Assigning a type to the variable 'parent_type_rule_file' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'parent_type_rule_file', call_assignment_7549_7745)
                
                # Assigning a Call to a Name (line 91):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7548' (line 91)
                call_assignment_7548_7746 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7548', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7747 = stypy_get_value_from_tuple(call_assignment_7548_7746, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_7550' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7550', stypy_get_value_from_tuple_call_result_7747)
                
                # Assigning a Name to a Name (line 91):
                # Getting the type of 'call_assignment_7550' (line 91)
                call_assignment_7550_7748 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7550')
                # Assigning a type to the variable 'own_type_rule_file' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'own_type_rule_file', call_assignment_7550_7748)
                # SSA join for if statement (line 86)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for try-except statement (line 80)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA branch for the else part of an if statement (line 77)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Call to a Tuple (line 95):
            
            # Assigning a Call to a Name:
            
            # Call to __modifier_files(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'proxy_obj' (line 95)
            proxy_obj_7751 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 78), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 95)
            parent_proxy_7752 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 78), proxy_obj_7751, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 95)
            name_7753 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 78), parent_proxy_7752, 'name')
            # Getting the type of 'proxy_obj' (line 96)
            proxy_obj_7754 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 78), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 96)
            name_7755 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 78), proxy_obj_7754, 'name')
            # Processing the call keyword arguments (line 95)
            kwargs_7756 = {}
            # Getting the type of 'self' (line 95)
            self_7749 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 56), 'self', False)
            # Obtaining the member '__modifier_files' of a type (line 95)
            modifier_files_7750 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 56), self_7749, '__modifier_files')
            # Calling __modifier_files(args, kwargs) (line 95)
            modifier_files_call_result_7757 = invoke(stypy.reporting.localization.Localization(__file__, 95, 56), modifier_files_7750, *[name_7753, name_7755], **kwargs_7756)
            
            # Assigning a type to the variable 'call_assignment_7551' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7551', modifier_files_call_result_7757)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7551' (line 95)
            call_assignment_7551_7758 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7551', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7759 = stypy_get_value_from_tuple(call_assignment_7551_7758, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_7552' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7552', stypy_get_value_from_tuple_call_result_7759)
            
            # Assigning a Name to a Name (line 95):
            # Getting the type of 'call_assignment_7552' (line 95)
            call_assignment_7552_7760 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7552')
            # Assigning a type to the variable 'parent_type_rule_file' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'parent_type_rule_file', call_assignment_7552_7760)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7551' (line 95)
            call_assignment_7551_7761 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7551', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7762 = stypy_get_value_from_tuple(call_assignment_7551_7761, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_7553' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7553', stypy_get_value_from_tuple_call_result_7762)
            
            # Assigning a Name to a Name (line 95):
            # Getting the type of 'call_assignment_7553' (line 95)
            call_assignment_7553_7763 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7553')
            # Assigning a type to the variable 'own_type_rule_file' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'own_type_rule_file', call_assignment_7553_7763)
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to isfile(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'parent_type_rule_file' (line 99)
        parent_type_rule_file_7767 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'parent_type_rule_file', False)
        # Processing the call keyword arguments (line 99)
        kwargs_7768 = {}
        # Getting the type of 'os' (line 99)
        os_7764 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 99)
        path_7765 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 23), os_7764, 'path')
        # Obtaining the member 'isfile' of a type (line 99)
        isfile_7766 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 23), path_7765, 'isfile')
        # Calling isfile(args, kwargs) (line 99)
        isfile_call_result_7769 = invoke(stypy.reporting.localization.Localization(__file__, 99, 23), isfile_7766, *[parent_type_rule_file_7767], **kwargs_7768)
        
        # Assigning a type to the variable 'parent_exist' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'parent_exist', isfile_call_result_7769)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to isfile(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'own_type_rule_file' (line 100)
        own_type_rule_file_7773 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 'own_type_rule_file', False)
        # Processing the call keyword arguments (line 100)
        kwargs_7774 = {}
        # Getting the type of 'os' (line 100)
        os_7770 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 100)
        path_7771 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), os_7770, 'path')
        # Obtaining the member 'isfile' of a type (line 100)
        isfile_7772 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), path_7771, 'isfile')
        # Calling isfile(args, kwargs) (line 100)
        isfile_call_result_7775 = invoke(stypy.reporting.localization.Localization(__file__, 100, 20), isfile_7772, *[own_type_rule_file_7773], **kwargs_7774)
        
        # Assigning a type to the variable 'own_exist' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'own_exist', isfile_call_result_7775)
        
        # Assigning a Str to a Name (line 101):
        
        # Assigning a Str to a Name (line 101):
        str_7776 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'str', '')
        # Assigning a type to the variable 'file_path' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'file_path', str_7776)
        # Getting the type of 'parent_exist' (line 103)
        parent_exist_7777 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'parent_exist')
        # Testing if the type of an if condition is none (line 103)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 103, 8), parent_exist_7777):
            pass
        else:
            
            # Testing the type of an if condition (line 103)
            if_condition_7778 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), parent_exist_7777)
            # Assigning a type to the variable 'if_condition_7778' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_7778', if_condition_7778)
            # SSA begins for if statement (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 104):
            
            # Assigning a Name to a Name (line 104):
            # Getting the type of 'parent_type_rule_file' (line 104)
            parent_type_rule_file_7779 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'parent_type_rule_file')
            # Assigning a type to the variable 'file_path' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'file_path', parent_type_rule_file_7779)
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'own_exist' (line 106)
        own_exist_7780 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'own_exist')
        # Testing if the type of an if condition is none (line 106)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 8), own_exist_7780):
            pass
        else:
            
            # Testing the type of an if condition (line 106)
            if_condition_7781 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), own_exist_7780)
            # Assigning a type to the variable 'if_condition_7781' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_7781', if_condition_7781)
            # SSA begins for if statement (line 106)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 107):
            
            # Assigning a Name to a Name (line 107):
            # Getting the type of 'own_type_rule_file' (line 107)
            own_type_rule_file_7782 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'own_type_rule_file')
            # Assigning a type to the variable 'file_path' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'file_path', own_type_rule_file_7782)
            # SSA join for if statement (line 106)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 110)
        parent_exist_7783 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'parent_exist')
        # Getting the type of 'own_exist' (line 110)
        own_exist_7784 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'own_exist')
        # Applying the binary operator 'or' (line 110)
        result_or_keyword_7785 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), 'or', parent_exist_7783, own_exist_7784)
        
        # Testing if the type of an if condition is none (line 110)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 110, 8), result_or_keyword_7785):
            pass
        else:
            
            # Testing the type of an if condition (line 110)
            if_condition_7786 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), result_or_keyword_7785)
            # Assigning a type to the variable 'if_condition_7786' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_7786', if_condition_7786)
            # SSA begins for if statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 111):
            
            # Assigning a Call to a Name (line 111):
            
            # Call to dirname(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'file_path' (line 111)
            file_path_7790 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'file_path', False)
            # Processing the call keyword arguments (line 111)
            kwargs_7791 = {}
            # Getting the type of 'os' (line 111)
            os_7787 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'os', False)
            # Obtaining the member 'path' of a type (line 111)
            path_7788 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 22), os_7787, 'path')
            # Obtaining the member 'dirname' of a type (line 111)
            dirname_7789 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 22), path_7788, 'dirname')
            # Calling dirname(args, kwargs) (line 111)
            dirname_call_result_7792 = invoke(stypy.reporting.localization.Localization(__file__, 111, 22), dirname_7789, *[file_path_7790], **kwargs_7791)
            
            # Assigning a type to the variable 'dirname' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'dirname', dirname_call_result_7792)
            
            # Assigning a Subscript to a Name (line 112):
            
            # Assigning a Subscript to a Name (line 112):
            
            # Obtaining the type of the subscript
            int_7793 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 45), 'int')
            int_7794 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 47), 'int')
            slice_7795 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 112, 20), int_7793, int_7794, None)
            
            # Obtaining the type of the subscript
            int_7796 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 41), 'int')
            
            # Call to split(...): (line 112)
            # Processing the call arguments (line 112)
            str_7799 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 36), 'str', '/')
            # Processing the call keyword arguments (line 112)
            kwargs_7800 = {}
            # Getting the type of 'file_path' (line 112)
            file_path_7797 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'file_path', False)
            # Obtaining the member 'split' of a type (line 112)
            split_7798 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), file_path_7797, 'split')
            # Calling split(args, kwargs) (line 112)
            split_call_result_7801 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), split_7798, *[str_7799], **kwargs_7800)
            
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___7802 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), split_call_result_7801, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_7803 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), getitem___7802, int_7796)
            
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___7804 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), subscript_call_result_7803, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_7805 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), getitem___7804, slice_7795)
            
            # Assigning a type to the variable 'file_' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'file_', subscript_call_result_7805)
            
            # Call to append(...): (line 114)
            # Processing the call arguments (line 114)
            # Getting the type of 'dirname' (line 114)
            dirname_7809 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'dirname', False)
            # Processing the call keyword arguments (line 114)
            kwargs_7810 = {}
            # Getting the type of 'sys' (line 114)
            sys_7806 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'sys', False)
            # Obtaining the member 'path' of a type (line 114)
            path_7807 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), sys_7806, 'path')
            # Obtaining the member 'append' of a type (line 114)
            append_7808 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), path_7807, 'append')
            # Calling append(args, kwargs) (line 114)
            append_call_result_7811 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), append_7808, *[dirname_7809], **kwargs_7810)
            
            
            # Assigning a Call to a Name (line 115):
            
            # Assigning a Call to a Name (line 115):
            
            # Call to __import__(...): (line 115)
            # Processing the call arguments (line 115)
            # Getting the type of 'file_' (line 115)
            file__7813 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'file_', False)
            
            # Call to globals(...): (line 115)
            # Processing the call keyword arguments (line 115)
            kwargs_7815 = {}
            # Getting the type of 'globals' (line 115)
            globals_7814 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 39), 'globals', False)
            # Calling globals(args, kwargs) (line 115)
            globals_call_result_7816 = invoke(stypy.reporting.localization.Localization(__file__, 115, 39), globals_7814, *[], **kwargs_7815)
            
            
            # Call to locals(...): (line 115)
            # Processing the call keyword arguments (line 115)
            kwargs_7818 = {}
            # Getting the type of 'locals' (line 115)
            locals_7817 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 50), 'locals', False)
            # Calling locals(args, kwargs) (line 115)
            locals_call_result_7819 = invoke(stypy.reporting.localization.Localization(__file__, 115, 50), locals_7817, *[], **kwargs_7818)
            
            # Processing the call keyword arguments (line 115)
            kwargs_7820 = {}
            # Getting the type of '__import__' (line 115)
            import___7812 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), '__import__', False)
            # Calling __import__(args, kwargs) (line 115)
            import___call_result_7821 = invoke(stypy.reporting.localization.Localization(__file__, 115, 21), import___7812, *[file__7813, globals_call_result_7816, locals_call_result_7819], **kwargs_7820)
            
            # Assigning a type to the variable 'module' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'module', import___call_result_7821)
            
            # Assigning a Subscript to a Name (line 116):
            
            # Assigning a Subscript to a Name (line 116):
            
            # Obtaining the type of the subscript
            int_7822 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 52), 'int')
            
            # Call to split(...): (line 116)
            # Processing the call arguments (line 116)
            str_7826 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 47), 'str', '.')
            # Processing the call keyword arguments (line 116)
            kwargs_7827 = {}
            # Getting the type of 'proxy_obj' (line 116)
            proxy_obj_7823 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 116)
            name_7824 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), proxy_obj_7823, 'name')
            # Obtaining the member 'split' of a type (line 116)
            split_7825 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), name_7824, 'split')
            # Calling split(args, kwargs) (line 116)
            split_call_result_7828 = invoke(stypy.reporting.localization.Localization(__file__, 116, 26), split_7825, *[str_7826], **kwargs_7827)
            
            # Obtaining the member '__getitem__' of a type (line 116)
            getitem___7829 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), split_call_result_7828, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 116)
            subscript_call_result_7830 = invoke(stypy.reporting.localization.Localization(__file__, 116, 26), getitem___7829, int_7822)
            
            # Assigning a type to the variable 'entity_name' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'entity_name', subscript_call_result_7830)
            
            
            # SSA begins for try-except statement (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 119):
            
            # Assigning a Call to a Name (line 119):
            
            # Call to getattr(...): (line 119)
            # Processing the call arguments (line 119)
            # Getting the type of 'module' (line 119)
            module_7832 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'module', False)
            # Obtaining the member 'TypeModifiers' of a type (line 119)
            TypeModifiers_7833 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 33), module_7832, 'TypeModifiers')
            # Getting the type of 'entity_name' (line 119)
            entity_name_7834 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 55), 'entity_name', False)
            # Processing the call keyword arguments (line 119)
            kwargs_7835 = {}
            # Getting the type of 'getattr' (line 119)
            getattr_7831 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'getattr', False)
            # Calling getattr(args, kwargs) (line 119)
            getattr_call_result_7836 = invoke(stypy.reporting.localization.Localization(__file__, 119, 25), getattr_7831, *[TypeModifiers_7833, entity_name_7834], **kwargs_7835)
            
            # Assigning a type to the variable 'method' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'method', getattr_call_result_7836)
            
            # Assigning a Name to a Subscript (line 120):
            
            # Assigning a Name to a Subscript (line 120):
            # Getting the type of 'method' (line 120)
            method_7837 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 51), 'method')
            # Getting the type of 'self' (line 120)
            self_7838 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'self')
            # Obtaining the member 'modifiers_cache' of a type (line 120)
            modifiers_cache_7839 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), self_7838, 'modifiers_cache')
            # Getting the type of 'cache_name' (line 120)
            cache_name_7840 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'cache_name')
            # Storing an element on a container (line 120)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 16), modifiers_cache_7839, (cache_name_7840, method_7837))
            # SSA branch for the except part of a try statement (line 117)
            # SSA branch for the except '<any exception>' branch of a try statement (line 117)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Subscript (line 123):
            
            # Assigning a Name to a Subscript (line 123):
            # Getting the type of 'True' (line 123)
            True_7841 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 63), 'True')
            # Getting the type of 'self' (line 123)
            self_7842 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'self')
            # Obtaining the member 'unavailable_modifiers_cache' of a type (line 123)
            unavailable_modifiers_cache_7843 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), self_7842, 'unavailable_modifiers_cache')
            # Getting the type of 'cache_name' (line 123)
            cache_name_7844 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 49), 'cache_name')
            # Storing an element on a container (line 123)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 16), unavailable_modifiers_cache_7843, (cache_name_7844, True_7841))
            # Getting the type of 'False' (line 124)
            False_7845 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'stypy_return_type', False_7845)
            # SSA join for try-except statement (line 117)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 126)
        parent_exist_7846 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'parent_exist')
        # Getting the type of 'own_exist' (line 126)
        own_exist_7847 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'own_exist')
        # Applying the binary operator 'or' (line 126)
        result_or_keyword_7848 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 16), 'or', parent_exist_7846, own_exist_7847)
        
        # Applying the 'not' unary operator (line 126)
        result_not__7849 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 11), 'not', result_or_keyword_7848)
        
        # Testing if the type of an if condition is none (line 126)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 126, 8), result_not__7849):
            pass
        else:
            
            # Testing the type of an if condition (line 126)
            if_condition_7850 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 8), result_not__7849)
            # Assigning a type to the variable 'if_condition_7850' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'if_condition_7850', if_condition_7850)
            # SSA begins for if statement (line 126)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'proxy_obj' (line 127)
            proxy_obj_7851 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 127)
            name_7852 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), proxy_obj_7851, 'name')
            # Getting the type of 'self' (line 127)
            self_7853 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'self')
            # Obtaining the member 'unavailable_modifiers_cache' of a type (line 127)
            unavailable_modifiers_cache_7854 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 37), self_7853, 'unavailable_modifiers_cache')
            # Applying the binary operator 'notin' (line 127)
            result_contains_7855 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 15), 'notin', name_7852, unavailable_modifiers_cache_7854)
            
            # Testing if the type of an if condition is none (line 127)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 127, 12), result_contains_7855):
                pass
            else:
                
                # Testing the type of an if condition (line 127)
                if_condition_7856 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), result_contains_7855)
                # Assigning a type to the variable 'if_condition_7856' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_7856', if_condition_7856)
                # SSA begins for if statement (line 127)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Subscript (line 129):
                
                # Assigning a Name to a Subscript (line 129):
                # Getting the type of 'True' (line 129)
                True_7857 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'True')
                # Getting the type of 'self' (line 129)
                self_7858 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'self')
                # Obtaining the member 'unavailable_modifiers_cache' of a type (line 129)
                unavailable_modifiers_cache_7859 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), self_7858, 'unavailable_modifiers_cache')
                # Getting the type of 'cache_name' (line 129)
                cache_name_7860 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'cache_name')
                # Storing an element on a container (line 129)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 16), unavailable_modifiers_cache_7859, (cache_name_7860, True_7857))
                # SSA join for if statement (line 127)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 126)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 131)
        parent_exist_7861 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'parent_exist')
        # Getting the type of 'own_exist' (line 131)
        own_exist_7862 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'own_exist')
        # Applying the binary operator 'or' (line 131)
        result_or_keyword_7863 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 15), 'or', parent_exist_7861, own_exist_7862)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', result_or_keyword_7863)
        
        # ################# End of 'applies_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'applies_to' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_7864 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7864)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'applies_to'
        return stypy_return_type_7864


    @norecursion
    def __call__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__call__'
        module_type_store = module_type_store.open_function_context('__call__', 133, 4, False)
        # Assigning a type to the variable 'self' (line 134)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 134, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_localization', localization)
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_type_store', module_type_store)
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_function_name', 'FileTypeModifier.__call__')
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_param_names_list', ['proxy_obj', 'localization', 'callable_entity'])
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_varargs_param_name', 'arg_types')
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_kwargs_param_name', 'kwargs_types')
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_call_defaults', defaults)
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_call_varargs', varargs)
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        FileTypeModifier.__call__.__dict__.__setitem__('stypy_declared_arg_number', 4)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'FileTypeModifier.__call__', ['proxy_obj', 'localization', 'callable_entity'], 'arg_types', 'kwargs_types', defaults, varargs, kwargs)

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

        str_7865 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', '\n        Calls the type modifier for callable entity to determine its return type.\n\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param localization: Caller information\n        :param callable_entity: Callable entity\n        :param arg_types: Arguments\n        :param kwargs_types: Keyword arguments\n        :return: Return type of the call\n        ')
        
        # Call to isclass(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'callable_entity' (line 144)
        callable_entity_7868 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'callable_entity', False)
        # Processing the call keyword arguments (line 144)
        kwargs_7869 = {}
        # Getting the type of 'inspect' (line 144)
        inspect_7866 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 144)
        isclass_7867 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 11), inspect_7866, 'isclass')
        # Calling isclass(args, kwargs) (line 144)
        isclass_call_result_7870 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), isclass_7867, *[callable_entity_7868], **kwargs_7869)
        
        # Testing if the type of an if condition is none (line 144)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 144, 8), isclass_call_result_7870):
            
            # Assigning a Attribute to a Name (line 147):
            
            # Assigning a Attribute to a Name (line 147):
            # Getting the type of 'proxy_obj' (line 147)
            proxy_obj_7876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 147)
            name_7877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), proxy_obj_7876, 'name')
            # Assigning a type to the variable 'cache_name' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'cache_name', name_7877)
        else:
            
            # Testing the type of an if condition (line 144)
            if_condition_7871 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), isclass_call_result_7870)
            # Assigning a type to the variable 'if_condition_7871' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_7871', if_condition_7871)
            # SSA begins for if statement (line 144)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 145):
            
            # Assigning a BinOp to a Name (line 145):
            # Getting the type of 'proxy_obj' (line 145)
            proxy_obj_7872 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 145)
            name_7873 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), proxy_obj_7872, 'name')
            str_7874 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 42), 'str', '.__init__')
            # Applying the binary operator '+' (line 145)
            result_add_7875 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 25), '+', name_7873, str_7874)
            
            # Assigning a type to the variable 'cache_name' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'cache_name', result_add_7875)
            # SSA branch for the else part of an if statement (line 144)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 147):
            
            # Assigning a Attribute to a Name (line 147):
            # Getting the type of 'proxy_obj' (line 147)
            proxy_obj_7876 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 147)
            name_7877 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), proxy_obj_7876, 'name')
            # Assigning a type to the variable 'cache_name' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'cache_name', name_7877)
            # SSA join for if statement (line 144)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Subscript to a Name (line 149):
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        # Getting the type of 'cache_name' (line 149)
        cache_name_7878 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 40), 'cache_name')
        # Getting the type of 'self' (line 149)
        self_7879 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'self')
        # Obtaining the member 'modifiers_cache' of a type (line 149)
        modifiers_cache_7880 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), self_7879, 'modifiers_cache')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___7881 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), modifiers_cache_7880, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_7882 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), getitem___7881, cache_name_7878)
        
        # Assigning a type to the variable 'modifier' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'modifier', subscript_call_result_7882)
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to tuple(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Call to list(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'arg_types' (line 152)
        arg_types_7885 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 36), 'arg_types', False)
        # Processing the call keyword arguments (line 152)
        kwargs_7886 = {}
        # Getting the type of 'list' (line 152)
        list_7884 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'list', False)
        # Calling list(args, kwargs) (line 152)
        list_call_result_7887 = invoke(stypy.reporting.localization.Localization(__file__, 152, 31), list_7884, *[arg_types_7885], **kwargs_7886)
        
        
        # Call to values(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_7890 = {}
        # Getting the type of 'kwargs_types' (line 152)
        kwargs_types_7888 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 49), 'kwargs_types', False)
        # Obtaining the member 'values' of a type (line 152)
        values_7889 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 49), kwargs_types_7888, 'values')
        # Calling values(args, kwargs) (line 152)
        values_call_result_7891 = invoke(stypy.reporting.localization.Localization(__file__, 152, 49), values_7889, *[], **kwargs_7890)
        
        # Applying the binary operator '+' (line 152)
        result_add_7892 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 31), '+', list_call_result_7887, values_call_result_7891)
        
        # Processing the call keyword arguments (line 152)
        kwargs_7893 = {}
        # Getting the type of 'tuple' (line 152)
        tuple_7883 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'tuple', False)
        # Calling tuple(args, kwargs) (line 152)
        tuple_call_result_7894 = invoke(stypy.reporting.localization.Localization(__file__, 152, 25), tuple_7883, *[result_add_7892], **kwargs_7893)
        
        # Assigning a type to the variable 'argument_types' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'argument_types', tuple_call_result_7894)
        
        # Call to modifier(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'localization' (line 153)
        localization_7896 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'localization', False)
        # Getting the type of 'proxy_obj' (line 153)
        proxy_obj_7897 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'proxy_obj', False)
        # Getting the type of 'argument_types' (line 153)
        argument_types_7898 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 49), 'argument_types', False)
        # Processing the call keyword arguments (line 153)
        kwargs_7899 = {}
        # Getting the type of 'modifier' (line 153)
        modifier_7895 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'modifier', False)
        # Calling modifier(args, kwargs) (line 153)
        modifier_call_result_7900 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), modifier_7895, *[localization_7896, proxy_obj_7897, argument_types_7898], **kwargs_7899)
        
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', modifier_call_result_7900)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_7901 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7901)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_7901


# Assigning a type to the variable 'FileTypeModifier' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'FileTypeModifier', FileTypeModifier)

# Assigning a Call to a Name (line 20):

# Call to dict(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_7903 = {}
# Getting the type of 'dict' (line 20)
dict_7902 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'dict', False)
# Calling dict(args, kwargs) (line 20)
dict_call_result_7904 = invoke(stypy.reporting.localization.Localization(__file__, 20, 22), dict_7902, *[], **kwargs_7903)

# Getting the type of 'FileTypeModifier'
FileTypeModifier_7905 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FileTypeModifier')
# Setting the type of the member 'modifiers_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FileTypeModifier_7905, 'modifiers_cache', dict_call_result_7904)

# Assigning a Call to a Name (line 23):

# Call to dict(...): (line 23)
# Processing the call keyword arguments (line 23)
kwargs_7907 = {}
# Getting the type of 'dict' (line 23)
dict_7906 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'dict', False)
# Calling dict(args, kwargs) (line 23)
dict_call_result_7908 = invoke(stypy.reporting.localization.Localization(__file__, 23, 34), dict_7906, *[], **kwargs_7907)

# Getting the type of 'FileTypeModifier'
FileTypeModifier_7909 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FileTypeModifier')
# Setting the type of the member 'unavailable_modifiers_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FileTypeModifier_7909, 'unavailable_modifiers_cache', dict_call_result_7908)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
