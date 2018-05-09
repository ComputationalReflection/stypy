
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: import os
2: import sys
3: import inspect
4: 
5: from stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy import CallHandler
6: from stypy_copy import stypy_parameters_copy
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
34:         parent_modifier_file = stypy_parameters.ROOT_PATH + stypy_parameters.RULE_FILE_PATH + parent_name + "/" \
35:                                + parent_name + stypy_parameters.type_modifier_file_postfix + ".py"
36: 
37:         own_modifier_file = stypy_parameters.ROOT_PATH + stypy_parameters.RULE_FILE_PATH + parent_name + "/" \
38:                             + entity_name.split('.')[-1] + "/" + entity_name.split('.')[
39:                                 -1] + stypy_parameters.type_modifier_file_postfix + ".py"
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

# 'from stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy import CallHandler' statement (line 5)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/type_modifiers_copy/')
import_7268 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy')

if (type(import_7268) is not StypyTypeError):

    if (import_7268 != 'pyd_module'):
        __import__(import_7268)
        sys_modules_7269 = sys.modules[import_7268]
        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy', sys_modules_7269.module_type_store, module_type_store, ['CallHandler'])
        nest_module(stypy.reporting.localization.Localization(__file__, 5, 0), __file__, sys_modules_7269, sys_modules_7269.module_type_store, module_type_store)
    else:
        from stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy import CallHandler

        import_from_module(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy', None, module_type_store, ['CallHandler'], [CallHandler])

else:
    # Assigning a type to the variable 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy' (line 5)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 5, 0), 'stypy_copy.python_lib_copy.member_call_copy.handlers_copy.call_handler_copy', import_7268)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/type_modifiers_copy/')

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 6, 0))

# 'from stypy_copy import stypy_parameters_copy' statement (line 6)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/type_modifiers_copy/')
import_7270 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy')

if (type(import_7270) is not StypyTypeError):

    if (import_7270 != 'pyd_module'):
        __import__(import_7270)
        sys_modules_7271 = sys.modules[import_7270]
        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy', sys_modules_7271.module_type_store, module_type_store, ['stypy_parameters_copy'])
        nest_module(stypy.reporting.localization.Localization(__file__, 6, 0), __file__, sys_modules_7271, sys_modules_7271.module_type_store, module_type_store)
    else:
        from stypy_copy import stypy_parameters_copy

        import_from_module(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy', None, module_type_store, ['stypy_parameters_copy'], [stypy_parameters_copy])

else:
    # Assigning a type to the variable 'stypy_copy' (line 6)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 6, 0), 'stypy_copy', import_7270)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/stypy_over_stypy_testing/stypy_code_copy/stypy_copy/python_lib_copy/member_call_copy/type_modifiers_copy/')

# Declaration of the 'FileTypeModifier' class
# Getting the type of 'CallHandler' (line 9)
CallHandler_7272 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'CallHandler')

class FileTypeModifier(CallHandler_7272, ):
    str_7273 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n    Apart from type rules stored in python files, there are a second type of file called the type modifier file. This\n    file contain functions whose name is identical to the member that they are attached to. In case a function for\n    a member exist, this function is transferred the execution control once the member is called and a type rule is\n    found to match with the call. Programming a type modifier is then a way to precisely control the return type of\n     a member call, overriding the one specified by the type rule. Of course, not every member call have a type\n     modifier associated, just those who need special treatment.\n    ')
    
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

        str_7274 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, (-1)), 'str', '\n        For a call to parent_name.entity_name(...), compose the name of the type modifier file that will correspond to\n        the entity or its parent, to look inside any of them for suitable modifiers to call\n        :param parent_name: Parent entity (module/class) name\n        :param entity_name: Callable entity (function/method) name\n        :return: A tuple of (name of the rule file of the parent, name of the type rule of the entity)\n        ')
        
        # Assigning a BinOp to a Name (line 34):
        
        # Assigning a BinOp to a Name (line 34):
        # Getting the type of 'stypy_parameters' (line 34)
        stypy_parameters_7275 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 31), 'stypy_parameters')
        # Obtaining the member 'ROOT_PATH' of a type (line 34)
        ROOT_PATH_7276 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 31), stypy_parameters_7275, 'ROOT_PATH')
        # Getting the type of 'stypy_parameters' (line 34)
        stypy_parameters_7277 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 60), 'stypy_parameters')
        # Obtaining the member 'RULE_FILE_PATH' of a type (line 34)
        RULE_FILE_PATH_7278 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 34, 60), stypy_parameters_7277, 'RULE_FILE_PATH')
        # Applying the binary operator '+' (line 34)
        result_add_7279 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 31), '+', ROOT_PATH_7276, RULE_FILE_PATH_7278)
        
        # Getting the type of 'parent_name' (line 34)
        parent_name_7280 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 94), 'parent_name')
        # Applying the binary operator '+' (line 34)
        result_add_7281 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 92), '+', result_add_7279, parent_name_7280)
        
        str_7282 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 34, 108), 'str', '/')
        # Applying the binary operator '+' (line 34)
        result_add_7283 = python_operator(stypy.reporting.localization.Localization(__file__, 34, 106), '+', result_add_7281, str_7282)
        
        # Getting the type of 'parent_name' (line 35)
        parent_name_7284 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 33), 'parent_name')
        # Applying the binary operator '+' (line 35)
        result_add_7285 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 31), '+', result_add_7283, parent_name_7284)
        
        # Getting the type of 'stypy_parameters' (line 35)
        stypy_parameters_7286 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 47), 'stypy_parameters')
        # Obtaining the member 'type_modifier_file_postfix' of a type (line 35)
        type_modifier_file_postfix_7287 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 47), stypy_parameters_7286, 'type_modifier_file_postfix')
        # Applying the binary operator '+' (line 35)
        result_add_7288 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 45), '+', result_add_7285, type_modifier_file_postfix_7287)
        
        str_7289 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 35, 93), 'str', '.py')
        # Applying the binary operator '+' (line 35)
        result_add_7290 = python_operator(stypy.reporting.localization.Localization(__file__, 35, 91), '+', result_add_7288, str_7289)
        
        # Assigning a type to the variable 'parent_modifier_file' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'parent_modifier_file', result_add_7290)
        
        # Assigning a BinOp to a Name (line 37):
        
        # Assigning a BinOp to a Name (line 37):
        # Getting the type of 'stypy_parameters' (line 37)
        stypy_parameters_7291 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 28), 'stypy_parameters')
        # Obtaining the member 'ROOT_PATH' of a type (line 37)
        ROOT_PATH_7292 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 28), stypy_parameters_7291, 'ROOT_PATH')
        # Getting the type of 'stypy_parameters' (line 37)
        stypy_parameters_7293 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 57), 'stypy_parameters')
        # Obtaining the member 'RULE_FILE_PATH' of a type (line 37)
        RULE_FILE_PATH_7294 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 37, 57), stypy_parameters_7293, 'RULE_FILE_PATH')
        # Applying the binary operator '+' (line 37)
        result_add_7295 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 28), '+', ROOT_PATH_7292, RULE_FILE_PATH_7294)
        
        # Getting the type of 'parent_name' (line 37)
        parent_name_7296 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 37, 91), 'parent_name')
        # Applying the binary operator '+' (line 37)
        result_add_7297 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 89), '+', result_add_7295, parent_name_7296)
        
        str_7298 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 37, 105), 'str', '/')
        # Applying the binary operator '+' (line 37)
        result_add_7299 = python_operator(stypy.reporting.localization.Localization(__file__, 37, 103), '+', result_add_7297, str_7298)
        
        
        # Obtaining the type of the subscript
        int_7300 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 53), 'int')
        
        # Call to split(...): (line 38)
        # Processing the call arguments (line 38)
        str_7303 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 48), 'str', '.')
        # Processing the call keyword arguments (line 38)
        kwargs_7304 = {}
        # Getting the type of 'entity_name' (line 38)
        entity_name_7301 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 30), 'entity_name', False)
        # Obtaining the member 'split' of a type (line 38)
        split_7302 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 30), entity_name_7301, 'split')
        # Calling split(args, kwargs) (line 38)
        split_call_result_7305 = invoke(stypy.reporting.localization.Localization(__file__, 38, 30), split_7302, *[str_7303], **kwargs_7304)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___7306 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 30), split_call_result_7305, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_7307 = invoke(stypy.reporting.localization.Localization(__file__, 38, 30), getitem___7306, int_7300)
        
        # Applying the binary operator '+' (line 38)
        result_add_7308 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 28), '+', result_add_7299, subscript_call_result_7307)
        
        str_7309 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 59), 'str', '/')
        # Applying the binary operator '+' (line 38)
        result_add_7310 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 57), '+', result_add_7308, str_7309)
        
        
        # Obtaining the type of the subscript
        int_7311 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 32), 'int')
        
        # Call to split(...): (line 38)
        # Processing the call arguments (line 38)
        str_7314 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 38, 83), 'str', '.')
        # Processing the call keyword arguments (line 38)
        kwargs_7315 = {}
        # Getting the type of 'entity_name' (line 38)
        entity_name_7312 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 65), 'entity_name', False)
        # Obtaining the member 'split' of a type (line 38)
        split_7313 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 65), entity_name_7312, 'split')
        # Calling split(args, kwargs) (line 38)
        split_call_result_7316 = invoke(stypy.reporting.localization.Localization(__file__, 38, 65), split_7313, *[str_7314], **kwargs_7315)
        
        # Obtaining the member '__getitem__' of a type (line 38)
        getitem___7317 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 38, 65), split_call_result_7316, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 38)
        subscript_call_result_7318 = invoke(stypy.reporting.localization.Localization(__file__, 38, 65), getitem___7317, int_7311)
        
        # Applying the binary operator '+' (line 38)
        result_add_7319 = python_operator(stypy.reporting.localization.Localization(__file__, 38, 63), '+', result_add_7310, subscript_call_result_7318)
        
        # Getting the type of 'stypy_parameters' (line 39)
        stypy_parameters_7320 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 39, 38), 'stypy_parameters')
        # Obtaining the member 'type_modifier_file_postfix' of a type (line 39)
        type_modifier_file_postfix_7321 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 39, 38), stypy_parameters_7320, 'type_modifier_file_postfix')
        # Applying the binary operator '+' (line 39)
        result_add_7322 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 36), '+', result_add_7319, type_modifier_file_postfix_7321)
        
        str_7323 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 39, 84), 'str', '.py')
        # Applying the binary operator '+' (line 39)
        result_add_7324 = python_operator(stypy.reporting.localization.Localization(__file__, 39, 82), '+', result_add_7322, str_7323)
        
        # Assigning a type to the variable 'own_modifier_file' (line 37)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 37, 8), 'own_modifier_file', result_add_7324)
        
        # Obtaining an instance of the builtin type 'tuple' (line 41)
        tuple_7325 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 41, 15), 'tuple')
        # Adding type elements to the builtin type 'tuple' instance (line 41)
        # Adding element type (line 41)
        # Getting the type of 'parent_modifier_file' (line 41)
        parent_modifier_file_7326 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 15), 'parent_modifier_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_7325, parent_modifier_file_7326)
        # Adding element type (line 41)
        # Getting the type of 'own_modifier_file' (line 41)
        own_modifier_file_7327 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 41, 37), 'own_modifier_file')
        add_contained_elements_type(stypy.reporting.localization.Localization(__file__, 41, 15), tuple_7325, own_modifier_file_7327)
        
        # Assigning a type to the variable 'stypy_return_type' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 8), 'stypy_return_type', tuple_7325)
        
        # ################# End of '__modifier_files(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__modifier_files' in the type store
        # Getting the type of 'stypy_return_type' (line 25)
        stypy_return_type_7328 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7328)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__modifier_files'
        return stypy_return_type_7328


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

        str_7329 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 56, (-1)), 'str', '\n        This method determines if this type modifier is able to respond to a call to callable_entity. The modifier\n        respond to any callable code that has a modifier file associated. This method search the modifier file and,\n        if found, loads and caches it for performance reasons. Cache also allows us to not to look for the same file on\n        the hard disk over and over, saving much time. callable_entity modifier files have priority over the rule files\n        of their parent entity should both exist.\n\n        Code of this method is mostly identical to the code that searches for rule files on type_rule_call_handler\n\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param callable_entity: Callable entity\n        :return: bool\n        ')
        
        # Call to isclass(...): (line 58)
        # Processing the call arguments (line 58)
        # Getting the type of 'callable_entity' (line 58)
        callable_entity_7332 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 27), 'callable_entity', False)
        # Processing the call keyword arguments (line 58)
        kwargs_7333 = {}
        # Getting the type of 'inspect' (line 58)
        inspect_7330 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 58, 11), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 58)
        isclass_7331 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 58, 11), inspect_7330, 'isclass')
        # Calling isclass(args, kwargs) (line 58)
        isclass_call_result_7334 = invoke(stypy.reporting.localization.Localization(__file__, 58, 11), isclass_7331, *[callable_entity_7332], **kwargs_7333)
        
        # Testing if the type of an if condition is none (line 58)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 58, 8), isclass_call_result_7334):
            
            # Assigning a Attribute to a Name (line 61):
            
            # Assigning a Attribute to a Name (line 61):
            # Getting the type of 'proxy_obj' (line 61)
            proxy_obj_7340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 61)
            name_7341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), proxy_obj_7340, 'name')
            # Assigning a type to the variable 'cache_name' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'cache_name', name_7341)
        else:
            
            # Testing the type of an if condition (line 58)
            if_condition_7335 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 58, 8), isclass_call_result_7334)
            # Assigning a type to the variable 'if_condition_7335' (line 58)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 58, 8), 'if_condition_7335', if_condition_7335)
            # SSA begins for if statement (line 58)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 59):
            
            # Assigning a BinOp to a Name (line 59):
            # Getting the type of 'proxy_obj' (line 59)
            proxy_obj_7336 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 59, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 59)
            name_7337 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 59, 25), proxy_obj_7336, 'name')
            str_7338 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 59, 42), 'str', '.__init__')
            # Applying the binary operator '+' (line 59)
            result_add_7339 = python_operator(stypy.reporting.localization.Localization(__file__, 59, 25), '+', name_7337, str_7338)
            
            # Assigning a type to the variable 'cache_name' (line 59)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 59, 12), 'cache_name', result_add_7339)
            # SSA branch for the else part of an if statement (line 58)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 61):
            
            # Assigning a Attribute to a Name (line 61):
            # Getting the type of 'proxy_obj' (line 61)
            proxy_obj_7340 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 61, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 61)
            name_7341 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 61, 25), proxy_obj_7340, 'name')
            # Assigning a type to the variable 'cache_name' (line 61)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 61, 12), 'cache_name', name_7341)
            # SSA join for if statement (line 58)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to get(...): (line 64)
        # Processing the call arguments (line 64)
        # Getting the type of 'cache_name' (line 64)
        cache_name_7345 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 48), 'cache_name', False)
        # Getting the type of 'False' (line 64)
        False_7346 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 60), 'False', False)
        # Processing the call keyword arguments (line 64)
        kwargs_7347 = {}
        # Getting the type of 'self' (line 64)
        self_7342 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 64, 11), 'self', False)
        # Obtaining the member 'unavailable_modifiers_cache' of a type (line 64)
        unavailable_modifiers_cache_7343 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), self_7342, 'unavailable_modifiers_cache')
        # Obtaining the member 'get' of a type (line 64)
        get_7344 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 64, 11), unavailable_modifiers_cache_7343, 'get')
        # Calling get(args, kwargs) (line 64)
        get_call_result_7348 = invoke(stypy.reporting.localization.Localization(__file__, 64, 11), get_7344, *[cache_name_7345, False_7346], **kwargs_7347)
        
        # Testing if the type of an if condition is none (line 64)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 64, 8), get_call_result_7348):
            pass
        else:
            
            # Testing the type of an if condition (line 64)
            if_condition_7349 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 64, 8), get_call_result_7348)
            # Assigning a type to the variable 'if_condition_7349' (line 64)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 64, 8), 'if_condition_7349', if_condition_7349)
            # SSA begins for if statement (line 64)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'False' (line 65)
            False_7350 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 65, 19), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 65)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 65, 12), 'stypy_return_type', False_7350)
            # SSA join for if statement (line 64)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Call to get(...): (line 68)
        # Processing the call arguments (line 68)
        # Getting the type of 'cache_name' (line 68)
        cache_name_7354 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 36), 'cache_name', False)
        # Getting the type of 'False' (line 68)
        False_7355 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 48), 'False', False)
        # Processing the call keyword arguments (line 68)
        kwargs_7356 = {}
        # Getting the type of 'self' (line 68)
        self_7351 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 68, 11), 'self', False)
        # Obtaining the member 'modifiers_cache' of a type (line 68)
        modifiers_cache_7352 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), self_7351, 'modifiers_cache')
        # Obtaining the member 'get' of a type (line 68)
        get_7353 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 68, 11), modifiers_cache_7352, 'get')
        # Calling get(args, kwargs) (line 68)
        get_call_result_7357 = invoke(stypy.reporting.localization.Localization(__file__, 68, 11), get_7353, *[cache_name_7354, False_7355], **kwargs_7356)
        
        # Testing if the type of an if condition is none (line 68)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 68, 8), get_call_result_7357):
            pass
        else:
            
            # Testing the type of an if condition (line 68)
            if_condition_7358 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 68, 8), get_call_result_7357)
            # Assigning a type to the variable 'if_condition_7358' (line 68)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 68, 8), 'if_condition_7358', if_condition_7358)
            # SSA begins for if statement (line 68)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            # Getting the type of 'True' (line 69)
            True_7359 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 69, 19), 'True')
            # Assigning a type to the variable 'stypy_return_type' (line 69)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 69, 12), 'stypy_return_type', True_7359)
            # SSA join for if statement (line 68)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Getting the type of 'proxy_obj' (line 72)
        proxy_obj_7360 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 11), 'proxy_obj')
        # Obtaining the member 'parent_proxy' of a type (line 72)
        parent_proxy_7361 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 72, 11), proxy_obj_7360, 'parent_proxy')
        # Getting the type of 'None' (line 72)
        None_7362 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 72, 41), 'None')
        # Applying the binary operator 'isnot' (line 72)
        result_is_not_7363 = python_operator(stypy.reporting.localization.Localization(__file__, 72, 11), 'isnot', parent_proxy_7361, None_7362)
        
        # Testing if the type of an if condition is none (line 72)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 72, 8), result_is_not_7363):
            pass
        else:
            
            # Testing the type of an if condition (line 72)
            if_condition_7364 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 72, 8), result_is_not_7363)
            # Assigning a type to the variable 'if_condition_7364' (line 72)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 72, 8), 'if_condition_7364', if_condition_7364)
            # SSA begins for if statement (line 72)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Call to get(...): (line 73)
            # Processing the call arguments (line 73)
            # Getting the type of 'proxy_obj' (line 73)
            proxy_obj_7368 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 40), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 73)
            parent_proxy_7369 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 40), proxy_obj_7368, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 73)
            name_7370 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 40), parent_proxy_7369, 'name')
            # Getting the type of 'False' (line 73)
            False_7371 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 69), 'False', False)
            # Processing the call keyword arguments (line 73)
            kwargs_7372 = {}
            # Getting the type of 'self' (line 73)
            self_7365 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 73, 15), 'self', False)
            # Obtaining the member 'modifiers_cache' of a type (line 73)
            modifiers_cache_7366 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), self_7365, 'modifiers_cache')
            # Obtaining the member 'get' of a type (line 73)
            get_7367 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 73, 15), modifiers_cache_7366, 'get')
            # Calling get(args, kwargs) (line 73)
            get_call_result_7373 = invoke(stypy.reporting.localization.Localization(__file__, 73, 15), get_7367, *[name_7370, False_7371], **kwargs_7372)
            
            # Testing if the type of an if condition is none (line 73)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 73, 12), get_call_result_7373):
                pass
            else:
                
                # Testing the type of an if condition (line 73)
                if_condition_7374 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 73, 12), get_call_result_7373)
                # Assigning a type to the variable 'if_condition_7374' (line 73)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 73, 12), 'if_condition_7374', if_condition_7374)
                # SSA begins for if statement (line 73)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                # Getting the type of 'True' (line 74)
                True_7375 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 74, 23), 'True')
                # Assigning a type to the variable 'stypy_return_type' (line 74)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 74, 16), 'stypy_return_type', True_7375)
                # SSA join for if statement (line 73)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 72)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        
        # Call to ismethod(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'callable_entity' (line 77)
        callable_entity_7378 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 28), 'callable_entity', False)
        # Processing the call keyword arguments (line 77)
        kwargs_7379 = {}
        # Getting the type of 'inspect' (line 77)
        inspect_7376 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 11), 'inspect', False)
        # Obtaining the member 'ismethod' of a type (line 77)
        ismethod_7377 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 11), inspect_7376, 'ismethod')
        # Calling ismethod(args, kwargs) (line 77)
        ismethod_call_result_7380 = invoke(stypy.reporting.localization.Localization(__file__, 77, 11), ismethod_7377, *[callable_entity_7378], **kwargs_7379)
        
        
        # Call to ismethoddescriptor(...): (line 77)
        # Processing the call arguments (line 77)
        # Getting the type of 'callable_entity' (line 77)
        callable_entity_7383 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 75), 'callable_entity', False)
        # Processing the call keyword arguments (line 77)
        kwargs_7384 = {}
        # Getting the type of 'inspect' (line 77)
        inspect_7381 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 77, 48), 'inspect', False)
        # Obtaining the member 'ismethoddescriptor' of a type (line 77)
        ismethoddescriptor_7382 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 77, 48), inspect_7381, 'ismethoddescriptor')
        # Calling ismethoddescriptor(args, kwargs) (line 77)
        ismethoddescriptor_call_result_7385 = invoke(stypy.reporting.localization.Localization(__file__, 77, 48), ismethoddescriptor_7382, *[callable_entity_7383], **kwargs_7384)
        
        # Applying the binary operator 'or' (line 77)
        result_or_keyword_7386 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), 'or', ismethod_call_result_7380, ismethoddescriptor_call_result_7385)
        
        # Evaluating a boolean operation
        
        # Call to isbuiltin(...): (line 78)
        # Processing the call arguments (line 78)
        # Getting the type of 'callable_entity' (line 78)
        callable_entity_7389 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 38), 'callable_entity', False)
        # Processing the call keyword arguments (line 78)
        kwargs_7390 = {}
        # Getting the type of 'inspect' (line 78)
        inspect_7387 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 78, 20), 'inspect', False)
        # Obtaining the member 'isbuiltin' of a type (line 78)
        isbuiltin_7388 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 78, 20), inspect_7387, 'isbuiltin')
        # Calling isbuiltin(args, kwargs) (line 78)
        isbuiltin_call_result_7391 = invoke(stypy.reporting.localization.Localization(__file__, 78, 20), isbuiltin_7388, *[callable_entity_7389], **kwargs_7390)
        
        
        # Call to isclass(...): (line 79)
        # Processing the call arguments (line 79)
        
        # Call to get_python_entity(...): (line 79)
        # Processing the call keyword arguments (line 79)
        kwargs_7397 = {}
        # Getting the type of 'proxy_obj' (line 79)
        proxy_obj_7394 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 37), 'proxy_obj', False)
        # Obtaining the member 'parent_proxy' of a type (line 79)
        parent_proxy_7395 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 37), proxy_obj_7394, 'parent_proxy')
        # Obtaining the member 'get_python_entity' of a type (line 79)
        get_python_entity_7396 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 37), parent_proxy_7395, 'get_python_entity')
        # Calling get_python_entity(args, kwargs) (line 79)
        get_python_entity_call_result_7398 = invoke(stypy.reporting.localization.Localization(__file__, 79, 37), get_python_entity_7396, *[], **kwargs_7397)
        
        # Processing the call keyword arguments (line 79)
        kwargs_7399 = {}
        # Getting the type of 'inspect' (line 79)
        inspect_7392 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 79, 21), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 79)
        isclass_7393 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 79, 21), inspect_7392, 'isclass')
        # Calling isclass(args, kwargs) (line 79)
        isclass_call_result_7400 = invoke(stypy.reporting.localization.Localization(__file__, 79, 21), isclass_7393, *[get_python_entity_call_result_7398], **kwargs_7399)
        
        # Applying the binary operator 'and' (line 78)
        result_and_keyword_7401 = python_operator(stypy.reporting.localization.Localization(__file__, 78, 20), 'and', isbuiltin_call_result_7391, isclass_call_result_7400)
        
        # Applying the binary operator 'or' (line 77)
        result_or_keyword_7402 = python_operator(stypy.reporting.localization.Localization(__file__, 77, 11), 'or', result_or_keyword_7386, result_and_keyword_7401)
        
        # Testing if the type of an if condition is none (line 77)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 77, 8), result_or_keyword_7402):
            
            # Assigning a Call to a Tuple (line 95):
            
            # Assigning a Call to a Name:
            
            # Call to __modifier_files(...): (line 95)
            # Processing the call arguments (line 95)
            # Getting the type of 'proxy_obj' (line 95)
            proxy_obj_7465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 78), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 95)
            parent_proxy_7466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 78), proxy_obj_7465, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 95)
            name_7467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 78), parent_proxy_7466, 'name')
            # Getting the type of 'proxy_obj' (line 96)
            proxy_obj_7468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 78), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 96)
            name_7469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 78), proxy_obj_7468, 'name')
            # Processing the call keyword arguments (line 95)
            kwargs_7470 = {}
            # Getting the type of 'self' (line 95)
            self_7463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 56), 'self', False)
            # Obtaining the member '__modifier_files' of a type (line 95)
            modifier_files_7464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 56), self_7463, '__modifier_files')
            # Calling __modifier_files(args, kwargs) (line 95)
            modifier_files_call_result_7471 = invoke(stypy.reporting.localization.Localization(__file__, 95, 56), modifier_files_7464, *[name_7467, name_7469], **kwargs_7470)
            
            # Assigning a type to the variable 'call_assignment_7265' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7265', modifier_files_call_result_7471)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7265' (line 95)
            call_assignment_7265_7472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7265', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7473 = stypy_get_value_from_tuple(call_assignment_7265_7472, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_7266' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7266', stypy_get_value_from_tuple_call_result_7473)
            
            # Assigning a Name to a Name (line 95):
            # Getting the type of 'call_assignment_7266' (line 95)
            call_assignment_7266_7474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7266')
            # Assigning a type to the variable 'parent_type_rule_file' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'parent_type_rule_file', call_assignment_7266_7474)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7265' (line 95)
            call_assignment_7265_7475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7265', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7476 = stypy_get_value_from_tuple(call_assignment_7265_7475, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_7267' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7267', stypy_get_value_from_tuple_call_result_7476)
            
            # Assigning a Name to a Name (line 95):
            # Getting the type of 'call_assignment_7267' (line 95)
            call_assignment_7267_7477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7267')
            # Assigning a type to the variable 'own_type_rule_file' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'own_type_rule_file', call_assignment_7267_7477)
        else:
            
            # Testing the type of an if condition (line 77)
            if_condition_7403 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 77, 8), result_or_keyword_7402)
            # Assigning a type to the variable 'if_condition_7403' (line 77)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 77, 8), 'if_condition_7403', if_condition_7403)
            # SSA begins for if statement (line 77)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            
            # SSA begins for try-except statement (line 80)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Tuple (line 81):
            
            # Assigning a Call to a Name:
            
            # Call to __modifier_files(...): (line 81)
            # Processing the call arguments (line 81)
            # Getting the type of 'callable_entity' (line 82)
            callable_entity_7406 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 82, 20), 'callable_entity', False)
            # Obtaining the member '__objclass__' of a type (line 82)
            objclass___7407 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 20), callable_entity_7406, '__objclass__')
            # Obtaining the member '__module__' of a type (line 82)
            module___7408 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 82, 20), objclass___7407, '__module__')
            # Getting the type of 'callable_entity' (line 83)
            callable_entity_7409 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 83, 20), 'callable_entity', False)
            # Obtaining the member '__objclass__' of a type (line 83)
            objclass___7410 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), callable_entity_7409, '__objclass__')
            # Obtaining the member '__name__' of a type (line 83)
            name___7411 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 83, 20), objclass___7410, '__name__')
            # Processing the call keyword arguments (line 81)
            kwargs_7412 = {}
            # Getting the type of 'self' (line 81)
            self_7404 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 60), 'self', False)
            # Obtaining the member '__modifier_files' of a type (line 81)
            modifier_files_7405 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 81, 60), self_7404, '__modifier_files')
            # Calling __modifier_files(args, kwargs) (line 81)
            modifier_files_call_result_7413 = invoke(stypy.reporting.localization.Localization(__file__, 81, 60), modifier_files_7405, *[module___7408, name___7411], **kwargs_7412)
            
            # Assigning a type to the variable 'call_assignment_7256' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7256', modifier_files_call_result_7413)
            
            # Assigning a Call to a Name (line 81):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7256' (line 81)
            call_assignment_7256_7414 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7256', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7415 = stypy_get_value_from_tuple(call_assignment_7256_7414, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_7257' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7257', stypy_get_value_from_tuple_call_result_7415)
            
            # Assigning a Name to a Name (line 81):
            # Getting the type of 'call_assignment_7257' (line 81)
            call_assignment_7257_7416 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7257')
            # Assigning a type to the variable 'parent_type_rule_file' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'parent_type_rule_file', call_assignment_7257_7416)
            
            # Assigning a Call to a Name (line 81):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7256' (line 81)
            call_assignment_7256_7417 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7256', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7418 = stypy_get_value_from_tuple(call_assignment_7256_7417, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_7258' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7258', stypy_get_value_from_tuple_call_result_7418)
            
            # Assigning a Name to a Name (line 81):
            # Getting the type of 'call_assignment_7258' (line 81)
            call_assignment_7258_7419 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 81, 16), 'call_assignment_7258')
            # Assigning a type to the variable 'own_type_rule_file' (line 81)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 81, 39), 'own_type_rule_file', call_assignment_7258_7419)
            # SSA branch for the except part of a try statement (line 80)
            # SSA branch for the except '<any exception>' branch of a try statement (line 80)
            module_type_store.open_ssa_branch('except')
            
            # Call to ismodule(...): (line 86)
            # Processing the call arguments (line 86)
            
            # Call to get_python_entity(...): (line 86)
            # Processing the call keyword arguments (line 86)
            kwargs_7425 = {}
            # Getting the type of 'proxy_obj' (line 86)
            proxy_obj_7422 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 36), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 86)
            parent_proxy_7423 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 36), proxy_obj_7422, 'parent_proxy')
            # Obtaining the member 'get_python_entity' of a type (line 86)
            get_python_entity_7424 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 36), parent_proxy_7423, 'get_python_entity')
            # Calling get_python_entity(args, kwargs) (line 86)
            get_python_entity_call_result_7426 = invoke(stypy.reporting.localization.Localization(__file__, 86, 36), get_python_entity_7424, *[], **kwargs_7425)
            
            # Processing the call keyword arguments (line 86)
            kwargs_7427 = {}
            # Getting the type of 'inspect' (line 86)
            inspect_7420 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 86, 19), 'inspect', False)
            # Obtaining the member 'ismodule' of a type (line 86)
            ismodule_7421 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 86, 19), inspect_7420, 'ismodule')
            # Calling ismodule(args, kwargs) (line 86)
            ismodule_call_result_7428 = invoke(stypy.reporting.localization.Localization(__file__, 86, 19), ismodule_7421, *[get_python_entity_call_result_7426], **kwargs_7427)
            
            # Testing if the type of an if condition is none (line 86)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 86, 16), ismodule_call_result_7428):
                
                # Assigning a Call to a Tuple (line 91):
                
                # Assigning a Call to a Name:
                
                # Call to __modifier_files(...): (line 91)
                # Processing the call arguments (line 91)
                # Getting the type of 'proxy_obj' (line 92)
                proxy_obj_7448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 92)
                parent_proxy_7449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), proxy_obj_7448, 'parent_proxy')
                # Obtaining the member 'parent_proxy' of a type (line 92)
                parent_proxy_7450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), parent_proxy_7449, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 92)
                name_7451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), parent_proxy_7450, 'name')
                # Getting the type of 'proxy_obj' (line 93)
                proxy_obj_7452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 93)
                parent_proxy_7453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), proxy_obj_7452, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 93)
                name_7454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), parent_proxy_7453, 'name')
                # Processing the call keyword arguments (line 91)
                kwargs_7455 = {}
                # Getting the type of 'self' (line 91)
                self_7446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 64), 'self', False)
                # Obtaining the member '__modifier_files' of a type (line 91)
                modifier_files_7447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 64), self_7446, '__modifier_files')
                # Calling __modifier_files(args, kwargs) (line 91)
                modifier_files_call_result_7456 = invoke(stypy.reporting.localization.Localization(__file__, 91, 64), modifier_files_7447, *[name_7451, name_7454], **kwargs_7455)
                
                # Assigning a type to the variable 'call_assignment_7262' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7262', modifier_files_call_result_7456)
                
                # Assigning a Call to a Name (line 91):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7262' (line 91)
                call_assignment_7262_7457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7262', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7458 = stypy_get_value_from_tuple(call_assignment_7262_7457, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_7263' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7263', stypy_get_value_from_tuple_call_result_7458)
                
                # Assigning a Name to a Name (line 91):
                # Getting the type of 'call_assignment_7263' (line 91)
                call_assignment_7263_7459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7263')
                # Assigning a type to the variable 'parent_type_rule_file' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'parent_type_rule_file', call_assignment_7263_7459)
                
                # Assigning a Call to a Name (line 91):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7262' (line 91)
                call_assignment_7262_7460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7262', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7461 = stypy_get_value_from_tuple(call_assignment_7262_7460, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_7264' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7264', stypy_get_value_from_tuple_call_result_7461)
                
                # Assigning a Name to a Name (line 91):
                # Getting the type of 'call_assignment_7264' (line 91)
                call_assignment_7264_7462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7264')
                # Assigning a type to the variable 'own_type_rule_file' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'own_type_rule_file', call_assignment_7264_7462)
            else:
                
                # Testing the type of an if condition (line 86)
                if_condition_7429 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 86, 16), ismodule_call_result_7428)
                # Assigning a type to the variable 'if_condition_7429' (line 86)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 86, 16), 'if_condition_7429', if_condition_7429)
                # SSA begins for if statement (line 86)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Call to a Tuple (line 87):
                
                # Assigning a Call to a Name:
                
                # Call to __modifier_files(...): (line 87)
                # Processing the call arguments (line 87)
                # Getting the type of 'proxy_obj' (line 88)
                proxy_obj_7432 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 88, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 88)
                parent_proxy_7433 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), proxy_obj_7432, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 88)
                name_7434 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 88, 24), parent_proxy_7433, 'name')
                # Getting the type of 'proxy_obj' (line 89)
                proxy_obj_7435 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 89, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 89)
                parent_proxy_7436 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), proxy_obj_7435, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 89)
                name_7437 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 89, 24), parent_proxy_7436, 'name')
                # Processing the call keyword arguments (line 87)
                kwargs_7438 = {}
                # Getting the type of 'self' (line 87)
                self_7430 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 64), 'self', False)
                # Obtaining the member '__modifier_files' of a type (line 87)
                modifier_files_7431 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 87, 64), self_7430, '__modifier_files')
                # Calling __modifier_files(args, kwargs) (line 87)
                modifier_files_call_result_7439 = invoke(stypy.reporting.localization.Localization(__file__, 87, 64), modifier_files_7431, *[name_7434, name_7437], **kwargs_7438)
                
                # Assigning a type to the variable 'call_assignment_7259' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7259', modifier_files_call_result_7439)
                
                # Assigning a Call to a Name (line 87):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7259' (line 87)
                call_assignment_7259_7440 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7259', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7441 = stypy_get_value_from_tuple(call_assignment_7259_7440, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_7260' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7260', stypy_get_value_from_tuple_call_result_7441)
                
                # Assigning a Name to a Name (line 87):
                # Getting the type of 'call_assignment_7260' (line 87)
                call_assignment_7260_7442 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7260')
                # Assigning a type to the variable 'parent_type_rule_file' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'parent_type_rule_file', call_assignment_7260_7442)
                
                # Assigning a Call to a Name (line 87):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7259' (line 87)
                call_assignment_7259_7443 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7259', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7444 = stypy_get_value_from_tuple(call_assignment_7259_7443, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_7261' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7261', stypy_get_value_from_tuple_call_result_7444)
                
                # Assigning a Name to a Name (line 87):
                # Getting the type of 'call_assignment_7261' (line 87)
                call_assignment_7261_7445 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 87, 20), 'call_assignment_7261')
                # Assigning a type to the variable 'own_type_rule_file' (line 87)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 87, 43), 'own_type_rule_file', call_assignment_7261_7445)
                # SSA branch for the else part of an if statement (line 86)
                module_type_store.open_ssa_branch('else')
                
                # Assigning a Call to a Tuple (line 91):
                
                # Assigning a Call to a Name:
                
                # Call to __modifier_files(...): (line 91)
                # Processing the call arguments (line 91)
                # Getting the type of 'proxy_obj' (line 92)
                proxy_obj_7448 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 92, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 92)
                parent_proxy_7449 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), proxy_obj_7448, 'parent_proxy')
                # Obtaining the member 'parent_proxy' of a type (line 92)
                parent_proxy_7450 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), parent_proxy_7449, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 92)
                name_7451 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 92, 24), parent_proxy_7450, 'name')
                # Getting the type of 'proxy_obj' (line 93)
                proxy_obj_7452 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 93, 24), 'proxy_obj', False)
                # Obtaining the member 'parent_proxy' of a type (line 93)
                parent_proxy_7453 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), proxy_obj_7452, 'parent_proxy')
                # Obtaining the member 'name' of a type (line 93)
                name_7454 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 93, 24), parent_proxy_7453, 'name')
                # Processing the call keyword arguments (line 91)
                kwargs_7455 = {}
                # Getting the type of 'self' (line 91)
                self_7446 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 64), 'self', False)
                # Obtaining the member '__modifier_files' of a type (line 91)
                modifier_files_7447 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 91, 64), self_7446, '__modifier_files')
                # Calling __modifier_files(args, kwargs) (line 91)
                modifier_files_call_result_7456 = invoke(stypy.reporting.localization.Localization(__file__, 91, 64), modifier_files_7447, *[name_7451, name_7454], **kwargs_7455)
                
                # Assigning a type to the variable 'call_assignment_7262' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7262', modifier_files_call_result_7456)
                
                # Assigning a Call to a Name (line 91):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7262' (line 91)
                call_assignment_7262_7457 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7262', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7458 = stypy_get_value_from_tuple(call_assignment_7262_7457, 2, 0)
                
                # Assigning a type to the variable 'call_assignment_7263' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7263', stypy_get_value_from_tuple_call_result_7458)
                
                # Assigning a Name to a Name (line 91):
                # Getting the type of 'call_assignment_7263' (line 91)
                call_assignment_7263_7459 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7263')
                # Assigning a type to the variable 'parent_type_rule_file' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'parent_type_rule_file', call_assignment_7263_7459)
                
                # Assigning a Call to a Name (line 91):
                
                # Call to stypy_get_value_from_tuple(...):
                # Processing the call arguments
                # Getting the type of 'call_assignment_7262' (line 91)
                call_assignment_7262_7460 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7262', False)
                # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
                stypy_get_value_from_tuple_call_result_7461 = stypy_get_value_from_tuple(call_assignment_7262_7460, 2, 1)
                
                # Assigning a type to the variable 'call_assignment_7264' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7264', stypy_get_value_from_tuple_call_result_7461)
                
                # Assigning a Name to a Name (line 91):
                # Getting the type of 'call_assignment_7264' (line 91)
                call_assignment_7264_7462 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 91, 20), 'call_assignment_7264')
                # Assigning a type to the variable 'own_type_rule_file' (line 91)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 91, 43), 'own_type_rule_file', call_assignment_7264_7462)
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
            proxy_obj_7465 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 78), 'proxy_obj', False)
            # Obtaining the member 'parent_proxy' of a type (line 95)
            parent_proxy_7466 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 78), proxy_obj_7465, 'parent_proxy')
            # Obtaining the member 'name' of a type (line 95)
            name_7467 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 78), parent_proxy_7466, 'name')
            # Getting the type of 'proxy_obj' (line 96)
            proxy_obj_7468 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 96, 78), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 96)
            name_7469 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 96, 78), proxy_obj_7468, 'name')
            # Processing the call keyword arguments (line 95)
            kwargs_7470 = {}
            # Getting the type of 'self' (line 95)
            self_7463 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 56), 'self', False)
            # Obtaining the member '__modifier_files' of a type (line 95)
            modifier_files_7464 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 95, 56), self_7463, '__modifier_files')
            # Calling __modifier_files(args, kwargs) (line 95)
            modifier_files_call_result_7471 = invoke(stypy.reporting.localization.Localization(__file__, 95, 56), modifier_files_7464, *[name_7467, name_7469], **kwargs_7470)
            
            # Assigning a type to the variable 'call_assignment_7265' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7265', modifier_files_call_result_7471)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7265' (line 95)
            call_assignment_7265_7472 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7265', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7473 = stypy_get_value_from_tuple(call_assignment_7265_7472, 2, 0)
            
            # Assigning a type to the variable 'call_assignment_7266' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7266', stypy_get_value_from_tuple_call_result_7473)
            
            # Assigning a Name to a Name (line 95):
            # Getting the type of 'call_assignment_7266' (line 95)
            call_assignment_7266_7474 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7266')
            # Assigning a type to the variable 'parent_type_rule_file' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'parent_type_rule_file', call_assignment_7266_7474)
            
            # Assigning a Call to a Name (line 95):
            
            # Call to stypy_get_value_from_tuple(...):
            # Processing the call arguments
            # Getting the type of 'call_assignment_7265' (line 95)
            call_assignment_7265_7475 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7265', False)
            # Calling stypy_get_value_from_tuple(tuple, tuple length, tuple pos)
            stypy_get_value_from_tuple_call_result_7476 = stypy_get_value_from_tuple(call_assignment_7265_7475, 2, 1)
            
            # Assigning a type to the variable 'call_assignment_7267' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7267', stypy_get_value_from_tuple_call_result_7476)
            
            # Assigning a Name to a Name (line 95):
            # Getting the type of 'call_assignment_7267' (line 95)
            call_assignment_7267_7477 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 95, 12), 'call_assignment_7267')
            # Assigning a type to the variable 'own_type_rule_file' (line 95)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 95, 35), 'own_type_rule_file', call_assignment_7267_7477)
            # SSA join for if statement (line 77)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 99):
        
        # Assigning a Call to a Name (line 99):
        
        # Call to isfile(...): (line 99)
        # Processing the call arguments (line 99)
        # Getting the type of 'parent_type_rule_file' (line 99)
        parent_type_rule_file_7481 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 38), 'parent_type_rule_file', False)
        # Processing the call keyword arguments (line 99)
        kwargs_7482 = {}
        # Getting the type of 'os' (line 99)
        os_7478 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 23), 'os', False)
        # Obtaining the member 'path' of a type (line 99)
        path_7479 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 23), os_7478, 'path')
        # Obtaining the member 'isfile' of a type (line 99)
        isfile_7480 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 23), path_7479, 'isfile')
        # Calling isfile(args, kwargs) (line 99)
        isfile_call_result_7483 = invoke(stypy.reporting.localization.Localization(__file__, 99, 23), isfile_7480, *[parent_type_rule_file_7481], **kwargs_7482)
        
        # Assigning a type to the variable 'parent_exist' (line 99)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 8), 'parent_exist', isfile_call_result_7483)
        
        # Assigning a Call to a Name (line 100):
        
        # Assigning a Call to a Name (line 100):
        
        # Call to isfile(...): (line 100)
        # Processing the call arguments (line 100)
        # Getting the type of 'own_type_rule_file' (line 100)
        own_type_rule_file_7487 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 35), 'own_type_rule_file', False)
        # Processing the call keyword arguments (line 100)
        kwargs_7488 = {}
        # Getting the type of 'os' (line 100)
        os_7484 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 20), 'os', False)
        # Obtaining the member 'path' of a type (line 100)
        path_7485 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), os_7484, 'path')
        # Obtaining the member 'isfile' of a type (line 100)
        isfile_7486 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 20), path_7485, 'isfile')
        # Calling isfile(args, kwargs) (line 100)
        isfile_call_result_7489 = invoke(stypy.reporting.localization.Localization(__file__, 100, 20), isfile_7486, *[own_type_rule_file_7487], **kwargs_7488)
        
        # Assigning a type to the variable 'own_exist' (line 100)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 8), 'own_exist', isfile_call_result_7489)
        
        # Assigning a Str to a Name (line 101):
        
        # Assigning a Str to a Name (line 101):
        str_7490 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 20), 'str', '')
        # Assigning a type to the variable 'file_path' (line 101)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 8), 'file_path', str_7490)
        # Getting the type of 'parent_exist' (line 103)
        parent_exist_7491 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 11), 'parent_exist')
        # Testing if the type of an if condition is none (line 103)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 103, 8), parent_exist_7491):
            pass
        else:
            
            # Testing the type of an if condition (line 103)
            if_condition_7492 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 103, 8), parent_exist_7491)
            # Assigning a type to the variable 'if_condition_7492' (line 103)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 8), 'if_condition_7492', if_condition_7492)
            # SSA begins for if statement (line 103)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 104):
            
            # Assigning a Name to a Name (line 104):
            # Getting the type of 'parent_type_rule_file' (line 104)
            parent_type_rule_file_7493 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 24), 'parent_type_rule_file')
            # Assigning a type to the variable 'file_path' (line 104)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 12), 'file_path', parent_type_rule_file_7493)
            # SSA join for if statement (line 103)
            module_type_store = module_type_store.join_ssa_context()
            

        # Getting the type of 'own_exist' (line 106)
        own_exist_7494 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 106, 11), 'own_exist')
        # Testing if the type of an if condition is none (line 106)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 106, 8), own_exist_7494):
            pass
        else:
            
            # Testing the type of an if condition (line 106)
            if_condition_7495 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 106, 8), own_exist_7494)
            # Assigning a type to the variable 'if_condition_7495' (line 106)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 106, 8), 'if_condition_7495', if_condition_7495)
            # SSA begins for if statement (line 106)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Name to a Name (line 107):
            
            # Assigning a Name to a Name (line 107):
            # Getting the type of 'own_type_rule_file' (line 107)
            own_type_rule_file_7496 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 107, 24), 'own_type_rule_file')
            # Assigning a type to the variable 'file_path' (line 107)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 107, 12), 'file_path', own_type_rule_file_7496)
            # SSA join for if statement (line 106)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 110)
        parent_exist_7497 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 11), 'parent_exist')
        # Getting the type of 'own_exist' (line 110)
        own_exist_7498 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 110, 27), 'own_exist')
        # Applying the binary operator 'or' (line 110)
        result_or_keyword_7499 = python_operator(stypy.reporting.localization.Localization(__file__, 110, 11), 'or', parent_exist_7497, own_exist_7498)
        
        # Testing if the type of an if condition is none (line 110)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 110, 8), result_or_keyword_7499):
            pass
        else:
            
            # Testing the type of an if condition (line 110)
            if_condition_7500 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 110, 8), result_or_keyword_7499)
            # Assigning a type to the variable 'if_condition_7500' (line 110)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 110, 8), 'if_condition_7500', if_condition_7500)
            # SSA begins for if statement (line 110)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a Call to a Name (line 111):
            
            # Assigning a Call to a Name (line 111):
            
            # Call to dirname(...): (line 111)
            # Processing the call arguments (line 111)
            # Getting the type of 'file_path' (line 111)
            file_path_7504 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 38), 'file_path', False)
            # Processing the call keyword arguments (line 111)
            kwargs_7505 = {}
            # Getting the type of 'os' (line 111)
            os_7501 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 111, 22), 'os', False)
            # Obtaining the member 'path' of a type (line 111)
            path_7502 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 22), os_7501, 'path')
            # Obtaining the member 'dirname' of a type (line 111)
            dirname_7503 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 111, 22), path_7502, 'dirname')
            # Calling dirname(args, kwargs) (line 111)
            dirname_call_result_7506 = invoke(stypy.reporting.localization.Localization(__file__, 111, 22), dirname_7503, *[file_path_7504], **kwargs_7505)
            
            # Assigning a type to the variable 'dirname' (line 111)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 111, 12), 'dirname', dirname_call_result_7506)
            
            # Assigning a Subscript to a Name (line 112):
            
            # Assigning a Subscript to a Name (line 112):
            
            # Obtaining the type of the subscript
            int_7507 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 45), 'int')
            int_7508 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 47), 'int')
            slice_7509 = ensure_slice_bounds(stypy.reporting.localization.Localization(__file__, 112, 20), int_7507, int_7508, None)
            
            # Obtaining the type of the subscript
            int_7510 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 41), 'int')
            
            # Call to split(...): (line 112)
            # Processing the call arguments (line 112)
            str_7513 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 112, 36), 'str', '/')
            # Processing the call keyword arguments (line 112)
            kwargs_7514 = {}
            # Getting the type of 'file_path' (line 112)
            file_path_7511 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 112, 20), 'file_path', False)
            # Obtaining the member 'split' of a type (line 112)
            split_7512 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), file_path_7511, 'split')
            # Calling split(args, kwargs) (line 112)
            split_call_result_7515 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), split_7512, *[str_7513], **kwargs_7514)
            
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___7516 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), split_call_result_7515, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_7517 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), getitem___7516, int_7510)
            
            # Obtaining the member '__getitem__' of a type (line 112)
            getitem___7518 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 112, 20), subscript_call_result_7517, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 112)
            subscript_call_result_7519 = invoke(stypy.reporting.localization.Localization(__file__, 112, 20), getitem___7518, slice_7509)
            
            # Assigning a type to the variable 'file_' (line 112)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 112, 12), 'file_', subscript_call_result_7519)
            
            # Call to append(...): (line 114)
            # Processing the call arguments (line 114)
            # Getting the type of 'dirname' (line 114)
            dirname_7523 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 28), 'dirname', False)
            # Processing the call keyword arguments (line 114)
            kwargs_7524 = {}
            # Getting the type of 'sys' (line 114)
            sys_7520 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 114, 12), 'sys', False)
            # Obtaining the member 'path' of a type (line 114)
            path_7521 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), sys_7520, 'path')
            # Obtaining the member 'append' of a type (line 114)
            append_7522 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 114, 12), path_7521, 'append')
            # Calling append(args, kwargs) (line 114)
            append_call_result_7525 = invoke(stypy.reporting.localization.Localization(__file__, 114, 12), append_7522, *[dirname_7523], **kwargs_7524)
            
            
            # Assigning a Call to a Name (line 115):
            
            # Assigning a Call to a Name (line 115):
            
            # Call to __import__(...): (line 115)
            # Processing the call arguments (line 115)
            # Getting the type of 'file_' (line 115)
            file__7527 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 32), 'file_', False)
            
            # Call to globals(...): (line 115)
            # Processing the call keyword arguments (line 115)
            kwargs_7529 = {}
            # Getting the type of 'globals' (line 115)
            globals_7528 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 39), 'globals', False)
            # Calling globals(args, kwargs) (line 115)
            globals_call_result_7530 = invoke(stypy.reporting.localization.Localization(__file__, 115, 39), globals_7528, *[], **kwargs_7529)
            
            
            # Call to locals(...): (line 115)
            # Processing the call keyword arguments (line 115)
            kwargs_7532 = {}
            # Getting the type of 'locals' (line 115)
            locals_7531 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 50), 'locals', False)
            # Calling locals(args, kwargs) (line 115)
            locals_call_result_7533 = invoke(stypy.reporting.localization.Localization(__file__, 115, 50), locals_7531, *[], **kwargs_7532)
            
            # Processing the call keyword arguments (line 115)
            kwargs_7534 = {}
            # Getting the type of '__import__' (line 115)
            import___7526 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 115, 21), '__import__', False)
            # Calling __import__(args, kwargs) (line 115)
            import___call_result_7535 = invoke(stypy.reporting.localization.Localization(__file__, 115, 21), import___7526, *[file__7527, globals_call_result_7530, locals_call_result_7533], **kwargs_7534)
            
            # Assigning a type to the variable 'module' (line 115)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 115, 12), 'module', import___call_result_7535)
            
            # Assigning a Subscript to a Name (line 116):
            
            # Assigning a Subscript to a Name (line 116):
            
            # Obtaining the type of the subscript
            int_7536 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 52), 'int')
            
            # Call to split(...): (line 116)
            # Processing the call arguments (line 116)
            str_7540 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 116, 47), 'str', '.')
            # Processing the call keyword arguments (line 116)
            kwargs_7541 = {}
            # Getting the type of 'proxy_obj' (line 116)
            proxy_obj_7537 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 116, 26), 'proxy_obj', False)
            # Obtaining the member 'name' of a type (line 116)
            name_7538 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), proxy_obj_7537, 'name')
            # Obtaining the member 'split' of a type (line 116)
            split_7539 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), name_7538, 'split')
            # Calling split(args, kwargs) (line 116)
            split_call_result_7542 = invoke(stypy.reporting.localization.Localization(__file__, 116, 26), split_7539, *[str_7540], **kwargs_7541)
            
            # Obtaining the member '__getitem__' of a type (line 116)
            getitem___7543 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 116, 26), split_call_result_7542, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 116)
            subscript_call_result_7544 = invoke(stypy.reporting.localization.Localization(__file__, 116, 26), getitem___7543, int_7536)
            
            # Assigning a type to the variable 'entity_name' (line 116)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 116, 12), 'entity_name', subscript_call_result_7544)
            
            
            # SSA begins for try-except statement (line 117)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'try-except')
            
            # Assigning a Call to a Name (line 119):
            
            # Assigning a Call to a Name (line 119):
            
            # Call to getattr(...): (line 119)
            # Processing the call arguments (line 119)
            # Getting the type of 'module' (line 119)
            module_7546 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 33), 'module', False)
            # Obtaining the member 'TypeModifiers' of a type (line 119)
            TypeModifiers_7547 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 119, 33), module_7546, 'TypeModifiers')
            # Getting the type of 'entity_name' (line 119)
            entity_name_7548 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 55), 'entity_name', False)
            # Processing the call keyword arguments (line 119)
            kwargs_7549 = {}
            # Getting the type of 'getattr' (line 119)
            getattr_7545 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 119, 25), 'getattr', False)
            # Calling getattr(args, kwargs) (line 119)
            getattr_call_result_7550 = invoke(stypy.reporting.localization.Localization(__file__, 119, 25), getattr_7545, *[TypeModifiers_7547, entity_name_7548], **kwargs_7549)
            
            # Assigning a type to the variable 'method' (line 119)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 119, 16), 'method', getattr_call_result_7550)
            
            # Assigning a Name to a Subscript (line 120):
            
            # Assigning a Name to a Subscript (line 120):
            # Getting the type of 'method' (line 120)
            method_7551 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 51), 'method')
            # Getting the type of 'self' (line 120)
            self_7552 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 16), 'self')
            # Obtaining the member 'modifiers_cache' of a type (line 120)
            modifiers_cache_7553 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 120, 16), self_7552, 'modifiers_cache')
            # Getting the type of 'cache_name' (line 120)
            cache_name_7554 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 120, 37), 'cache_name')
            # Storing an element on a container (line 120)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 120, 16), modifiers_cache_7553, (cache_name_7554, method_7551))
            # SSA branch for the except part of a try statement (line 117)
            # SSA branch for the except '<any exception>' branch of a try statement (line 117)
            module_type_store.open_ssa_branch('except')
            
            # Assigning a Name to a Subscript (line 123):
            
            # Assigning a Name to a Subscript (line 123):
            # Getting the type of 'True' (line 123)
            True_7555 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 63), 'True')
            # Getting the type of 'self' (line 123)
            self_7556 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 16), 'self')
            # Obtaining the member 'unavailable_modifiers_cache' of a type (line 123)
            unavailable_modifiers_cache_7557 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 123, 16), self_7556, 'unavailable_modifiers_cache')
            # Getting the type of 'cache_name' (line 123)
            cache_name_7558 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 123, 49), 'cache_name')
            # Storing an element on a container (line 123)
            set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 123, 16), unavailable_modifiers_cache_7557, (cache_name_7558, True_7555))
            # Getting the type of 'False' (line 124)
            False_7559 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 124, 23), 'False')
            # Assigning a type to the variable 'stypy_return_type' (line 124)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 124, 16), 'stypy_return_type', False_7559)
            # SSA join for try-except statement (line 117)
            module_type_store = module_type_store.join_ssa_context()
            
            # SSA join for if statement (line 110)
            module_type_store = module_type_store.join_ssa_context()
            

        
        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 126)
        parent_exist_7560 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 16), 'parent_exist')
        # Getting the type of 'own_exist' (line 126)
        own_exist_7561 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 126, 32), 'own_exist')
        # Applying the binary operator 'or' (line 126)
        result_or_keyword_7562 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 16), 'or', parent_exist_7560, own_exist_7561)
        
        # Applying the 'not' unary operator (line 126)
        result_not__7563 = python_operator(stypy.reporting.localization.Localization(__file__, 126, 11), 'not', result_or_keyword_7562)
        
        # Testing if the type of an if condition is none (line 126)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 126, 8), result_not__7563):
            pass
        else:
            
            # Testing the type of an if condition (line 126)
            if_condition_7564 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 126, 8), result_not__7563)
            # Assigning a type to the variable 'if_condition_7564' (line 126)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 126, 8), 'if_condition_7564', if_condition_7564)
            # SSA begins for if statement (line 126)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Getting the type of 'proxy_obj' (line 127)
            proxy_obj_7565 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 15), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 127)
            name_7566 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 15), proxy_obj_7565, 'name')
            # Getting the type of 'self' (line 127)
            self_7567 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 127, 37), 'self')
            # Obtaining the member 'unavailable_modifiers_cache' of a type (line 127)
            unavailable_modifiers_cache_7568 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 127, 37), self_7567, 'unavailable_modifiers_cache')
            # Applying the binary operator 'notin' (line 127)
            result_contains_7569 = python_operator(stypy.reporting.localization.Localization(__file__, 127, 15), 'notin', name_7566, unavailable_modifiers_cache_7568)
            
            # Testing if the type of an if condition is none (line 127)

            if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 127, 12), result_contains_7569):
                pass
            else:
                
                # Testing the type of an if condition (line 127)
                if_condition_7570 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 127, 12), result_contains_7569)
                # Assigning a type to the variable 'if_condition_7570' (line 127)
                module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 127, 12), 'if_condition_7570', if_condition_7570)
                # SSA begins for if statement (line 127)
                module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
                
                # Assigning a Name to a Subscript (line 129):
                
                # Assigning a Name to a Subscript (line 129):
                # Getting the type of 'True' (line 129)
                True_7571 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 63), 'True')
                # Getting the type of 'self' (line 129)
                self_7572 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 16), 'self')
                # Obtaining the member 'unavailable_modifiers_cache' of a type (line 129)
                unavailable_modifiers_cache_7573 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 129, 16), self_7572, 'unavailable_modifiers_cache')
                # Getting the type of 'cache_name' (line 129)
                cache_name_7574 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 129, 49), 'cache_name')
                # Storing an element on a container (line 129)
                set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 129, 16), unavailable_modifiers_cache_7573, (cache_name_7574, True_7571))
                # SSA join for if statement (line 127)
                module_type_store = module_type_store.join_ssa_context()
                

            # SSA join for if statement (line 126)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Evaluating a boolean operation
        # Getting the type of 'parent_exist' (line 131)
        parent_exist_7575 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 15), 'parent_exist')
        # Getting the type of 'own_exist' (line 131)
        own_exist_7576 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 131, 31), 'own_exist')
        # Applying the binary operator 'or' (line 131)
        result_or_keyword_7577 = python_operator(stypy.reporting.localization.Localization(__file__, 131, 15), 'or', parent_exist_7575, own_exist_7576)
        
        # Assigning a type to the variable 'stypy_return_type' (line 131)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 131, 8), 'stypy_return_type', result_or_keyword_7577)
        
        # ################# End of 'applies_to(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'applies_to' in the type store
        # Getting the type of 'stypy_return_type' (line 43)
        stypy_return_type_7578 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 43, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7578)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'applies_to'
        return stypy_return_type_7578


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

        str_7579 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 143, (-1)), 'str', '\n        Calls the type modifier for callable entity to determine its return type.\n\n        :param proxy_obj: TypeInferenceProxy that hold the callable entity\n        :param localization: Caller information\n        :param callable_entity: Callable entity\n        :param arg_types: Arguments\n        :param kwargs_types: Keyword arguments\n        :return: Return type of the call\n        ')
        
        # Call to isclass(...): (line 144)
        # Processing the call arguments (line 144)
        # Getting the type of 'callable_entity' (line 144)
        callable_entity_7582 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 27), 'callable_entity', False)
        # Processing the call keyword arguments (line 144)
        kwargs_7583 = {}
        # Getting the type of 'inspect' (line 144)
        inspect_7580 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 144, 11), 'inspect', False)
        # Obtaining the member 'isclass' of a type (line 144)
        isclass_7581 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 144, 11), inspect_7580, 'isclass')
        # Calling isclass(args, kwargs) (line 144)
        isclass_call_result_7584 = invoke(stypy.reporting.localization.Localization(__file__, 144, 11), isclass_7581, *[callable_entity_7582], **kwargs_7583)
        
        # Testing if the type of an if condition is none (line 144)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 144, 8), isclass_call_result_7584):
            
            # Assigning a Attribute to a Name (line 147):
            
            # Assigning a Attribute to a Name (line 147):
            # Getting the type of 'proxy_obj' (line 147)
            proxy_obj_7590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 147)
            name_7591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), proxy_obj_7590, 'name')
            # Assigning a type to the variable 'cache_name' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'cache_name', name_7591)
        else:
            
            # Testing the type of an if condition (line 144)
            if_condition_7585 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 144, 8), isclass_call_result_7584)
            # Assigning a type to the variable 'if_condition_7585' (line 144)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 144, 8), 'if_condition_7585', if_condition_7585)
            # SSA begins for if statement (line 144)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Assigning a BinOp to a Name (line 145):
            
            # Assigning a BinOp to a Name (line 145):
            # Getting the type of 'proxy_obj' (line 145)
            proxy_obj_7586 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 145, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 145)
            name_7587 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 145, 25), proxy_obj_7586, 'name')
            str_7588 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 145, 42), 'str', '.__init__')
            # Applying the binary operator '+' (line 145)
            result_add_7589 = python_operator(stypy.reporting.localization.Localization(__file__, 145, 25), '+', name_7587, str_7588)
            
            # Assigning a type to the variable 'cache_name' (line 145)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 145, 12), 'cache_name', result_add_7589)
            # SSA branch for the else part of an if statement (line 144)
            module_type_store.open_ssa_branch('else')
            
            # Assigning a Attribute to a Name (line 147):
            
            # Assigning a Attribute to a Name (line 147):
            # Getting the type of 'proxy_obj' (line 147)
            proxy_obj_7590 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 147, 25), 'proxy_obj')
            # Obtaining the member 'name' of a type (line 147)
            name_7591 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 147, 25), proxy_obj_7590, 'name')
            # Assigning a type to the variable 'cache_name' (line 147)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 147, 12), 'cache_name', name_7591)
            # SSA join for if statement (line 144)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Subscript to a Name (line 149):
        
        # Assigning a Subscript to a Name (line 149):
        
        # Obtaining the type of the subscript
        # Getting the type of 'cache_name' (line 149)
        cache_name_7592 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 40), 'cache_name')
        # Getting the type of 'self' (line 149)
        self_7593 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 149, 19), 'self')
        # Obtaining the member 'modifiers_cache' of a type (line 149)
        modifiers_cache_7594 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), self_7593, 'modifiers_cache')
        # Obtaining the member '__getitem__' of a type (line 149)
        getitem___7595 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 149, 19), modifiers_cache_7594, '__getitem__')
        # Calling the subscript (__getitem__) to obtain the elements type (line 149)
        subscript_call_result_7596 = invoke(stypy.reporting.localization.Localization(__file__, 149, 19), getitem___7595, cache_name_7592)
        
        # Assigning a type to the variable 'modifier' (line 149)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 149, 8), 'modifier', subscript_call_result_7596)
        
        # Assigning a Call to a Name (line 152):
        
        # Assigning a Call to a Name (line 152):
        
        # Call to tuple(...): (line 152)
        # Processing the call arguments (line 152)
        
        # Call to list(...): (line 152)
        # Processing the call arguments (line 152)
        # Getting the type of 'arg_types' (line 152)
        arg_types_7599 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 36), 'arg_types', False)
        # Processing the call keyword arguments (line 152)
        kwargs_7600 = {}
        # Getting the type of 'list' (line 152)
        list_7598 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 31), 'list', False)
        # Calling list(args, kwargs) (line 152)
        list_call_result_7601 = invoke(stypy.reporting.localization.Localization(__file__, 152, 31), list_7598, *[arg_types_7599], **kwargs_7600)
        
        
        # Call to values(...): (line 152)
        # Processing the call keyword arguments (line 152)
        kwargs_7604 = {}
        # Getting the type of 'kwargs_types' (line 152)
        kwargs_types_7602 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 49), 'kwargs_types', False)
        # Obtaining the member 'values' of a type (line 152)
        values_7603 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 152, 49), kwargs_types_7602, 'values')
        # Calling values(args, kwargs) (line 152)
        values_call_result_7605 = invoke(stypy.reporting.localization.Localization(__file__, 152, 49), values_7603, *[], **kwargs_7604)
        
        # Applying the binary operator '+' (line 152)
        result_add_7606 = python_operator(stypy.reporting.localization.Localization(__file__, 152, 31), '+', list_call_result_7601, values_call_result_7605)
        
        # Processing the call keyword arguments (line 152)
        kwargs_7607 = {}
        # Getting the type of 'tuple' (line 152)
        tuple_7597 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 152, 25), 'tuple', False)
        # Calling tuple(args, kwargs) (line 152)
        tuple_call_result_7608 = invoke(stypy.reporting.localization.Localization(__file__, 152, 25), tuple_7597, *[result_add_7606], **kwargs_7607)
        
        # Assigning a type to the variable 'argument_types' (line 152)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 152, 8), 'argument_types', tuple_call_result_7608)
        
        # Call to modifier(...): (line 153)
        # Processing the call arguments (line 153)
        # Getting the type of 'localization' (line 153)
        localization_7610 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 24), 'localization', False)
        # Getting the type of 'proxy_obj' (line 153)
        proxy_obj_7611 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 38), 'proxy_obj', False)
        # Getting the type of 'argument_types' (line 153)
        argument_types_7612 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 49), 'argument_types', False)
        # Processing the call keyword arguments (line 153)
        kwargs_7613 = {}
        # Getting the type of 'modifier' (line 153)
        modifier_7609 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 153, 15), 'modifier', False)
        # Calling modifier(args, kwargs) (line 153)
        modifier_call_result_7614 = invoke(stypy.reporting.localization.Localization(__file__, 153, 15), modifier_7609, *[localization_7610, proxy_obj_7611, argument_types_7612], **kwargs_7613)
        
        # Assigning a type to the variable 'stypy_return_type' (line 153)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 153, 8), 'stypy_return_type', modifier_call_result_7614)
        
        # ################# End of '__call__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function '__call__' in the type store
        # Getting the type of 'stypy_return_type' (line 133)
        stypy_return_type_7615 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 133, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_7615)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function '__call__'
        return stypy_return_type_7615


# Assigning a type to the variable 'FileTypeModifier' (line 9)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 9, 0), 'FileTypeModifier', FileTypeModifier)

# Assigning a Call to a Name (line 20):

# Call to dict(...): (line 20)
# Processing the call keyword arguments (line 20)
kwargs_7617 = {}
# Getting the type of 'dict' (line 20)
dict_7616 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 20, 22), 'dict', False)
# Calling dict(args, kwargs) (line 20)
dict_call_result_7618 = invoke(stypy.reporting.localization.Localization(__file__, 20, 22), dict_7616, *[], **kwargs_7617)

# Getting the type of 'FileTypeModifier'
FileTypeModifier_7619 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FileTypeModifier')
# Setting the type of the member 'modifiers_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FileTypeModifier_7619, 'modifiers_cache', dict_call_result_7618)

# Assigning a Call to a Name (line 23):

# Call to dict(...): (line 23)
# Processing the call keyword arguments (line 23)
kwargs_7621 = {}
# Getting the type of 'dict' (line 23)
dict_7620 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 34), 'dict', False)
# Calling dict(args, kwargs) (line 23)
dict_call_result_7622 = invoke(stypy.reporting.localization.Localization(__file__, 23, 34), dict_7620, *[], **kwargs_7621)

# Getting the type of 'FileTypeModifier'
FileTypeModifier_7623 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'FileTypeModifier')
# Setting the type of the member 'unavailable_modifiers_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), FileTypeModifier_7623, 'unavailable_modifiers_cache', dict_call_result_7622)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
