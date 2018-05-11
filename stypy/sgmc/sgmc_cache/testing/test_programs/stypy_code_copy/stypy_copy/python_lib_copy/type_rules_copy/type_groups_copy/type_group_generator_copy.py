
# -*- coding: utf-8 -*-

"""
ORIGINAL PROGRAM SOURCE CODE:
1: from type_groups_copy import *
2: 
3: 
4: class RuleGroupGenerator:
5:     '''
6:     This class is used to generate type group instances from lists of types defined in the TypeGroup object of the
7:     type_groups.py file, using the TypeGroup class as a canvas class to generate them.
8:     '''
9:     rule_group_cache = dict()
10: 
11:     def create_rule_group_class(self, class_name):
12:         '''
13:         Creates a new class named as class_name, with all the members of the TypeGroup class
14:         :param class_name: Name of the new class
15:         :return: A new class, structurally identical to the TypeGroup class. TypeGroup class with the same name can
16:         only be created once. If we try to create one that has been already created, the created one is returned instead
17:         '''
18:         if class_name in self.rule_group_cache.keys():
19:             return self.rule_group_cache[class_name]
20: 
21:         group_class = type(class_name, TypeGroup.__bases__, dict(TypeGroup.__dict__))
22:         instance = group_class(getattr(TypeGroups, class_name))
23:         self.rule_group_cache[class_name] = instance
24: 
25:         return instance
26: 
27:     def create_rule_group_class_list(self, classes_name):
28:         '''
29:         Mass-creation of rule group classes calling the previous method
30:         :param classes_name: List of class names
31:         :return: List of classes
32:         '''
33:         instances = []
34:         for class_name in classes_name:
35:             instance = self.create_rule_group_class(class_name)
36:             instances.append(instance)
37: 
38:         return instances
39: 
40:     def __init__(self):
41:         # self.rule_group_compliance_dict = dict()
42:         # for rule in TypeGroups.get_rule_groups():
43:         #     self.rule_group_compliance_dict[rule] = [False] * eval("len(TypeGroups.{0})".format(rule))
44:         self.added_types = []
45:         self.unclassified_types = []
46: 
47:     #TODO: Delete?
48:     # def add_type(self, type_):
49:     #     if not type_ in self.added_types:
50:     #         self.added_types.append(type_)
51:     #
52:     #     added = False
53:     #     for rule_group in TypeGroups.get_rule_groups():
54:     #         type_list = getattr(TypeGroups, rule_group)
55:     #         if type_ in type_list:
56:     #             for i in range(len(type_list)):
57:     #                 if type_ == type_list[i]:
58:     #                     self.rule_group_compliance_dict[rule_group][i] = True
59:     #                     added = True
60:     #
61:     #     if not added:
62:     #         if not type_ in self.unclassified_types:
63:     #             self.unclassified_types.append(type_)
64:     #
65:     # def get_rule_group(self):
66:     #     ret_rule_group = None
67:     #     added = False
68:     #
69:     #     for (marked_rule_group, marks) in self.rule_group_compliance_dict.items():
70:     #         true_marks = len(filter(lambda x: x == True, marks))
71:     #         if len(getattr(TypeGroups, marked_rule_group)) == true_marks:
72:     #             if ret_rule_group is None:
73:     #                 ret_rule_group = [marked_rule_group]
74:     #             else:
75:     #                 for i in range(len(ret_rule_group)):
76:     #                     if getattr(TypeGroups, marked_rule_group) > getattr(TypeGroups, ret_rule_group[i]):
77:     #                         ret_rule_group[i] = marked_rule_group
78:     #                         added = True
79:     #                     if getattr(TypeGroups, marked_rule_group) < getattr(TypeGroups, ret_rule_group[i]):
80:     #                         added = True
81:     #                 if not added:
82:     #                     ret_rule_group.append(marked_rule_group)
83:     #
84:     #     if ret_rule_group is not None:
85:     #         ret_list = self.create_rule_group_class_list(ret_rule_group)
86:     #         return ret_list + self.unclassified_types
87:     #     else:
88:     #         if len(self.unclassified_types) == 0:
89:     #             return None
90:     #         else:
91:     #             return self.unclassified_types
92:     #
93:     # def is_type_in_rule_group(self, rule_group, type_):
94:     #     return type_ in getattr(TypeGroups, rule_group)
95: 
96: '''
97: TypeGroups composed by collections of types
98: '''
99: RealNumber = RuleGroupGenerator().create_rule_group_class("RealNumber")
100: Number = RuleGroupGenerator().create_rule_group_class("Number")
101: Integer = RuleGroupGenerator().create_rule_group_class("Integer")
102: Str = RuleGroupGenerator().create_rule_group_class("Str")
103: IterableDataStructure = RuleGroupGenerator().create_rule_group_class("IterableDataStructure")
104: IterableObject = RuleGroupGenerator().create_rule_group_class("IterableObject")
105: ByteSequence = RuleGroupGenerator().create_rule_group_class("ByteSequence")
106: 

"""

# Import the stypy library necessary elements
from stypy.type_inference_programs.type_inference_programs_imports import *

# Create the module type store
module_type_store = Context(None, __file__)

# ################# Begin of the type inference program ##################

stypy.reporting.localization.Localization.set_current(stypy.reporting.localization.Localization(__file__, 1, 0))

# 'from type_groups_copy import ' statement (line 1)
update_path_to_current_file_folder('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')
import_16033 = generate_type_inference_code_for_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_groups_copy')

if (type(import_16033) is not StypyTypeError):

    if (import_16033 != 'pyd_module'):
        __import__(import_16033)
        sys_modules_16034 = sys.modules[import_16033]
        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_groups_copy', sys_modules_16034.module_type_store, module_type_store, ['*'])
        nest_module(stypy.reporting.localization.Localization(__file__, 1, 0), __file__, sys_modules_16034, sys_modules_16034.module_type_store, module_type_store)
    else:
        from type_groups_copy import *

        import_from_module(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_groups_copy', None, module_type_store, ['*'], None)

else:
    # Assigning a type to the variable 'type_groups_copy' (line 1)
    module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 1, 0), 'type_groups_copy', import_16033)

remove_current_file_folder_from_path('C:/Users/redon/PycharmProjects/stypyV2/testing/test_programs/stypy_code_copy/stypy_copy/python_lib_copy/type_rules_copy/type_groups_copy/')

# Declaration of the 'RuleGroupGenerator' class

class RuleGroupGenerator:
    str_16035 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 8, (-1)), 'str', '\n    This class is used to generate type group instances from lists of types defined in the TypeGroup object of the\n    type_groups.py file, using the TypeGroup class as a canvas class to generate them.\n    ')

    @norecursion
    def create_rule_group_class(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_rule_group_class'
        module_type_store = module_type_store.open_function_context('create_rule_group_class', 11, 4, False)
        # Assigning a type to the variable 'self' (line 12)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 12, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_localization', localization)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_type_store', module_type_store)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_function_name', 'RuleGroupGenerator.create_rule_group_class')
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_param_names_list', ['class_name'])
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_varargs_param_name', None)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_call_defaults', defaults)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_call_varargs', varargs)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RuleGroupGenerator.create_rule_group_class.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RuleGroupGenerator.create_rule_group_class', ['class_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_rule_group_class', localization, ['class_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_rule_group_class(...)' code ##################

        str_16036 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 17, (-1)), 'str', '\n        Creates a new class named as class_name, with all the members of the TypeGroup class\n        :param class_name: Name of the new class\n        :return: A new class, structurally identical to the TypeGroup class. TypeGroup class with the same name can\n        only be created once. If we try to create one that has been already created, the created one is returned instead\n        ')
        
        # Getting the type of 'class_name' (line 18)
        class_name_16037 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 11), 'class_name')
        
        # Call to keys(...): (line 18)
        # Processing the call keyword arguments (line 18)
        kwargs_16041 = {}
        # Getting the type of 'self' (line 18)
        self_16038 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 18, 25), 'self', False)
        # Obtaining the member 'rule_group_cache' of a type (line 18)
        rule_group_cache_16039 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), self_16038, 'rule_group_cache')
        # Obtaining the member 'keys' of a type (line 18)
        keys_16040 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 18, 25), rule_group_cache_16039, 'keys')
        # Calling keys(args, kwargs) (line 18)
        keys_call_result_16042 = invoke(stypy.reporting.localization.Localization(__file__, 18, 25), keys_16040, *[], **kwargs_16041)
        
        # Applying the binary operator 'in' (line 18)
        result_contains_16043 = python_operator(stypy.reporting.localization.Localization(__file__, 18, 11), 'in', class_name_16037, keys_call_result_16042)
        
        # Testing if the type of an if condition is none (line 18)

        if evaluates_to_none(stypy.reporting.localization.Localization(__file__, 18, 8), result_contains_16043):
            pass
        else:
            
            # Testing the type of an if condition (line 18)
            if_condition_16044 = is_suitable_condition(stypy.reporting.localization.Localization(__file__, 18, 8), result_contains_16043)
            # Assigning a type to the variable 'if_condition_16044' (line 18)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 18, 8), 'if_condition_16044', if_condition_16044)
            # SSA begins for if statement (line 18)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'if')
            
            # Obtaining the type of the subscript
            # Getting the type of 'class_name' (line 19)
            class_name_16045 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 41), 'class_name')
            # Getting the type of 'self' (line 19)
            self_16046 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 19, 19), 'self')
            # Obtaining the member 'rule_group_cache' of a type (line 19)
            rule_group_cache_16047 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), self_16046, 'rule_group_cache')
            # Obtaining the member '__getitem__' of a type (line 19)
            getitem___16048 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 19, 19), rule_group_cache_16047, '__getitem__')
            # Calling the subscript (__getitem__) to obtain the elements type (line 19)
            subscript_call_result_16049 = invoke(stypy.reporting.localization.Localization(__file__, 19, 19), getitem___16048, class_name_16045)
            
            # Assigning a type to the variable 'stypy_return_type' (line 19)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 19, 12), 'stypy_return_type', subscript_call_result_16049)
            # SSA join for if statement (line 18)
            module_type_store = module_type_store.join_ssa_context()
            

        
        # Assigning a Call to a Name (line 21):
        
        # Call to type(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'class_name' (line 21)
        class_name_16051 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 27), 'class_name', False)
        # Getting the type of 'TypeGroup' (line 21)
        TypeGroup_16052 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 39), 'TypeGroup', False)
        # Obtaining the member '__bases__' of a type (line 21)
        bases___16053 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 39), TypeGroup_16052, '__bases__')
        
        # Call to dict(...): (line 21)
        # Processing the call arguments (line 21)
        # Getting the type of 'TypeGroup' (line 21)
        TypeGroup_16055 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 65), 'TypeGroup', False)
        # Obtaining the member '__dict__' of a type (line 21)
        dict___16056 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 21, 65), TypeGroup_16055, '__dict__')
        # Processing the call keyword arguments (line 21)
        kwargs_16057 = {}
        # Getting the type of 'dict' (line 21)
        dict_16054 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 60), 'dict', False)
        # Calling dict(args, kwargs) (line 21)
        dict_call_result_16058 = invoke(stypy.reporting.localization.Localization(__file__, 21, 60), dict_16054, *[dict___16056], **kwargs_16057)
        
        # Processing the call keyword arguments (line 21)
        kwargs_16059 = {}
        # Getting the type of 'type' (line 21)
        type_16050 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 21, 22), 'type', False)
        # Calling type(args, kwargs) (line 21)
        type_call_result_16060 = invoke(stypy.reporting.localization.Localization(__file__, 21, 22), type_16050, *[class_name_16051, bases___16053, dict_call_result_16058], **kwargs_16059)
        
        # Assigning a type to the variable 'group_class' (line 21)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 21, 8), 'group_class', type_call_result_16060)
        
        # Assigning a Call to a Name (line 22):
        
        # Call to group_class(...): (line 22)
        # Processing the call arguments (line 22)
        
        # Call to getattr(...): (line 22)
        # Processing the call arguments (line 22)
        # Getting the type of 'TypeGroups' (line 22)
        TypeGroups_16063 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 39), 'TypeGroups', False)
        # Getting the type of 'class_name' (line 22)
        class_name_16064 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 51), 'class_name', False)
        # Processing the call keyword arguments (line 22)
        kwargs_16065 = {}
        # Getting the type of 'getattr' (line 22)
        getattr_16062 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 31), 'getattr', False)
        # Calling getattr(args, kwargs) (line 22)
        getattr_call_result_16066 = invoke(stypy.reporting.localization.Localization(__file__, 22, 31), getattr_16062, *[TypeGroups_16063, class_name_16064], **kwargs_16065)
        
        # Processing the call keyword arguments (line 22)
        kwargs_16067 = {}
        # Getting the type of 'group_class' (line 22)
        group_class_16061 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 22, 19), 'group_class', False)
        # Calling group_class(args, kwargs) (line 22)
        group_class_call_result_16068 = invoke(stypy.reporting.localization.Localization(__file__, 22, 19), group_class_16061, *[getattr_call_result_16066], **kwargs_16067)
        
        # Assigning a type to the variable 'instance' (line 22)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 22, 8), 'instance', group_class_call_result_16068)
        
        # Assigning a Name to a Subscript (line 23):
        # Getting the type of 'instance' (line 23)
        instance_16069 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 44), 'instance')
        # Getting the type of 'self' (line 23)
        self_16070 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 8), 'self')
        # Obtaining the member 'rule_group_cache' of a type (line 23)
        rule_group_cache_16071 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 23, 8), self_16070, 'rule_group_cache')
        # Getting the type of 'class_name' (line 23)
        class_name_16072 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 23, 30), 'class_name')
        # Storing an element on a container (line 23)
        set_contained_elements_type(stypy.reporting.localization.Localization(__file__, 23, 8), rule_group_cache_16071, (class_name_16072, instance_16069))
        # Getting the type of 'instance' (line 25)
        instance_16073 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 25, 15), 'instance')
        # Assigning a type to the variable 'stypy_return_type' (line 25)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 25, 8), 'stypy_return_type', instance_16073)
        
        # ################# End of 'create_rule_group_class(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_rule_group_class' in the type store
        # Getting the type of 'stypy_return_type' (line 11)
        stypy_return_type_16074 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 11, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16074)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_rule_group_class'
        return stypy_return_type_16074


    @norecursion
    def create_rule_group_class_list(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function 'create_rule_group_class_list'
        module_type_store = module_type_store.open_function_context('create_rule_group_class_list', 27, 4, False)
        # Assigning a type to the variable 'self' (line 28)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 28, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_localization', localization)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_type_of_self', type_of_self)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_type_store', module_type_store)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_function_name', 'RuleGroupGenerator.create_rule_group_class_list')
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_param_names_list', ['classes_name'])
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_varargs_param_name', None)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_kwargs_param_name', None)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_call_defaults', defaults)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_call_varargs', varargs)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_call_kwargs', kwargs)
        RuleGroupGenerator.create_rule_group_class_list.__dict__.__setitem__('stypy_declared_arg_number', 2)
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RuleGroupGenerator.create_rule_group_class_list', ['classes_name'], None, None, defaults, varargs, kwargs)

        if is_error_type(arguments):
            # Destroy the current context
            module_type_store = module_type_store.close_function_context()
            return arguments

        # Initialize method data
        init_call_information(module_type_store, 'create_rule_group_class_list', localization, ['classes_name'], arguments)
        
        # Default return type storage variable (SSA)
        # Assigning a type to the variable 'stypy_return_type'
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'stypy_return_type', None)
        
        
        # ################# Begin of 'create_rule_group_class_list(...)' code ##################

        str_16075 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 32, (-1)), 'str', '\n        Mass-creation of rule group classes calling the previous method\n        :param classes_name: List of class names\n        :return: List of classes\n        ')
        
        # Assigning a List to a Name (line 33):
        
        # Obtaining an instance of the builtin type 'list' (line 33)
        list_16076 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 33, 20), 'list')
        # Adding type elements to the builtin type 'list' instance (line 33)
        
        # Assigning a type to the variable 'instances' (line 33)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 33, 8), 'instances', list_16076)
        
        # Getting the type of 'classes_name' (line 34)
        classes_name_16077 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 34, 26), 'classes_name')
        # Assigning a type to the variable 'classes_name_16077' (line 34)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'classes_name_16077', classes_name_16077)
        # Testing if the for loop is going to be iterated (line 34)
        # Testing the type of a for loop iterable (line 34)
        is_suitable_for_loop_condition(stypy.reporting.localization.Localization(__file__, 34, 8), classes_name_16077)

        if will_iterate_loop(stypy.reporting.localization.Localization(__file__, 34, 8), classes_name_16077):
            # Getting the type of the for loop variable (line 34)
            for_loop_var_16078 = get_type_of_for_loop_variable(stypy.reporting.localization.Localization(__file__, 34, 8), classes_name_16077)
            # Assigning a type to the variable 'class_name' (line 34)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 34, 8), 'class_name', for_loop_var_16078)
            # SSA begins for a for statement (line 34)
            module_type_store = SSAContext.create_ssa_context(module_type_store, 'for loop')
            
            # Assigning a Call to a Name (line 35):
            
            # Call to create_rule_group_class(...): (line 35)
            # Processing the call arguments (line 35)
            # Getting the type of 'class_name' (line 35)
            class_name_16081 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 52), 'class_name', False)
            # Processing the call keyword arguments (line 35)
            kwargs_16082 = {}
            # Getting the type of 'self' (line 35)
            self_16079 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 35, 23), 'self', False)
            # Obtaining the member 'create_rule_group_class' of a type (line 35)
            create_rule_group_class_16080 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 35, 23), self_16079, 'create_rule_group_class')
            # Calling create_rule_group_class(args, kwargs) (line 35)
            create_rule_group_class_call_result_16083 = invoke(stypy.reporting.localization.Localization(__file__, 35, 23), create_rule_group_class_16080, *[class_name_16081], **kwargs_16082)
            
            # Assigning a type to the variable 'instance' (line 35)
            module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 35, 12), 'instance', create_rule_group_class_call_result_16083)
            
            # Call to append(...): (line 36)
            # Processing the call arguments (line 36)
            # Getting the type of 'instance' (line 36)
            instance_16086 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 29), 'instance', False)
            # Processing the call keyword arguments (line 36)
            kwargs_16087 = {}
            # Getting the type of 'instances' (line 36)
            instances_16084 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 36, 12), 'instances', False)
            # Obtaining the member 'append' of a type (line 36)
            append_16085 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 36, 12), instances_16084, 'append')
            # Calling append(args, kwargs) (line 36)
            append_call_result_16088 = invoke(stypy.reporting.localization.Localization(__file__, 36, 12), append_16085, *[instance_16086], **kwargs_16087)
            
            # SSA join for a for statement
            module_type_store = module_type_store.join_ssa_context()

        
        # Getting the type of 'instances' (line 38)
        instances_16089 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 38, 15), 'instances')
        # Assigning a type to the variable 'stypy_return_type' (line 38)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 38, 8), 'stypy_return_type', instances_16089)
        
        # ################# End of 'create_rule_group_class_list(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        
        # Storing the return type of function 'create_rule_group_class_list' in the type store
        # Getting the type of 'stypy_return_type' (line 27)
        stypy_return_type_16090 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 27, 4), 'stypy_return_type')
        module_type_store.store_return_type_of_current_context(stypy_return_type_16090)
        
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()
        
        # Return type of the function 'create_rule_group_class_list'
        return stypy_return_type_16090


    @norecursion
    def __init__(type_of_self, localization, *varargs, **kwargs):
        global module_type_store
        # Assign values to the parameters with defaults
        defaults = []
        # Create a new context for function '__init__'
        module_type_store = module_type_store.open_function_context('__init__', 40, 4, False)
        # Assigning a type to the variable 'self' (line 41)
        module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 41, 4), 'self', type_of_self)
        
        # Passed parameters checking function
        arguments = process_argument_values(localization, type_of_self, module_type_store, 'RuleGroupGenerator.__init__', [], None, None, defaults, varargs, kwargs)

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

        
        # Assigning a List to a Attribute (line 44):
        
        # Obtaining an instance of the builtin type 'list' (line 44)
        list_16091 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 44, 27), 'list')
        # Adding type elements to the builtin type 'list' instance (line 44)
        
        # Getting the type of 'self' (line 44)
        self_16092 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 44, 8), 'self')
        # Setting the type of the member 'added_types' of a type (line 44)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 44, 8), self_16092, 'added_types', list_16091)
        
        # Assigning a List to a Attribute (line 45):
        
        # Obtaining an instance of the builtin type 'list' (line 45)
        list_16093 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 45, 34), 'list')
        # Adding type elements to the builtin type 'list' instance (line 45)
        
        # Getting the type of 'self' (line 45)
        self_16094 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 45, 8), 'self')
        # Setting the type of the member 'unclassified_types' of a type (line 45)
        module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 45, 8), self_16094, 'unclassified_types', list_16093)
        
        # ################# End of '__init__(...)' code ##################

        # Teardown call information
        teardown_call_information(localization, arguments)
        # Destroy the current context
        module_type_store = module_type_store.close_function_context()


# Assigning a type to the variable 'RuleGroupGenerator' (line 4)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 4, 0), 'RuleGroupGenerator', RuleGroupGenerator)

# Assigning a Call to a Name (line 9):

# Call to dict(...): (line 9)
# Processing the call keyword arguments (line 9)
kwargs_16096 = {}
# Getting the type of 'dict' (line 9)
dict_16095 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 9, 23), 'dict', False)
# Calling dict(args, kwargs) (line 9)
dict_call_result_16097 = invoke(stypy.reporting.localization.Localization(__file__, 9, 23), dict_16095, *[], **kwargs_16096)

# Getting the type of 'RuleGroupGenerator'
RuleGroupGenerator_16098 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 0, 0), 'RuleGroupGenerator')
# Setting the type of the member 'rule_group_cache' of a type
module_type_store.set_type_of_member(stypy.reporting.localization.Localization(__file__, 0, 0), RuleGroupGenerator_16098, 'rule_group_cache', dict_call_result_16097)
str_16099 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 98, (-1)), 'str', '\nTypeGroups composed by collections of types\n')

# Assigning a Call to a Name (line 99):

# Call to create_rule_group_class(...): (line 99)
# Processing the call arguments (line 99)
str_16104 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 99, 58), 'str', 'RealNumber')
# Processing the call keyword arguments (line 99)
kwargs_16105 = {}

# Call to RuleGroupGenerator(...): (line 99)
# Processing the call keyword arguments (line 99)
kwargs_16101 = {}
# Getting the type of 'RuleGroupGenerator' (line 99)
RuleGroupGenerator_16100 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 99, 13), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 99)
RuleGroupGenerator_call_result_16102 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), RuleGroupGenerator_16100, *[], **kwargs_16101)

# Obtaining the member 'create_rule_group_class' of a type (line 99)
create_rule_group_class_16103 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 99, 13), RuleGroupGenerator_call_result_16102, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 99)
create_rule_group_class_call_result_16106 = invoke(stypy.reporting.localization.Localization(__file__, 99, 13), create_rule_group_class_16103, *[str_16104], **kwargs_16105)

# Assigning a type to the variable 'RealNumber' (line 99)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 99, 0), 'RealNumber', create_rule_group_class_call_result_16106)

# Assigning a Call to a Name (line 100):

# Call to create_rule_group_class(...): (line 100)
# Processing the call arguments (line 100)
str_16111 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 100, 54), 'str', 'Number')
# Processing the call keyword arguments (line 100)
kwargs_16112 = {}

# Call to RuleGroupGenerator(...): (line 100)
# Processing the call keyword arguments (line 100)
kwargs_16108 = {}
# Getting the type of 'RuleGroupGenerator' (line 100)
RuleGroupGenerator_16107 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 100, 9), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 100)
RuleGroupGenerator_call_result_16109 = invoke(stypy.reporting.localization.Localization(__file__, 100, 9), RuleGroupGenerator_16107, *[], **kwargs_16108)

# Obtaining the member 'create_rule_group_class' of a type (line 100)
create_rule_group_class_16110 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 100, 9), RuleGroupGenerator_call_result_16109, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 100)
create_rule_group_class_call_result_16113 = invoke(stypy.reporting.localization.Localization(__file__, 100, 9), create_rule_group_class_16110, *[str_16111], **kwargs_16112)

# Assigning a type to the variable 'Number' (line 100)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 100, 0), 'Number', create_rule_group_class_call_result_16113)

# Assigning a Call to a Name (line 101):

# Call to create_rule_group_class(...): (line 101)
# Processing the call arguments (line 101)
str_16118 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 101, 55), 'str', 'Integer')
# Processing the call keyword arguments (line 101)
kwargs_16119 = {}

# Call to RuleGroupGenerator(...): (line 101)
# Processing the call keyword arguments (line 101)
kwargs_16115 = {}
# Getting the type of 'RuleGroupGenerator' (line 101)
RuleGroupGenerator_16114 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 101, 10), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 101)
RuleGroupGenerator_call_result_16116 = invoke(stypy.reporting.localization.Localization(__file__, 101, 10), RuleGroupGenerator_16114, *[], **kwargs_16115)

# Obtaining the member 'create_rule_group_class' of a type (line 101)
create_rule_group_class_16117 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 101, 10), RuleGroupGenerator_call_result_16116, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 101)
create_rule_group_class_call_result_16120 = invoke(stypy.reporting.localization.Localization(__file__, 101, 10), create_rule_group_class_16117, *[str_16118], **kwargs_16119)

# Assigning a type to the variable 'Integer' (line 101)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 101, 0), 'Integer', create_rule_group_class_call_result_16120)

# Assigning a Call to a Name (line 102):

# Call to create_rule_group_class(...): (line 102)
# Processing the call arguments (line 102)
str_16125 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 102, 51), 'str', 'Str')
# Processing the call keyword arguments (line 102)
kwargs_16126 = {}

# Call to RuleGroupGenerator(...): (line 102)
# Processing the call keyword arguments (line 102)
kwargs_16122 = {}
# Getting the type of 'RuleGroupGenerator' (line 102)
RuleGroupGenerator_16121 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 102, 6), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 102)
RuleGroupGenerator_call_result_16123 = invoke(stypy.reporting.localization.Localization(__file__, 102, 6), RuleGroupGenerator_16121, *[], **kwargs_16122)

# Obtaining the member 'create_rule_group_class' of a type (line 102)
create_rule_group_class_16124 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 102, 6), RuleGroupGenerator_call_result_16123, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 102)
create_rule_group_class_call_result_16127 = invoke(stypy.reporting.localization.Localization(__file__, 102, 6), create_rule_group_class_16124, *[str_16125], **kwargs_16126)

# Assigning a type to the variable 'Str' (line 102)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 102, 0), 'Str', create_rule_group_class_call_result_16127)

# Assigning a Call to a Name (line 103):

# Call to create_rule_group_class(...): (line 103)
# Processing the call arguments (line 103)
str_16132 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 103, 69), 'str', 'IterableDataStructure')
# Processing the call keyword arguments (line 103)
kwargs_16133 = {}

# Call to RuleGroupGenerator(...): (line 103)
# Processing the call keyword arguments (line 103)
kwargs_16129 = {}
# Getting the type of 'RuleGroupGenerator' (line 103)
RuleGroupGenerator_16128 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 103, 24), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 103)
RuleGroupGenerator_call_result_16130 = invoke(stypy.reporting.localization.Localization(__file__, 103, 24), RuleGroupGenerator_16128, *[], **kwargs_16129)

# Obtaining the member 'create_rule_group_class' of a type (line 103)
create_rule_group_class_16131 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 103, 24), RuleGroupGenerator_call_result_16130, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 103)
create_rule_group_class_call_result_16134 = invoke(stypy.reporting.localization.Localization(__file__, 103, 24), create_rule_group_class_16131, *[str_16132], **kwargs_16133)

# Assigning a type to the variable 'IterableDataStructure' (line 103)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 103, 0), 'IterableDataStructure', create_rule_group_class_call_result_16134)

# Assigning a Call to a Name (line 104):

# Call to create_rule_group_class(...): (line 104)
# Processing the call arguments (line 104)
str_16139 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 104, 62), 'str', 'IterableObject')
# Processing the call keyword arguments (line 104)
kwargs_16140 = {}

# Call to RuleGroupGenerator(...): (line 104)
# Processing the call keyword arguments (line 104)
kwargs_16136 = {}
# Getting the type of 'RuleGroupGenerator' (line 104)
RuleGroupGenerator_16135 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 104, 17), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 104)
RuleGroupGenerator_call_result_16137 = invoke(stypy.reporting.localization.Localization(__file__, 104, 17), RuleGroupGenerator_16135, *[], **kwargs_16136)

# Obtaining the member 'create_rule_group_class' of a type (line 104)
create_rule_group_class_16138 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 104, 17), RuleGroupGenerator_call_result_16137, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 104)
create_rule_group_class_call_result_16141 = invoke(stypy.reporting.localization.Localization(__file__, 104, 17), create_rule_group_class_16138, *[str_16139], **kwargs_16140)

# Assigning a type to the variable 'IterableObject' (line 104)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 104, 0), 'IterableObject', create_rule_group_class_call_result_16141)

# Assigning a Call to a Name (line 105):

# Call to create_rule_group_class(...): (line 105)
# Processing the call arguments (line 105)
str_16146 = get_builtin_python_type_instance(stypy.reporting.localization.Localization(__file__, 105, 60), 'str', 'ByteSequence')
# Processing the call keyword arguments (line 105)
kwargs_16147 = {}

# Call to RuleGroupGenerator(...): (line 105)
# Processing the call keyword arguments (line 105)
kwargs_16143 = {}
# Getting the type of 'RuleGroupGenerator' (line 105)
RuleGroupGenerator_16142 = module_type_store.get_type_of(stypy.reporting.localization.Localization(__file__, 105, 15), 'RuleGroupGenerator', False)
# Calling RuleGroupGenerator(args, kwargs) (line 105)
RuleGroupGenerator_call_result_16144 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), RuleGroupGenerator_16142, *[], **kwargs_16143)

# Obtaining the member 'create_rule_group_class' of a type (line 105)
create_rule_group_class_16145 = module_type_store.get_type_of_member(stypy.reporting.localization.Localization(__file__, 105, 15), RuleGroupGenerator_call_result_16144, 'create_rule_group_class')
# Calling create_rule_group_class(args, kwargs) (line 105)
create_rule_group_class_call_result_16148 = invoke(stypy.reporting.localization.Localization(__file__, 105, 15), create_rule_group_class_16145, *[str_16146], **kwargs_16147)

# Assigning a type to the variable 'ByteSequence' (line 105)
module_type_store.set_type_of(stypy.reporting.localization.Localization(__file__, 105, 0), 'ByteSequence', create_rule_group_class_call_result_16148)

# ################# End of the type inference program ##################

module_errors = stypy.errors.type_error.StypyTypeError.get_error_msgs()
module_warnings = stypy.errors.type_warning.TypeWarning.get_warning_msgs()
