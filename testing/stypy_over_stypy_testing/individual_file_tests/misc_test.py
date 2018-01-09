import types


class TypeError:
    def __init__(self, msg):
        self.msg = msg

    def __repr__(self):
        return "Error!: " + self.msg

    def __str__(self):
        return self.__repr__()


def supports_intercession(entity):
    return not type(entity.__dict__) is types.DictProxyType


def get_type_of_member(obj, member):
    if hasattr(obj, "stypy_types_of_members"):
        if member in obj.stypy_types_of_members:
            return obj.stypy_types_of_members[member]

    if hasattr(obj, member):
        return type(getattr(obj, member))
    else:
        return TypeError("Member " + member + " do not exist in " + str(obj))


def set_type_of_member(obj, member, type):
    try:
        if not hasattr(obj, "stypy_types_of_members"):
            obj.stypy_types_of_members = {}

        obj.stypy_types_of_members[member] = type
    except Exception as exc:
        TypeError(str(exc))

def get_type(obj):
    if isinstance(obj, CloneDelta):
        return object.__getattribute__(obj, "python_type")
    else:
        return type(obj)


def clone(obj):
    if not supports_intercession(obj):
        return obj
    else:
        if isinstance(obj, CloneDelta):
            obj.clone_delta.append(dict())
            return obj

        return CloneDelta(obj)

def get_type_of_member_from_clone_delta (clone_delta_stack, attribute_name):
    return None

def set_type_of_member_from_clone_delta (clone_delta_stack, attribute_name, attribute_type):
    return None

non_overloadable_attributes = ["__dict__", "python_type", "stypy_types_of_members", "__repr__",
                               "is_clone", "clone_delta"]


class CloneDelta(object):
    def __init__(self, python_type):
        # The python entity will be now read-only
        object.__setattr__(self, "python_type", python_type)
        object.__setattr__(self, "clone_delta", list(dict()))

    def __getattribute__(self, attribute_name):
        global non_overloadable_attributes

        python_type = object.__getattribute__(self, "python_type")

        if attribute_name in non_overloadable_attributes:
            return object.__getattribute__(python_type, attribute_name)
        else:
            value_from_delta = get_type_of_member_from_clone_delta(self.clone_delta, attribute_name)
            if value_from_delta is not None:
                return value_from_delta

            return get_type_of_member(python_type, attribute_name)

    def __setattr__(self, attribute_name, attribute_value):
        global non_overloadable_attributes

        python_type = object.__getattribute__(self, "python_type")

        try:
            if attribute_name in non_overloadable_attributes:
                object.__setattr__(python_type, attribute_name, attribute_value)
            else:
                set_type_of_member_from_clone_delta(self.clone_delta, attribute_name, attribute_value)

                #set_type_of_member(python_type, attribute_name, attribute_value)
        except Exception as exc:
            TypeError(str(exc))

    def __repr__(self):
        s = self.python_type.__name__

        return s

    def __str__(self):
        return self.__repr__()

            # def get_type_of_member(self, member):
            #     if self.is_clone():
            #         if self.clone_delta.has_key(member):
            #             return self.clone_delta[member]
            #
            #     if member not in self.types_of_members:
            #         value = self.python_entity.__dict__[member]
            #         self.types_of_members[member] = TypeInferenceProxy.get_instance(type(value))
            #
            #     return self.types_of_members[member]
            #
            # def set_type_of_member(self, member, member_type):
            #     self.declared_members.add(member)
            #     member_type = TypeInferenceProxy.get_instance(member_type)
            #
            #     if self.is_clone():
            #         self.clone_delta[member] = member_type
            #     else:
            #         self.types_of_members[member] = member_type
            #
            # def __consolidate_clone(self, clone_delta):
            #     for member in clone_delta:
            #         self.set_type_of_member(member, clone_delta[member].clone())
            #
            # def clone(self):
            #     if not supports_intercession(self.python_entity):
            #         return self
            #
            #     cloned_obj = TypeInferenceProxy(self.python_entity)
            #     cloned_obj.types_of_members = dict(
            #         map(lambda tuple_: (tuple_[0], tuple_[1].clone()), self.types_of_members.items()))
            #     cloned_obj.clone_delta = {}
            #
            #     if self.is_clone():
            #         cloned_obj.__consolidate_clone(self.clone_delta)
            #
            #     return cloned_obj
            #
            # def __dir__(self):
            #     return list(self.declared_members)
            #
            # def __repr__(self):
            #     s = self.python_entity.__name__
            #     # for v in dir(self.declared_members):
            #     #     s += v + ": " + str(self.get_type_of_member(v)) + "\n"
            #
            #     return s
            #
            # def __str__(self):
            #     return self.__repr__()


# t1 = int
#
# f = get_type_of_member(t1, "__doc__")
# print f
# print get_type_of_member(f, "title")
#
# tc1 = CloneDelta(t1)
# f = get_type_of_member(tc1, "__doc__")
# print f
# print get_type_of_member(f, "title")
#
# set_type_of_member(f, "title", int)
# print get_type_of_member(f, "title")

import math
f = get_type_of_member(math, "cos")
print f
set_type_of_member(math, "cos", int)
f = get_type_of_member(math, "cos")
print f

m2 = clone(math)
f = get_type_of_member(m2, "cos")
print f
print m2
print get_type(m2)
# f2 = f.clone()

# print f
# print f2
# print f2.is_clone()
#
# f.set_type_of_member("foo", int)
#
# print f.get_type_of_member("foo")
# try:
#     print f2.get_type_of_member("foo")
# except:
#     print "error"
#
# f3 = f.clone()
#
# try:
#     print f3.get_type_of_member("foo")
# except:
#     print "error"



# clone a simple type


# clone a data structure

# clone a function

# clone a class

# clone a module

# clone a function context

# clone a type store





# import math
#
# math_clone = clone(math)
#
# math_clone.set("foo", int)
# math_clone.set("bar", str)
# math_clone.set("cos", int)
#
# print dir(math)
# print dir(math_clone)
#
# print math.get("cos")

# fc1 = FC()
# fc1.set("x", int)
# fc1.set("y", float)
# fc1.set("z", str)
#
# fc2 = fc1.clone()
# fc2.set("y", list)
# fc2.set("w", complex)
#
# print fc1
# print fc2



# def supports_structural_reflection(obj):
#     if not hasattr(obj, '__dict__'):
#         return False
#
#     if type(obj.__dict__) is dict:
#         return True
#     else:
#         try:
#             obj.__dict__["__stypy_probe"] = None
#             del obj.__dict__["__stypy_probe"]
#             return True
#         except:
#             return False
#
# class Foo:
#     pass


# y = supports_structural_reflection(Foo())

# #Probablemente haya un problema en el clone de TypeInferenceProxy
#
# import sys
# import inspect
#
# class Type:
#     def set_type_instance(self, value):
#         """
#         Change the type instance value
#         :param value:
#         :return:
#         """
#         self.type_instance = value
#
# class UndefinedType:
#     pass
#
# class TypeInferenceProxy(Type):
#     def __init__(self, python_entity, name=None, parent=None, instance=None, value=UndefinedType):
#         """
#         Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This constructor
#         should NOT be called directly. Use the instance(...) method instead to take advantage of the implemented
#         type memoization of this class.
#         :param python_entity: Represented python entity.
#         :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property
#         value is used instead. Instances have an special name indicating that this entity holds an instance of a class.
#         :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.
#         :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead
#         of representing the class is representing a particular class instance. This is important to properly model
#         instance intercession, as altering the structure of single class instances is possible.
#         """
#         if name is None:
#             if hasattr(python_entity, "__name__"):
#                 self.name = python_entity.__name__
#             else:
#                 if hasattr(python_entity, "__class__"):
#                     self.name = python_entity.__class__.__name__
#                 else:
#                     if hasattr(python_entity, "__module__"):
#                         self.name = python_entity.__module__
#
#             if instance is not None:
#                 self.name = "<" + self.name + " instance>"
#         else:
#             self.name = name
#
#         self.python_entity = python_entity
#         self.__assign_parent_proxy(parent)
#         self.instance = instance
#         if instance is not None:
#             self.set_type_instance(True)
#
#         # Attribute values that have not been name (structure of the object is not known)
#         self.additional_members = list()
#
#         # If this is a type, store the original variable whose type is
#         self.type_of = None
#
#         # Store if the structure of the object is fully known or it has been manipulated without knowing precise
#         # attribute values
#         self.known_structure = True
#
#         if value is not UndefinedType:
#             self.value = value
#             self.set_type_instance(True)
#
#     def __assign_parent_proxy(self, parent):
#         """
#         Changes the parent object of the represented object to the one specified. This is used to trace the nesting
#         of proxies that hold types that are placed inside other proxies represented entities. This property is NOT
#         related with dynamic inheritance.
#         :param parent: The new parent object or None. If the passed parent is None, the class tries to autocalculate it.
#         If there is no possible parent, it is assigned to None
#         """
#         if parent is not None:
#             self.parent_proxy = parent
#         else:
#             if not inspect.ismodule(self.python_entity):
#                 self.parent_proxy = TypeInferenceProxy.__get_parent_proxy(self.python_entity)
#             else:
#                 self.parent_proxy = None  # Root Python entity
#
#     def clone(self):
#         return self
#
#     @staticmethod
#     def instance(python_entity, name=None, parent=None, instance=None, value=None):
#         """
#         Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This is the
#         preferred way to create proxy instances, as this method implement a memoization optimization.
#
#         :param python_entity: Represented python entity.
#         :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property
#         value is used instead. Instances have an special name indicating that this entity holds an instance of a class.
#         :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.
#         :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead
#         of representing the class is representing a particular class instance. This is important to properly model
#         instance intercession, as altering the structure of single instances is possible.
#         """
#
#         # TODO: Remove? Disabled because identity problems
#         # try:
#         #     if python_entity in TypeInferenceProxy.type_proxy_cache:
#         #         return TypeInferenceProxy.type_proxy_cache[python_entity]
#         # except:
#         #     pass
#
#         if isinstance(python_entity, Type):
#             return python_entity
#
#         return TypeInferenceProxy(python_entity, name, parent, instance, value)
#
#     @staticmethod
#     def __get_parent_proxy(parent):
#         """
#         Gets the python entity that can be considered the "parent" of the passed entity
#         :param parent: Any Python entity
#         :return: The parent of this entity, if any
#         """
#         if hasattr(parent, '__module__'):
#             return TypeInferenceProxy.instance(inspect.getmodule(parent))
#         if hasattr(parent, '__class__'):
#             return TypeInferenceProxy.instance(parent.__class__)
#
#         return None
#
# def __put_module_in_sys_cache(module_name, module_obj):
#     """
#     Puts a module in the sys stypy module cache
#     :param module_name: Name of the module
#     :param module_obj: Object representing the module
#     :return: None
#     """
#     #try:
#         #if hasattr(sys, 'stypy_module_cache'):
#     sys.stypy_module_cache[module_name] = module_obj
#         # else:
#         #     __preload_sys_module_cache()
#         #     sys.stypy_module_cache[module_name] = module_obj
#     # except:
#     #     pass
#     # finally:
#     #     return None
#
# def __load_python_module_dynamically(module_name, put_in_cache=True):
#     """
#     Loads a Python library module dynamically if it has not been previously loaded
#     :param module_name:
#     :return: Proxy holding the module
#     """
#     if module_name in sys.modules:
#         module_obj = sys.modules[module_name]
#     else:
#         exec ("import {0}".format(module_name))
#         module_obj = eval(module_name)
#
#     module_obj = TypeInferenceProxy(module_obj).clone()
#     if put_in_cache:
#         __put_module_in_sys_cache(module_name, module_obj)
#     return module_obj
#
#
# def __preload_sys_module_cache():
#     """
#     The "sys" Python module holds a cache of stypy-generated module files in order to save time. A Python library
#     module was chosen to hold these data so it can be available through executions and module imports from external
#     files. This function preloads
#     :return:
#     """
#     # Preload sys module
#     sys.stypy_module_cache = {
#         'sys': __load_python_module_dynamically('sys', False)}  # By default, add original sys module clone
#
#     # Preload builtins module
#     sys.stypy_module_cache['__builtin__'] = __load_python_module_dynamically('__builtin__', False)
#     sys.stypy_module_cache['ctypes'] = __load_python_module_dynamically('ctypes', False)
#
#
# def get_module_from_sys_cache(module_name):
#     """
#     Gets a previously loaded module from the sys module cache
#     :param module_name: Module name
#     :return: A Type object or None if there is no such module
#     """
#     try:
#         if hasattr(sys, 'stypy_module_cache'):
#             return sys.stypy_module_cache[module_name]
#         else:
#             __preload_sys_module_cache()
#             return sys.stypy_module_cache[module_name]
#     except:
#         return None
#
# builtin_module = get_module_from_sys_cache('__builtin__')

# t = TypeInferenceProxy.instance(int)

#
# class C:
#     def func(self):
#         try:
#             return self._instance
#         except AttributeError:
#             self._instance = 3
#             return self._instance
#
#
# c = C()
# c.func()

#
# class UndefinedType:
#     pass
#
# class DynamicType:
#     pass
#
#
# class BaseTypeGroup(object):
#     """
#     All type groups inherit from this class
#     """
#     def __str__(self):
#         return self.__repr__()
#
#
# class TypeGroup(BaseTypeGroup):
#     """
#     A TypeGroup is an entity used in the rule files to group several Python types over a named identity. Type groups
#     are collections of types that have something in common, and Python functions and methods usually admits any of them
#     as a parameter when one of them is valid. For example, if a Python library function works with an int as the first
#     parameter, we can also use bool and long as the first parameter without runtime errors. This is for exameple a
#     TypeGroup that will be called Integer
#
#     Not all type groups are defined by collections of Python concrete types. Other groups identify Python objects with
#     a common member or structure (Iterable, Overloads__str__ identify any Python object that is iterable and any Python
#     object that has defined the __str__ method properly) or even class relationships (SubtypeOf type group only matches
#     with classes that are a subtype of the one specified.
#
#     Type groups are the workhorse of the type rule specification mechanism and have a great expressiveness and
#     flexibility to specify admitted types in any Python callable entity.
#
#     Type groups are created in the file type_groups.py
#     """
#     def __init__(self, grouped_types):
#         """
#         Create a new type group that represent the list of types passed as a parameter
#         :param grouped_types: List of types that are included inside this type group
#         :return:
#         """
#         self.grouped_types = grouped_types
#
#     def __contains__(self, type_):
#         """
#         Test if this type group contains the specified type (in operator)
#         :param type_: Type to test
#         :return: bool
#         """
#         # if hasattr(type_, "get_python_type"):
#         #     return type_.get_python_type() in self.grouped_types
#         #
#         # return type_ in self.grouped_types
#         try:
#             return type_.get_python_type() in self.grouped_types
#         except:
#             return type_ in self.grouped_types
#
#     def __eq__(self, type_):
#         """
#         Test if this type group contains the specified type (== operator)
#         :param type_: Type to test
#         :return: bool
#         """
#         # if hasattr(type_, "get_python_type"):
#         #     return type_.get_python_type() in self.grouped_types
#         # return type_ in self.grouped_types
#         try:
#             cond1 = type_.get_python_type() in self.grouped_types
#             cond2 = type_.is_type_instance()
#
#             return cond1 and cond2
#         except:
#             return type_ in self.grouped_types
#
#     def __cmp__(self, type_):
#         """
#         Test if this type group contains the specified type (compatarion operators)
#         :param type_: Type to test
#         :return: bool
#         """
#         # if hasattr(type_, "get_python_type"):
#         #     return type_.get_python_type() in self.grouped_types
#         #
#         # return type_ in self.grouped_types
#         try:
#             # return type_.get_python_type() in self.grouped_types
#             cond1 = type_.get_python_type() in self.grouped_types
#             cond2 = type_.is_type_instance()
#
#             return cond1 and cond2
#         except:
#             return type_ in self.grouped_types
#
#     def __gt__(self, other):
#         """
#         Type group sorting. A type group is less than other type group if contains less types or the types contained
#         in the type group are all contained in the other one. Otherwise, is greater than the other type group.
#         :param other: Another type group
#         :return: bool
#         """
#         if len(self.grouped_types) < len(other.grouped_types):
#             return False
#
#         for type_ in self.grouped_types:
#             if type_ not in other.grouped_types:
#                 return False
#
#         return True
#
#     def __lt__(self, other):
#         """
#         Type group sorting. A type group is less than other type group if contains less types or the types contained
#         in the type group are all contained in the other one. Otherwise, is greater than the other type group.
#         :param other: Another type group
#         :return: bool
#         """
#         if len(self.grouped_types) > len(other.grouped_types):
#             return False
#
#         for type_ in self.grouped_types:
#             if type_ not in other.grouped_types:
#                 return False
#
#         return True
#
#     def __repr__(self):
#         """
#         Textual representation of the type group
#         :return: str
#         """
#         # ret_str = type(self).__name__  + "("
#         # for type_ in self.grouped_types:
#         #     if hasattr(type_, '__name__'):
#         #         ret_str += type_.__name__ + ", "
#         #     else:
#         #         ret_str += str(type_) + ", "
#         #
#         # ret_str = ret_str[:-2]
#         # ret_str+=")"
#
#         ret_str = type(self).__name__
#         return ret_str
#
# class DependentType:
#     """
#     A DependentType is a special base class that indicates that a type group has to be called to obtain the real
#     type it represent. Call is done using the parameters that are trying to match the rule. For example, imagine that
#     we call the + operator with an object that defines the __add__ method and another type to add to. With an object
#     that defines an __add__ method we don't really know what will be the result of calling __add__ over this object
#     with the second parameter, so the __add__ method has to be called (well, in fact, the type inference equivalent
#     version of the __add__ method will be called) to obtain the real return type.
#
#     Dependent types are a powerful mechanism to calculate the return type of operations that depend on calls to
#     certain object members or even to detect incorrect definitions of members in objects (__int__ method defined in
#     object that do not return int, for example).
#     """
#
#     def __init__(self, report_errors=False):
#         """
#         Build a Dependent type instance
#         :param report_errors: Flag to indicate if errors found when calling this type will be reported or not (in that
#         case other code will do it)
#         """
#         self.report_errors = report_errors
#         self.call_arity = 0
#
#     def __call__(self, *call_args, **call_kwargs):
#         """
#         Call the dependent type. Empty in this implementation, concrete calls must be defined in subclasses
#         """
#         pass
#
#
# """
# Type groups with special meaning. All of them define a __eq__ method that indicates if the passed type matches with
# the type group, storing this passed type. They also define a __call__ method that actually perform the type checking
# and calculate the return type. __eq__ and __call__ methods are called sequentially if __eq__ result is True, so the
# storage of the passed type is safe to use in the __call__ as each time an __eq__ is called is replaced. This is the
# way the type rule checking mechanism works: TypeGroups are not meant to be used in other parts of the stypy runtime,
# and if they do, only the __eq__ method should be used to check if a type belongs to a group.
# """
#
#
# class HasMember(TypeGroup, DependentType):
#     """
#         Type of any object that has a member with the specified arity, and that can be called with the corresponding
#         params.
#     """
#
#     def __init__(self, member, expected_return_type, call_arity=0, report_errors=False):
#         DependentType.__init__(self, report_errors)
#         TypeGroup.__init__(self, [])
#         self.member = member
#         self.expected_return_type = expected_return_type
#         self.member_obj = None
#         self.call_arity = call_arity
#
#     def format_arity(self):
#         str_ = "("
#         for i in range(self.call_arity):
#             str_ += "parameter" + str(i) + ", "
#
#         if self.call_arity > 0:
#             str_ = str_[:-2]
#
#         return str_ + ")"
#
#     def __eq__(self, type_):
#         self.member_obj = type_.get_type_of_member(None, self.member)
#         if isinstance(self.member_obj, TypeError):
#             if not self.report_errors:
#                 TypeError.remove_error_msg(self.member_obj)
#             return False
#
#         return True
#
#     def __call__(self, localization, *call_args, **call_kwargs):
#         if callable(self.member_obj.get_python_type()):
#             # Call the member
#             equivalent_type = self.member_obj.invoke(localization, *call_args, **call_kwargs)
#
#             # Call was impossible: Invokation error has to be removed because we provide a general one later
#             if isinstance(equivalent_type, TypeError):
#                 if not self.report_errors:
#                     TypeError.remove_error_msg(equivalent_type)
#                 self.member_obj = None
#                 return False, equivalent_type
#
#             # Call was possible, but the expected return type cannot be predetermined (we have to recheck it later)
#             if isinstance(self.expected_return_type, UndefinedType):
#                 self.member_obj = None
#                 return True, equivalent_type
#
#             # Call was possible, but the expected return type is Any)
#             if self.expected_return_type is DynamicType:
#                 self.member_obj = None
#                 return True, equivalent_type
#
#             # Call was possible, so we check if the predetermined return type is the same that the one that is returned
#             if not issubclass(equivalent_type.get_python_type(), self.expected_return_type):
#                 self.member_obj = None
#                 return False, equivalent_type
#             else:
#                 return True, equivalent_type
#
#         self.member_obj = None
#         return True, None
#
#     def __repr__(self):
#         ret_str = "Instance defining "
#         ret_str += str(self.member)
#         ret_str += self.format_arity()
#         return ret_str
#
#
# class TypeGroups:
#     """
#     Class to hold definitions of type groups that are composed by lists of known Python types
#     """
#
#     def __init__(self):
#         pass
#
#     @staticmethod
#     def get_rule_groups():
#         """
#         Obtain all the types defined in this class
#         """
#
#         def filter_func(element):
#             return isinstance(element, list)
#
#         return filter(lambda member: filter_func(getattr(TypeGroups, member)), TypeGroups.__dict__)
#
#     # Reals
#     RealNumber = [int, long, float, bool]
#
#     # Any number
#     Number = [int, long, float, bool, complex]
#
#     # Integers
#     Integer = [int, long, bool]
#
#     # strings
#     Str = [str, unicode, buffer]
#
#     # Bynary strings
#     ByteSequence = [buffer, bytearray, str, memoryview]
#
#
# class RuleGroupGenerator:
#     """
#     This class is used to generate type group instances from lists of types defined in the TypeGroup object of the
#     type_groups.py file, using the TypeGroup class as a canvas class to generate them.
#     """
#     rule_group_cache = dict()
#
#     def create_rule_group_class(self, class_name):
#         """
#         Creates a new class named as class_name, with all the members of the TypeGroup class
#         :param class_name: Name of the new class
#         :return: A new class, structurally identical to the TypeGroup class. TypeGroup class with the same name can
#         only be created once. If we try to create one that has been already created, the created one is returned instead
#         """
#         if class_name in self.rule_group_cache.keys():
#             return self.rule_group_cache[class_name]
#
#         group_class = type(class_name, TypeGroup.__bases__, dict(TypeGroup.__dict__))
#         instance = group_class(getattr(TypeGroups, class_name))
#         self.rule_group_cache[class_name] = instance
#
#         return instance
#
#     def create_rule_group_class_list(self, classes_name):
#         """
#         Mass-creation of rule group classes calling the previous method
#         :param classes_name: List of class names
#         :return: List of classes
#         """
#         instances = []
#         for class_name in classes_name:
#             instance = self.create_rule_group_class(class_name)
#             instances.append(instance)
#
#         return instances
#
#     def __init__(self):
#         self.rule_group_compliance_dict = dict()
#         for rule in TypeGroups.get_rule_groups():
#             self.rule_group_compliance_dict[rule] = [False] * eval("len(TypeGroups.{0})".format(rule))
#         self.added_types = []
#         self.unclassified_types = []
#
#
# RealNumber = RuleGroupGenerator().create_rule_group_class("RealNumber")
# Number = RuleGroupGenerator().create_rule_group_class("Number")
# Integer = RuleGroupGenerator().create_rule_group_class("Integer")
# Str = RuleGroupGenerator().create_rule_group_class("Str")
# ByteSequence = RuleGroupGenerator().create_rule_group_class("ByteSequence")
