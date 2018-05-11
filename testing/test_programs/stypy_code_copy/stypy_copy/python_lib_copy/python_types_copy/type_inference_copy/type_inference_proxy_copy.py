import inspect
import types

import type_inference_proxy_management_copy
from ....errors_copy.type_error_copy import TypeError
from ....errors_copy.type_warning_copy import TypeWarning
from ....python_lib_copy.member_call_copy import call_handlers_copy
from ....python_lib_copy.python_types_copy.type_copy import Type
from ....python_lib_copy.python_types_copy.type_introspection_copy import type_equivalence_copy
from ....python_lib_copy.python_types_copy.type_inference_copy import union_type_copy
from ....python_lib_copy.python_types_copy.type_inference_copy import undefined_type_copy
from ....python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import simple_python_types
from ....type_store_copy.type_annotation_record_copy import TypeAnnotationRecord
from .....stypy_copy import type_store_copy
from .....stypy_copy import stypy_parameters_copy


class TypeInferenceProxy(Type):
    """
    The type inference proxy is the main class of stypy. Its main purpose is to represent any kind of Python type,
     holding a reference to it. It is also responsible of a lot of possible operations that can be done with the
     contained type, including:

     - Returning/setting the type of any member of the Python entity it holds.
     - Invoke any of its associated callable members, returning the invokation result
     - Support structural reflection operations, it the enclosed object is able to support them
     - Obtain relationships with other entities (modules that contain a represented function, class that contains
     a represented method,...)
     - Manipulate stored types (if the represented entity is able to store other types)
     - Clone itself to support the SSA algorithm
     - Respond to builtin operations such as dir and __dict__ calls
     - Hold values for the represented object type

     All Python entities (functions, variables, methods, classes, modules,...) might be enclosed in a type inference
     proxy. For those method that are not applicable to the enclosed Python entity, the class will return a TypeError.
    """

    # Memoization of TypeInferenceProxy instances in those cases in which these instances can be reused. Proxies of
    # any Python entity that is not an instance and do not support structural reflection may be reused.
    type_proxy_cache = dict()

    # Type annotation is an special feature not related to the functionality of this class, but related to this class
    # and the FunctionContext class. Type annotation record an annotation on a special table that holds variable names,
    # types of these variables, and source lines. These data indicate that the variable name has been changed its type
    # to the annotated type in the passed line. This information is used to generate a type annotated source program,
    # which is an optional feature of stypy. This flags control if types are annotated or not when changing the
    # type of a member or other related type-changing operations.
    annotate_types = True

    # ################################ PRIVATE METHODS #################################

    @staticmethod
    def __get_parent_proxy(parent):
        """
        Gets the python entity that can be considered the "parent" of the passed entity
        :param parent: Any Python entity
        :return: The parent of this entity, if any
        """
        if hasattr(parent, '__module__'):
            return TypeInferenceProxy.instance(inspect.getmodule(parent))
        if hasattr(parent, '__class__'):
            return TypeInferenceProxy.instance(parent.__class__)

        return None

    def __assign_parent_proxy(self, parent):
        """
        Changes the parent object of the represented object to the one specified. This is used to trace the nesting
        of proxies that hold types that are placed inside other proxies represented entities. This property is NOT
        related with dynamic inheritance.
        :param parent: The new parent object or None. If the passed parent is None, the class tries to autocalculate it.
        If there is no possible parent, it is assigned to None
        """
        if parent is not None:
            self.parent_proxy = parent
        else:
            if not inspect.ismodule(self.python_entity):
                self.parent_proxy = TypeInferenceProxy.__get_parent_proxy(self.python_entity)
            else:
                self.parent_proxy = None  # Root Python entity

    def __change_class_base_types_checks(self, localization, new_type):
        """
        Performs all the possible checks to see if a base type change is possible for the currently hold python entity.
        This includes:
        - Making sure that the currently hold object represent a class. No base type change is possible if the hold
        entity is not a class. For checking the availability of an instance type change, see the
        "__change_instance_type_checks" private method.
        - Making sure that the hold class is not a new-style class. New style Python classes cannot change its base
        type directly, as its __mro__ (Method Resolution Order) property is readonly. For this purpose a metaclass
        has to be created, like in this example:

        class change_mro_meta(type):
            def __new__(cls, cls_name, cls_bases, cls_dict):
                    out_cls = super(change_mro_meta, cls).__new__(cls, cls_name, cls_bases, cls_dict)
                    out_cls.change_mro = False
                    out_cls.hack_mro   = classmethod(cls.hack_mro)
                    out_cls.fix_mro    = classmethod(cls.fix_mro)
                    out_cls.recalc_mro = classmethod(cls.recalc_mro)
                    return out_cls

            @staticmethod
            def hack_mro(cls):
                cls.change_mro = True
                cls.recalc_mro()

            @staticmethod
            def fix_mro(cls):
                cls.change_mro = False
                cls.recalc_mro()

            @staticmethod
            def recalc_mro(cls):
                # Changing a class' base causes __mro__ recalculation
                cls.__bases__  = cls.__bases__ + tuple()

            def mro(cls):
                default_mro = super(change_mro_meta, cls).mro()
                if hasattr(cls, "change_mro") and cls.change_mro:
                    return default_mro[1:2] + default_mro
                else:
                    return default_mro

        - Making sure that new base class do not belong to a different class style as the current one: base type of
        old-style classes can only be changed to another old-style class.

        :param localization: Call localization data
        :param new_type: New base type to change to
        :return: A Type error specifying the problem encountered with the base type change or None if no error is found
        """
        if not type_inference_proxy_management_copy.is_class(self.python_entity):
            return TypeError(localization, "Cannot change the base type of a non-class Python entity")

        if type_inference_proxy_management_copy.is_new_style_class(self.python_entity):
            return TypeError(localization,
                             "Cannot change the class hierarchy of a new-style class: "
                             "The __mro__ (Method Resolution Order) property is readonly")

        if self.instance is not None:
            return TypeError(localization, "Cannot change the class hierarchy of a class using an instance")

        old_style_existing = type_inference_proxy_management_copy.is_old_style_class(self.python_entity)
        if not isinstance(new_type, TypeError):
            old_style_new = type_inference_proxy_management_copy.is_old_style_class(new_type.python_entity)
        else:
            return TypeError(localization, "Cannot change the class hierarchy to a type error")

        # Did the existing and new class belong to the same class definition type?
        if not old_style_existing == old_style_new:
            return TypeError(localization, "Cannot change the class hierarchy from an old-style Python parent class "
                                           "to a new-style Python parent class")

        return None

    def __change_instance_type_checks(self, localization, new_type):
        """
        Performs all the checks that ensure that changing the type of an instance is possible. This includes:
        - Making sure that we are changing the type of an user-defined class instance. Type change for Python
        library classes instances is not possible.
        - Making sure that the old instance type and the new instance type are of the same class style, as mixing
        old-style and new-style types is not possible in Python.

        :param localization: Call localization data
        :param new_type: New instance type.
        :return:
        """
        # Is a class?
        if not type_inference_proxy_management_copy.is_class(self.python_entity):
            return TypeError(localization, "Cannot change the type of a Python entity that it is not a class")

        # Is the class user-defined?
        if not type_inference_proxy_management_copy.is_user_defined_class(self.instance.__class__):
            return TypeError(localization, "Cannot change the type of an instance of a non user-defined class")

        # Is this object representing a class instance? (so we can change its type)
        if self.instance is None:
            return TypeError(localization, "Cannot change the type of a class object; Type change is only possible"
                                           "with class instances")

        old_style_existing = type_inference_proxy_management_copy.is_old_style_class(self.instance.__class__)
        old_style_new = type_inference_proxy_management_copy.is_old_style_class(new_type.python_entity)

        # Did the existing and new class belong to the same class definition type?
        if not old_style_existing == old_style_new:
            return TypeError(localization, "Cannot change the type of an instances from an old-style Python class to a "
                                           "new-style Python class or viceversa")

        return None

        # ################################ PYTHON METHODS #################################

    def __init__(self, python_entity, name=None, parent=None, instance=None, value=undefined_type_copy.UndefinedType):
        """
        Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This constructor
        should NOT be called directly. Use the instance(...) method instead to take advantage of the implemented
        type memoization of this class.
        :param python_entity: Represented python entity.
        :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property
        value is used instead. Instances have an special name indicating that this entity holds an instance of a class.
        :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.
        :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead
        of representing the class is representing a particular class instance. This is important to properly model
        instance intercession, as altering the structure of single class instances is possible.
        """
        if name is None:
            if hasattr(python_entity, "__name__"):
                self.name = python_entity.__name__
            else:
                if hasattr(python_entity, "__class__"):
                    self.name = python_entity.__class__.__name__
                else:
                    if hasattr(python_entity, "__module__"):
                        self.name = python_entity.__module__

            if instance is not None:
                self.name = "<" + self.name + " instance>"
        else:
            self.name = name

        self.python_entity = python_entity
        self.__assign_parent_proxy(parent)
        self.instance = instance
        if instance is not None:
            self.set_type_instance(True)

        # Attribute values that have not been name (structure of the object is not known)
        self.additional_members = list()

        # If this is a type, store the original variable whose type is
        self.type_of = None

        # Store if the structure of the object is fully known or it has been manipulated without knowing precise
        # attribute values
        self.known_structure = True

        if value is not undefined_type_copy.UndefinedType:
            self.value = value
            self.set_type_instance(True)

            # self.annotation_record = TypeAnnotationRecord.get_instance_for_file(__file__)

            # Instances of "immutable" entities are stored in a cache to save memory. Those include:
            # Python entities that do not support structural reflection, therefore its structure
            # will always be the same. This means that the entity has a dictproxy object as its __dict__ property
            # instead of a plain dict. If the proxy has a non-None instance, it also means that individual instances of
            # this class object are also mutable, and therefore individual instance types are held to allow this. In
            # this case the type proxy cache is also NOT used.
            # TODO: Remove? Temporally disabled because instance identity problems
            # if python_entity in inmutable_python_types:
            #     try:
            #         TypeInferenceProxy.type_proxy_cache[python_entity] = self
            #     except:
            #         pass

    # TODO: Remove?
    # def __get_member_type_repr(self):
    #     repr_str = ""
    #     members = self.dir()
    #     for member in members:
    #         entity = self.get_type_of_member(None, member).get_python_entity()
    #         if hasattr(entity, '__name__'):
    #             type_str = entity.__name__
    #         else:
    #             type_str = str(entity)
    #
    #             repr_str += member + ": " + type_str + "; "
    #
    #     if len(repr_str) > 2:
    #         repr_str = repr_str[:-2]
    #
    #     return repr_str

    def __repr__(self):
        """
        String representation of this proxy and its contents. Python builtin types have a very concise representation.
        The method have been stripped down of much of its information gathering code to favor a more concise and clear
        representation of entities.
        :return: str
        """

        if isinstance(self.python_entity, types.InstanceType) or isinstance(self.python_entity, types.ClassType):
            return self.name

        # Simple Python type only prints its name
        if self.python_entity in simple_python_types:
            return self.get_python_type().__name__

        parent_str = ""
        # TODO: Remove?
        # if self.parent_proxy is None:
        #     parent_str = ""
        # else:
        #     if not (self.parent_proxy.python_entity is None):
        #         if self.parent_proxy.python_entity.__name__ == no_recursion.__name__:
        #             parent_str = ""
        #         else:
        #             parent_str = " from <{0}>".format(self.parent_proxy)
        #     else:
        #         parent_str = " from <{0}>".format(self.parent_proxy)

        str_mark = ""
        # TODO: Remove?
        # if self.supports_structural_reflection():
        #     str_mark = "*"
        # else:
        #     str_mark = ""

        # Representation of instances
        # if not self.instance is None:
        #     instance_type = "Instance of the " + self.instance.__class__.__name__ + " class"
        #     return "{0}{1} {2}".format(instance_type, str_mark, self.__get_member_type_repr()) + parent_str
        #     #return "{0}{1} {2}".format(instance_type, str_mark, self.dir()) + parent_str
        # else:
        #     instance_type = ""

        # Instances of classes
        if self.instance is not None:
            instance_type = self.instance.__class__.__name__ + " instance"
            return instance_type
        else:
            instance_type = ""

        # Representation of lists, tuples, dicts, (types that contain other types)...
        if hasattr(self, self.contained_elements_property_name):
            contained_str = "[" + str(getattr(self, self.contained_elements_property_name)) + "]"
            return "{0}".format(self.get_python_type().__name__) \
                   + contained_str
        else:
            if self.can_store_elements():
                contained_str = "[]"
                return "{0}".format(self.get_python_type().__name__) \
                       + contained_str
            else:
                if self.can_store_keypairs():
                    contained_str = "{}"
                    return "{0}".format(self.get_python_type().__name__) \
                           + contained_str

        own_name = ""
        # TODO: Remove?
        # if inspect.isfunction(self.python_entity):
        #     own_name = ""
        # else:
        #     own_name = self.name

        return "{0}{3}{1}{2}".format(self.get_python_type().__name__, own_name, instance_type, str_mark) + parent_str

        # TODO: Remove?
        # else:
        #     return "{0}{4} {1}{3} from <{2}>".format(self.get_python_type().__name__, self.name, self.parent_proxy,
        #                                                instance_type, str_mark)

    @staticmethod
    def __equal_property_value(property_name, obj1, obj2):
        """
        Determines if a property of two objects have the same value.
        :param property_name: Name of the property to test
        :param obj1: First object
        :param obj2: Second object
        :return: bool (True if same value or both object do not have the property
        """
        if hasattr(obj1, property_name) and hasattr(obj2, property_name):
            if not getattr(obj1, property_name) == getattr(obj2, property_name):
                return False

        return True

    @staticmethod
    def contains_an_undefined_type(value):
        """
        Determines if the passed argument is an UndefinedType or contains an UndefinedType
        :param value: Any Type
        :return: Tuple (bool, int) (contains an undefined type, the value holds n more types)
        """
        if isinstance(value, union_type_copy.UnionType):
            for type_ in value.types:
                if isinstance(type_, undefined_type_copy.UndefinedType):
                    return True, len(value.types) - 1
        else:
            if isinstance(value, undefined_type_copy.UndefinedType):
                return True, 0

        return False, 0

    def __eq__(self, other):
        """
        Type proxy equality. The equality algorithm is represented as follows:
        - Both objects have to be type inference proxies.
        - Both objects have to hold the same type of python entity
        - Both objects held entity name has to be the same (same class, same function, same module, ...), if the
        proxy is not holding an instance
        - If the hold entity do not support structural reflection, comparison will be done using the is operator
        (reference comparison)
        - If not, comparison by structure is performed (same amount of members, same types for these members)

        :param other: The other object to compare with
        :return: bool
        """
        if self is other:
            return True  # Same reference

        if not type(other) is TypeInferenceProxy:
            return False

        # Both do not represent the same Python entity
        if not type(self.python_entity) == type(other.python_entity):
            return False

        # Different values for the "instance" property for both proxies (None vs filled)
        if (self.instance is None) ^ (other.instance is None):
            return False

        # One object is a type name and the other is a type instance (int is not the same as '3')
        self_instantiated = self.is_type_instance()
        other_instantiated = other.is_type_instance()
        if self_instantiated != other_instantiated:
            return False

        self_entity = self.python_entity
        other_entity = other.python_entity

        # Compare several properties key to determine object equality
        for prop_name in Type.special_properties_for_equality:
            if not self.__equal_property_value(prop_name, self_entity, other_entity):
                return False

        # Contains the same elements?
        if not self.__equal_property_value(TypeInferenceProxy.contained_elements_property_name, self_entity,
                                           other_entity):
            return False

        # Class or instance?
        if self.instance is None:
            # Class

            # Both support structural reflection: structure comparison
            if self.supports_structural_reflection() and other.supports_structural_reflection():
                # Compare class structure
                return type_equivalence_copy.structural_equivalence(self_entity, other_entity, True)
            else:
                # No structural reflection: reference comparison
                return type(self_entity) is type(other_entity)
        else:
            # Instance: Compare the class first and the instance later.

            # Both support structural reflection: structure comparison
            if self.supports_structural_reflection() and other.supports_structural_reflection():
                # Compare class structure
                equivalent = type_equivalence_copy.structural_equivalence(self_entity, other_entity, True)

                if not equivalent:
                    return False

                # Compare instance structure
                self_entity = self.instance
                other_entity = other.instance
                return type_equivalence_copy.structural_equivalence(self_entity, other_entity, False)
            else:
                # No structural reflection: reference comparison
                equivalent = type(self.python_entity) is type(other.python_entity)
                if not equivalent:
                    return False

                # No structural reflection: reference comparison
                self_entity = self.instance
                other_entity = other.instance
                return self_entity is other_entity

    # ###################### INSTANCE CREATION ###############

    @staticmethod
    def instance(python_entity, name=None, parent=None, instance=None, value=undefined_type_copy.UndefinedType):
        """
        Creates a new Type inference proxy for the passed python entity (function, module, class, ...). This is the
        preferred way to create proxy instances, as this method implement a memoization optimization.

        :param python_entity: Represented python entity.
        :param name: Name of the represented Python entity. If nothing is provided, the Python entity __name__ property
        value is used instead. Instances have an special name indicating that this entity holds an instance of a class.
        :param parent: Parent proxy object. If nothing is provided, the parent proxy is autocalculated, if possible.
        :param instance: Instance of the represented class. If this proxy holds a class, it is possible that instead
        of representing the class is representing a particular class instance. This is important to properly model
        instance intercession, as altering the structure of single instances is possible.
        """

        # TODO: Remove? Disabled because identity problems
        # try:
        #     if python_entity in TypeInferenceProxy.type_proxy_cache:
        #         return TypeInferenceProxy.type_proxy_cache[python_entity]
        # except:
        #     pass

        if isinstance(python_entity, Type):
            return python_entity

        return TypeInferenceProxy(python_entity, name, parent, instance, value)

    # ################### STORED PYTHON ENTITY (CLASS, METHOD...) AND PYTHON TYPE/INSTANCE OF THE ENTITY ###############

    def get_python_entity(self):
        """
        Returns the Python entity (function, method, class, object, module...) represented by this Type.
        :return: A Python entity
        """
        return self.python_entity

    def get_python_type(self):
        """
        Get the python type of the hold entity. This is equivalent to call the type(hold_python_entity). If a user-
        defined class instance is hold, a types.InstanceType is returned (as Python does)
        :return: A python type
        """
        if not inspect.isclass(self.python_entity):
            return type(self.python_entity)

        if type_inference_proxy_management_copy.is_user_defined_class(self.python_entity) and self.instance is not None:
            return types.InstanceType

        return self.python_entity

    def get_instance(self):
        """
        Gets the stored class instance (if any). Class instances are only stored for instance intercession purposes, as
        we need an entity to store these kind of changes.
        :return:
        """
        return self.instance

    def has_value(self):
        """
        Determines if this proxy holds a value to the type it represents
        :return:
        """
        return hasattr(self, "value")

    def get_value(self):
        """
        Gets the value held by this proxy
        :return: Value of the proxt
        """
        return self.value

    def set_value(self, value):
        """
        Sets the value held by this proxy. No type check is performed
        :return: Value of the proxt
        """
        self.value = value

    # ############################## MEMBER TYPE GET / SET ###############################

    def __get_module_file(self):
        while True:
            current = self.parent_proxy.python_entity
            if current is None:
                return ""
            if isinstance(current, types.ModuleType):
                return current.__file__

    def get_type_of_member(self, localization, member_name):
        """
        Returns the type of the passed member name or a TypeError if the stored entity has no member with the mentioned
        name.
        :param localization: Call localization data
        :param member_name: Member name
        :return: A type proxy with the member type or a TypeError
        """
        try:
            if self.instance is None:
                return TypeInferenceProxy.instance(getattr(self.python_entity, member_name),
                                                   self.name + "." + member_name,
                                                   parent=self)
            else:
                # Copy-on-write attribute values for instances
                if hasattr(self.instance, member_name):
                    return TypeInferenceProxy.instance(getattr(self.instance, member_name),
                                                       self.name + "." + member_name,
                                                       parent=self)
                else:
                    # module_path = self.parent_proxy.python_entity.__file__.replace("__type_inference", "")
                    # module_path = module_path.replace("/type_inference", "")
                    # module_path = module_path.replace('\\', '/')

                    module_path = stypy_parameters_copy.get_original_program_from_type_inference_file(
                        self.__get_module_file())
                    ts = type_store_copy.typestore.TypeStore.get_type_store_of_module(module_path)
                    typ = ts.get_type_of(localization, self.python_entity.__name__)
                    return typ.get_type_of_member(localization, member_name)
                    # return TypeInferenceProxy.instance(getattr(self.python_entity, member_name),
                    #                                    self.name + "." + member_name,
                    #                                    parent=self)
        except AttributeError:
            return TypeError(localization,
                             "{0} has no member '{1}'".format(self.get_python_type().__name__, member_name))

    def set_type_of_member(self, localization, member_name, member_type):
        """
        Set the type of a member of the represented object. If the member do not exist, it is created with the passed
        name and types (except iif the represented object do not support reflection, in that case a TypeError is
        returned)
        :param localization: Caller information
        :param member_name: Name of the member
        :param member_type: Type of the member
        :return: None or a TypeError
        """
        try:
            contains_undefined, more_types_in_value = TypeInferenceProxy.contains_an_undefined_type(member_type)
            if contains_undefined:
                if more_types_in_value == 0:
                    TypeError(localization, "Assigning to {0}.{1} the value of a previously undefined variable".
                              format(self.parent_proxy.name, member_name))
                else:
                    TypeWarning.instance(localization,
                                         "Potentialy assigning to {0}.{1} the value of a previously undefined variable".
                                         format(self.parent_proxy.name, member_name))

            if self.instance is not None:
                # value = self.__parse_member(self.instance, member_name, member_value)
                setattr(self.instance, member_name, member_type)
                if self.annotate_types:
                    self.__annotate_type(localization.line, localization.column, member_name,
                                         member_type)
                return None

            if type_inference_proxy_management_copy.supports_structural_reflection(self.python_entity) or hasattr(
                    self.python_entity, member_name):
                # value = self.__parse_member(self.python_entity, member_name, member_value)
                setattr(self.python_entity, member_name, member_type)
                if self.annotate_types:
                    self.__annotate_type(localization.line, localization.column, member_name,
                                         member_type)
                return None
        except Exception as exc:
            return TypeError(localization,
                             "Cannot modify the structure of '{0}': {1}".format(self.__repr__(), str(exc)))

        return TypeError(localization,
                         "Cannot modify the structure of a python library type or instance")

    # ############################## MEMBER INVOKATION ###############################

    def invoke(self, localization, *args, **kwargs):
        """
        Invoke a callable member of the hold python entity with the specified arguments and keyword arguments.
        NOTE: Calling a class constructor returns a type proxy of an instance of this class. But an instance object
        is only stored if the instances of this class support structural reflection.

        :param localization: Call localization data
        :param args: Arguments of the call
        :param kwargs: Keyword arguments of the call
        :return:
        """

        # Is it callable?
        if not callable(self.python_entity):
            return TypeError(localization, "Cannot invoke on a non callable type")
        else:
            # If it is callable, call it using a call handler
            result_ = call_handlers_copy.perform_call(self, self.python_entity, localization, *args, **kwargs)

            if TypeAnnotationRecord.is_type_changing_method(self.name) and self.annotate_types:
                self.__annotate_type(localization.line, localization.column, self.parent_proxy.name,
                                     self.parent_proxy.get_python_type())

            # If the result is an error, return it
            if isinstance(result_, TypeError):
                return result_

            if isinstance(result_, Type):
                result_.set_type_instance(True)
                return result_

            # If calling a class then we are building an instance of this class. The instance is returned as a
            # consequence of the call, we built the rest of the instance if applicable
            if inspect.isclass(self.python_entity):
                # Instances are stored only to handle object-based structural reflection
                if type_inference_proxy_management_copy.supports_structural_reflection(result_):
                    instance = result_

                    # Calculate the class of the obtained instance
                    result_ = type(result_)
                else:
                    instance = None
            else:
                instance = None

            # If the returned object is not a Python type proxy but a Python type, build it.
            if not isinstance(result_, Type):
                ret = TypeInferenceProxy.instance(result_, instance=instance)
                ret.set_type_instance(True)

                return ret
            else:
                result_.set_type_instance(True)
                return result_

    # ############################## STORED ELEMENTS TYPES (IF ANY) ###############################

    def __check_undefined_stored_value(self, localization, value):
        """
        For represented containers, this method checks if we are trying to store Undefined variables inside them
        :param localization: Caller information
        :param value: Value we are trying to store
        :return:
        """
        contains_undefined, more_types_in_value = TypeInferenceProxy.contains_an_undefined_type(value)
        if contains_undefined:
            if more_types_in_value == 0:
                TypeError(localization, "Storing in '{0}' the value of a previously undefined variable".
                          format(self.name))
            else:
                TypeWarning.instance(localization,
                                     "Potentially storing in '{0}' the value of a previously undefined variable".
                                     format(self.name))
        return contains_undefined, more_types_in_value

    def can_store_elements(self):
        """
        Determines if this proxy represents a Python type able to store elements (lists, tuples, ...)
        :return: bool
        """
        is_iterator = ("dictionary-" in self.name and "iterator" in self.name) or ("iterator" in self.name and
                                                                                   "dict" not in self.name)

        data_structures = [list, set, tuple, types.GeneratorType, bytearray, slice, range, xrange, enumerate, reversed,
                           frozenset]
        return (self.python_entity in data_structures) or is_iterator

    def can_store_keypairs(self):
        """
        Determines if this proxy represents a Python type able to store keypairs (dict, dict iterators)
        :return: bool
        """
        is_iterator = "iterator" in self.name and "dict" in self.name

        return self.python_entity is dict or is_iterator

    def is_empty(self):
        """
        Determines if a proxy able to store elements can be considered empty (no elements were inserted through its
        lifespan
        :return: None or TypeError
        """
        if not self.can_store_elements() and not self.can_store_keypairs():
            return TypeError(None,
                             "STYPY CRITICAL ERROR: Attempt to determine if a container is empty over a python type ({0}) "
                             "that is not able to do it")
        return hasattr(self, self.contained_elements_property_name)

    def get_elements_type(self):
        """
        Obtains the elements stored by this type, returning an error if this is called over a proxy that represent
        a non element holding Python type
        :return: None or TypeError
        """
        if not self.can_store_elements() and not self.can_store_keypairs():
            return TypeError(None,
                             "STYPY CRITICAL ERROR: Attempt to return stored elements over a python type ({0}) "
                             "that is not able to do it")
        if hasattr(self, self.contained_elements_property_name):
            return getattr(self, self.contained_elements_property_name)
        else:
            return undefined_type_copy.UndefinedType()

    def set_elements_type(self, localization, elements_type, record_annotation=True):
        """
        Sets the elements stored by this type, returning an error if this is called over a proxy that represent
        a non element holding Python type. It also checks if we are trying to store an undefined variable.
        :param localization: Caller information
        :param elements_type: New stored elements type
        :param record_annotation: Whether to annotate the type change or not
        :return: The stored elements type
        """
        if not self.can_store_elements() and not self.can_store_keypairs():
            return TypeError(localization,
                             "STYPY CRITICAL ERROR: Attempt to set stored elements types over a python type ({0}) "
                             "that is not able to do it".format(self.get_python_type()))

        contains_undefined, more_types_in_value = TypeInferenceProxy.contains_an_undefined_type(elements_type)
        if contains_undefined:
            if more_types_in_value == 0:
                TypeError(localization, "Storing in '{0}' the value of a previously undefined variable".
                          format(self.name))
            else:
                TypeWarning.instance(localization,
                                     "Potentially storing in '{0}' the value of a previously undefined variable".
                                     format(self.name))

        setattr(self, self.contained_elements_property_name, elements_type)
        if record_annotation and self.annotate_types:
            self.__annotate_type(localization.line, localization.column, "<container elements type>",
                                 getattr(self, self.contained_elements_property_name))

    def add_type(self, localization, type_, record_annotation=True):
        """
        Adds type_ to the elements stored by this type, returning an error if this is called over a proxy that represent
        a non element holding Python type. It also checks if we are trying to store an undefined variable.
        :param localization: Caller information
        :param type_: Type to store
        :param record_annotation: Whether to annotate the type change or not
        :return: None or TypeError
        """
        if not self.can_store_elements():
            return TypeError(localization,
                             "STYPY CRITICAL ERROR: Attempt to store elements over a python type ({0}) that is not"
                             " able to do it".format(self.get_python_type()))

        existing_type = None
        if hasattr(self, self.contained_elements_property_name):
            existing_type = getattr(self, self.contained_elements_property_name)

        value_to_store = union_type_copy.UnionType.add(existing_type, type_)
        self.__check_undefined_stored_value(localization, value_to_store)

        setattr(self, self.contained_elements_property_name, value_to_store)

        if record_annotation and self.annotate_types:
            self.__annotate_type(localization.line, localization.column, "<container elements type>",
                                 getattr(self, self.contained_elements_property_name))

    def add_types_from_list(self, localization, type_list, record_annotation=True):
        """
        Adds the types on type_list to the elements stored by this type, returning an error if this is called over a
        proxy that represent a non element holding Python type. It also checks if we are trying to store an undefined
        variable.
        :param localization: Caller information
        :param type_list: List of types to add
        :param record_annotation: Whether to annotate the type change or not
        :return: None or TypeError
        """
        if not self.can_store_elements():
            return TypeError(localization,
                             "STYPY CRITICAL ERROR: Attempt to store elements over a python type ({0}) that is not"
                             " able to do it".format(self.get_python_type()))

        if hasattr(self, self.contained_elements_property_name):
            existing_type = getattr(self, self.contained_elements_property_name)
            type_list = [existing_type] + type_list

        setattr(self, self.contained_elements_property_name,
                union_type_copy.UnionType.create_union_type_from_types(*type_list))
        if record_annotation and self.annotate_types:
            self.__annotate_type(localization.line, localization.column, "<container elements type>",
                                 getattr(self, self.contained_elements_property_name))

    def __exist_key(self, key):
        """
        Helper method to see if the stored keypairs contains a key equal to the passed one.
        :param key:
        :return:
        """
        existing_type_map = getattr(self, self.contained_elements_property_name)
        keys = existing_type_map.keys()
        for element in keys:
            if key == element:
                return True
        return False

    def add_key_and_value_type(self, localization, type_tuple, record_annotation=True):
        """
        Adds type_tuple to the elements stored by this type, returning an error if this is called over a proxy that
        represent a non keypair holding Python type. It also checks if we are trying to store an undefined variable.
        :param localization: Caller information
        :param type_tuple: Tuple of types to store (key type, value type)
        :param record_annotation: Whether to annotate the type change or not
        :return: None or TypeError
        """
        key = type_tuple[0]
        value = type_tuple[1]

        if not self.can_store_keypairs():
            if not self.can_store_elements():
                return TypeError(localization,
                                 "STYPY CRITICAL ERROR: Attempt to store keypairs over a python type ({0}) that is not"
                                 "a dict".format(self.get_python_type()))
            else:
                if key.get_python_type() is not int:
                    return TypeError(localization,
                                     "STYPY CRITICAL ERROR: Attempt to store keypairs on a python collection")
                else:
                    self.add_type(localization, value, record_annotation)
                    return

        if not hasattr(self, self.contained_elements_property_name):
            setattr(self, self.contained_elements_property_name, dict())

        existing_type_map = getattr(self, self.contained_elements_property_name)

        self.__check_undefined_stored_value(localization, value)

        # if key in existing_type_map.keys():
        if self.__exist_key(key):
            # We cannot directly use the dictionary because type inference proxies are not hashable, but are comparable
            stored_key_index = existing_type_map.keys().index(key)
            stored_key = existing_type_map.keys()[stored_key_index]
            existing_type = existing_type_map[stored_key]
            existing_type_map[stored_key] = union_type_copy.UnionType.add(existing_type, value)
        else:
            existing_type_map[key] = value

        if record_annotation and self.annotate_types:
            self.__annotate_type(localization.line, localization.column, "<dictionary elements type>",
                                 getattr(self, self.contained_elements_property_name))

    def get_values_from_key(self, localization, key):
        """
        Get the poosible values associated to a key type on a keypair storing proxy

        :param localization: Caller information
        :param key: Key type
        :return: Value type list
        """
        existing_type_map = getattr(self, self.contained_elements_property_name)

        try:
            # We cannot directly use the dictionary because type inference proxies are not hashable, but are comparable
            stored_key_index = existing_type_map.keys().index(key)
            stored_key = existing_type_map.keys()[stored_key_index]
            value = existing_type_map[stored_key]
            return value
        except:
            return TypeError(localization, "No value is associated to key type '{0}'".format(key))

    # ############################## STRUCTURAL REFLECTION ###############################

    def supports_structural_reflection(self):
        """
        Determines whether the stored python entity supports intercession. This means that this proxy stores an
        instance (which are created precisely for this purpose) or the stored entity has a dict as the type of
        its __dict__ property (and not a dictproxy instance, that is read-only).

        :return: bool
        """
        return self.instance is not None or type_inference_proxy_management_copy.supports_structural_reflection(
            self.python_entity)

    def delete_member(self, localization, member_name):
        """
        Set the type of the member whose name is passed to the specified value. There are cases in which deepcopies of
        the stored python entities are not supported when cloning the type proxy (cloning is needed for SSA), but
        structural reflection is supported. Therefore, the additional_members attribute have to be created to still
        support structural reflection while maintaining the ability to create fully independent clones of the stored
        python entity.

        :param localization: Call localization data
        :param member_name: Member name
        :return:
        """
        try:
            if self.instance is not None:
                # value = self.__parse_member(self.instance, member_name, member_value)
                delattr(self.instance, member_name)
                return None

            if type_inference_proxy_management_copy.supports_structural_reflection(self.python_entity):
                # value = self.__parse_member(self.python_entity, member_name, member_value)
                delattr(self.python_entity, member_name)
                return None
        except Exception as exc:
            return TypeError(localization,
                             "'{2}' member deletion is impossible: Cannot modify the structure of '{0}': {1}".
                             format(self.__repr__(), str(exc), member_name))

        return TypeError(localization,
                         "'{0}' member deletion is impossible: Cannot modify the structure of a python library "
                         "type or instance".format(member_name))

    def change_type(self, localization, new_type):
        """
        Changes the type of the stored entity, provided it is an instance (so it supports structural reflection).
        Type change is only available in Python for instances of user-defined classes.

        You can only assign to the __class__ attribute of an instance of a user-defined class
        (i.e. defined using the class keyword), and the new value must also be a user-defined class.
        Whether the classes are new-style or old-style does not matter. (You can't mix them, though.
        You can't turn an old-style class instance into a new-style class instance.)

        :param localization: Call localization data
        :param new_type: New type of the instance.
        :return: A TypeError or None
        """
        result = self.__change_instance_type_checks(localization, new_type)

        if isinstance(result, TypeError):
            return result

        # If all these tests are passed, change the class:
        if type_inference_proxy_management_copy.is_user_defined_class(new_type.python_entity):
            self.python_entity = types.InstanceType
        else:
            self.python_entity = new_type.python_entity

        setattr(self.instance, '__class__', new_type.python_entity)
        return None

    def change_base_types(self, localization, new_types):
        """
        Changes, if possible, the base types of the hold Python class. For determining if the change is possible, a
        series of checks (defined before) are made.

        For new-style classes, changing of the mro is not possible, you need to define a metaclass that does the trick

        Old-style classes admits changing its __bases__ attribute (its a tuple), so we can add or substitute

        :param localization: Call localization data
        :param new_types: New base types (in the form of a tuple)
        :return: A TypeError or None
        """
        if not type(new_types) is tuple:
            return TypeError(localization, "New subtypes have to be specified using a tuple")

        for base_type in new_types:
            check = self.__change_class_base_types_checks(localization, base_type)
            if isinstance(check, TypeError):
                return check

        base_classes = map(lambda tproxy: tproxy.python_entity, new_types)

        self.python_entity.__bases__ = tuple(base_classes)
        return None

    def add_base_types(self, localization, new_types):
        """
        Adds, if possible, the base types of the hold Python class existing base types.
        For determining if the change is possible, a series of checks (defined before) are made.

        :param localization: Call localization data
        :param new_types: New base types (in the form of a tuple)
        :return: A TypeError or None
        """
        if not type(new_types) is tuple:
            return TypeError(localization, "New subtypes have to be specified using a tuple")

        for base_type in new_types:
            check = self.__change_class_base_types_checks(localization, base_type)
            if isinstance(check, TypeError):
                return check

        base_classes = map(lambda tproxy: tproxy.python_entity, new_types)
        self.python_entity.__bases__ += tuple(base_classes)
        return None

    # ############################## TYPE CLONING ###############################

    def clone(self):
        """
        Clones the type proxy, making an independent copy of the stored python entity. Physical cloning is not
        performed if the hold python entity do not support intercession, as its structure is immutable.

        :return: A clone of this proxy
        """
        if not self.supports_structural_reflection() and not self.can_store_elements() and \
                not self.can_store_keypairs():
            return self
        else:
            return type_inference_proxy_management_copy.create_duplicate(self)

    # ############################## TYPE INSPECTION ###############################

    def dir(self):
        """
        Calls the dir Python builtin over the stored Python object and returns the result
        :return: list of strings
        """
        return dir(self.python_entity)

    def dict(self, localization):
        """
        Equivalent to call __dict__ over the stored Python instance
        :param localization:
        :return:
        """
        members = self.dir()
        ret_dict = TypeInferenceProxy.instance(dict)
        ret_dict.set_type_instance(True)
        for member in members:
            str_instance = TypeInferenceProxy.instance(str, value=member)

            value = self.get_type_of_member(localization, member)
            ret_dict.add_key_and_value_type(localization, (str_instance, value), False)

        return ret_dict

    def is_user_defined_class(self):
        """
        Determines whether this proxy holds an user-defined class or not
        :return:
        """
        return type_inference_proxy_management_copy.is_user_defined_class(self.python_entity)

    # ############################## TYPE ANNOTATION ###############################

    def __annotate_type(self, line, column, name, type_):
        """
        Annotate a type into the proxy type annotation record
        :param line: Source code line when the type change is performed
        :param column: Source code column when the type change is performed
        :param name: Name of the variable whose type is changed
        :param type_: New type
        :return: None
        """
        if hasattr(self, "annotation_record"):
            self.annotation_record.annotate_type(line, column, name, type_)
