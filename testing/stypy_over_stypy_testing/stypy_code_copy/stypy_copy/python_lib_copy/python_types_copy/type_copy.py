import abc


class Type:
    """
    Abstract class that represent the various types we manage in the type-inference equivalent program. It is the
    base type of several key classes in stypy:

    - TypeInferenceProxy: That enables us to infer the type of any member of any Python entity that it represents
    - TypeError: A special type to model errors found when type checking program.
    - UnionType: A special type to model the union of various possible types, obtained as a result of applying the SSA
    algorithm.

    So therefore the basic purpose of the Type subclasses is to represent (and therefore, store) Python types. Methods
    of this class are created to work with these represented types.
    """
    # This is an abstract class
    __metaclass__ = abc.ABCMeta

    # Some Type derived classes may contain elements. This is the property that is used to store these elements.
    contained_elements_property_name = "contained_elements_type"

    # Equality between types is a very frequently used operation. This list of properties are key to determine
    # equality or inequality on most objects. Therefore these properties are the first to be looked upon to determine
    # equality of Types and its subclasses, in order to save performance.
    special_properties_for_equality = [
        "__name__", "im_class", "im_self", "__module__", "__objclass__"
    ]

    def __str__(self):
        """
        str representation of the class
        :return:
        """
        return self.__repr__()

    # ################## STORED PYTHON ENTITY (CLASS, METHOD...) AND PYTHON TYPE/INSTANCE OF THE ENTITY ###############

    @abc.abstractmethod
    def get_python_entity(self):
        """
        Returns the Python entity (function, method, class, object, module...) represented by this Type.
        :return: A Python entity
        """
        return

    @abc.abstractmethod
    def get_python_type(self):
        """
        Returns the Python type of the Python entity (function, method, class, object, module...) represented by this
        Type. It is almost equal to get_python_entity except for class instances. The Python type for any class instance
        is types.InstanceType, not the type of the class.
        :return: A Python type
        """
        return

    @abc.abstractmethod
    def get_instance(self):
        """
        If this Type represent an instance of a class, return this instance.
        :return:
        """
        return

    def has_type_instance_value(self):
        """
        Returns if this Type has a value for the "type_instance" property
        :return: bool
        """
        return hasattr(self, "type_instance")

    def is_type_instance(self):
        """
        For the Python type represented by this object, this method is used to distinguish between a type name and a
        the type of the element represented by this type. For example, if this element represent the type of the
        value '3' (int) is_type_instance returns true. If, however, this element represents the type 'int', the method
         returns False. It also returns false for types that do not have instances (functions, modules...)
        :return:
        """
        if not hasattr(self, "type_instance"):
            return False
        return self.type_instance

    def set_type_instance(self, value):
        """
        Change the type instance value
        :param value:
        :return:
        """
        self.type_instance = value

    # ############################## MEMBER TYPE GET / SET ###############################

    @abc.abstractmethod
    def get_type_of_member(self, localization, member_name):
        """
        Gets the type of a member of the stored type
        :param localization: Caller information
        :param member_name: Name of the member
        :return: Type of the member
        """
        return

    @abc.abstractmethod
    def set_type_of_member(self, localization, member_name, member_type):
        """
        Set the type of a member of the represented object. If the member do not exist, it is created with the passed
        name and types (except iif the represented object do not support reflection, in that case a TypeError is
        returned)
        :param localization: Caller information
        :param member_name: Name of the member
        :param member_type: Type of the member
        :return: None
        """
        return

    # ############################## MEMBER INVOKATION ###############################

    @abc.abstractmethod
    def invoke(self, localization, *args, **kwargs):
        """
        Invoke the represented object if this is a callable one (function/method/lambda function/class (instance
        construction is modelled by invoking the class with appropriate constructor parameters).
        :param localization: Caller information
        :param args: Arguments of the call
        :param kwargs: Keyword arguments of the call
        :return: The type that the performed call returned
        """
        return

    # ############################## STRUCTURAL REFLECTION ###############################

    @abc.abstractmethod
    def delete_member(self, localization, member):
        """
        Removes a member by its name, provided the represented object support structural reflection
        :param localization: Caller information
        :param member: Name of the member to delete
        :return: None
        """
        return

    @abc.abstractmethod
    def supports_structural_reflection(self):
        """
        Checks whether the represented object support structural reflection or not
        :return: bool
        """
        return

    @abc.abstractmethod
    def change_type(self, localization, new_type):
        """
        Changes the type of the represented object to new_type, should the represented type support structural
        reflection
        :param localization: Caller information
        :param new_type: Type to change the object to
        :return: None
        """
        return

    @abc.abstractmethod
    def change_base_types(self, localization, new_types):
        """
        Changes the supertype of the represented object to the ones in new_types, should the represented type support
        structural reflection
        :param localization: Caller information
        :param new_types: Types to assign as new supertypes of the object
        :return: None
        """
        return

    @abc.abstractmethod
    def add_base_types(self, localization, new_types):
        """
        Adds to the supertypes of the represented object the ones in new_types, should the represented type support
        structural reflection
        :param localization: Caller information
        :param new_types: Types to add to the supertypes of the object
        :return: None
        """
        return

    # ############################## TYPE CLONING ###############################

    @abc.abstractmethod
    def clone(self):
        """
        Make a deep copy of the represented object
        :return:
        """
        return
