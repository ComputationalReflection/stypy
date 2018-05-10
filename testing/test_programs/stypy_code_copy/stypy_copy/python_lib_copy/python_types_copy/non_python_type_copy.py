import type_copy
#from stypy_copy.errors_copy.type_error_copy import TypeError


class NonPythonType(type_copy.Type):
    """
    Types store common Python language types. This subclass is used to be the parent of some types used by stypy
    that are not Python types (such as DynamicType), but are needed for modeling some operations. Much of this type
    methods are overriden to return errors if called, as non-python types are not meant to be called on normal
    code execution
    """
    # #################### STORED PYTHON ENTITY (CLASS, METHOD...) AND PYTHON TYPE/INSTANCE OF THE ENTITY ############

    def get_python_entity(self):
        return self

    def get_python_type(self):
        return self

    def get_instance(self):
        return None

    # ############################## MEMBER TYPE GET / SET ###############################

    def get_type_of_member(self, localization, member_name):
        """
        Returns an error if called
        """
        return TypeError(localization, "Cannot get the type of a member over a {0}".format(self.__class__.__name__))

    def set_type_of_member(self, localization, member_name, member_value):
        """
        Returns an error if called
        """
        return TypeError(localization, "Cannot set the type of a member over a {0}".format(self.__class__.__name__))

    # ############################## MEMBER INVOKATION ###############################

    def invoke(self, localization, *args, **kwargs):
        """
        Returns an error if called
        """
        return TypeError(localization, "Cannot invoke a method over a {0}".format(self.__class__.__name__))

    # ############################## STRUCTURAL REFLECTION ###############################

    def delete_member(self, localization, member):
        """
        Returns an error if called
        """
        return TypeError(localization, "Cannot delete a member of a {0}".format(self.__class__.__name__))

    def supports_structural_reflection(self):
        """
        Returns an error if called
        """
        return False

    def change_type(self, localization, new_type):
        """
        Returns an error if called
        """
        return TypeError(localization, "Cannot change the type of a {0}".format(self.__class__.__name__))

    def change_base_types(self, localization, new_types):
        """
        Returns an error if called
        """
        return TypeError(localization, "Cannot change the base types of a {0}".format(self.__class__.__name__))

    def add_base_types(self, localization, new_types):
        """
        Returns an error if called
        """
        self.change_base_types(localization, new_types)

    # ############################## TYPE CLONING ###############################

    def clone(self):
        return self
