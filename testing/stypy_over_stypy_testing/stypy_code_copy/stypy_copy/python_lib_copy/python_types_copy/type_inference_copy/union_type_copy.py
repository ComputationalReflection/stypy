# -----------
# Union types
# -----------

import copy
import inspect
import types

import stypy_copy.python_lib_copy.python_types_copy.type_introspection_copy.runtime_type_inspection_copy
import undefined_type_copy
from stypy_copy.python_lib_copy.python_types_copy.non_python_type_copy import NonPythonType
from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
from stypy_copy.errors_copy.type_error_copy import TypeError
from stypy_copy.python_lib_copy.python_types_copy import type_inference_copy
from stypy_copy.reporting_copy.print_utils_copy import format_function_name


class UnionType(NonPythonType):
    """
    UnionType is a collection of types that represent the fact that a certain Python element can have any of the listed
    types in a certain point of the execution of the program. UnionTypes are created by the application of the SSA
    algorithm when dealing with branches in the processed program source code.
    """

    @staticmethod
    def _wrap_type(type_):
        """
        Internal method to store Python types in a TypeInferenceProxy if they are not already a TypeInferenceProxy
        :param type_: Any Python object
        :return:
        """
        if not isinstance(type_, Type):
            ret_type = type_inference_copy.type_inference_proxy.TypeInferenceProxy.instance(type_)

            if not ret_type.is_type_instance():
                ret_type.set_type_instance(True)
            return ret_type
        else:
            # At least the Type instance has a value for this property, we set if to true
            if not type_.has_type_instance_value():
                type_.set_type_instance(True)

        return type_

    # ############################### UNION TYPE CREATION ################################

    def __init__(self, type1=None, type2=None):
        """
        Creates a new UnionType, optionally adding the passed parameters. If only a type is passed, this type
        is returned instead
        :param type1: Optional type to add. It can be another union type.
        :param type2: Optional type to add . It cannot be another union type
        :return:
        """
        self.types = []

        # If the first type is a UnionType, add all its types to the newly created union type
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type1):
            for type_ in type1.types:
                self.types.append(type_)
            return

        # Append passed types, if it exist
        if type1 is not None:
            self.types.append(UnionType._wrap_type(type1))

        if type2 is not None:
            self.types.append(UnionType._wrap_type(type2))

    @staticmethod
    def create_union_type_from_types(*types):
        """
        Utility method to create a union type from a list of types
        :param types: List of types
        :return: UnionType
        """
        union_instance = UnionType()

        for type_ in types:
            UnionType.__add_unconditionally(union_instance, type_)

        if len(union_instance.types) == 1:
            return union_instance.types[0]
        return union_instance

    # ############################### ADD TYPES TO THE UNION ################################

    @staticmethod
    def __add_unconditionally(type1, type2):
        """
        Helper method of create_union_type_from_types
        :param type1: Type to add
        :param type2: Type to add
        :return: UnionType
        """
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type1):
            return type1._add(UnionType._wrap_type(type2))
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type2):
            return type2._add(UnionType._wrap_type(type1))

        if type1 == type2:
            return UnionType._wrap_type(type1)

        return UnionType(type1, type2)

    @staticmethod
    def add(type1, type2):
        """
        Adds type1 and type2 to potentially form a UnionType, with the following rules:
        - If either type1 or type2 are None, the other type is returned and no UnionType is formed
        - If either type1 or type2 are UndefinedType, the other type is returned and no UnionType is formed
        - If either type1 or type2 are UnionTypes, they are mergued in a new UnionType that contains the types
        represented by both of them.
        - If both types are the same, the first is returned
        - Else, a new UnionType formed by the two passed types are returned.

        :param type1: Type to add
        :param type2: Type to add
        :return: A UnionType
        """
        if type1 is None:
            return UnionType._wrap_type(type2)

        if type2 is None:
            return UnionType._wrap_type(type1)

        if isinstance(type1, TypeError) and isinstance(type2, TypeError):
            if UnionType._wrap_type(type1) == UnionType._wrap_type(type2):
                return UnionType._wrap_type(type1) # Equal errors are not added
            else:
                type1.error_msg += type2.error_msg
                TypeError.remove_error_msg(type2)
                return type1

        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_undefined_type(type1):
            return UnionType._wrap_type(type2)
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_undefined_type(type2):
            return UnionType._wrap_type(type1)

        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type1):
            return type1._add(type2)
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type2):
            return type2._add(type1)

        if UnionType._wrap_type(type1) == UnionType._wrap_type(type2):
            return UnionType._wrap_type(type1)

        return UnionType(type1, type2)

    def _add(self, other_type):
        """
        Adds the passed type to the current UnionType object. If other_type is a UnionType, all its contained types
        are added to the current.
        :param other_type: Type to add
        :return: The self object
        """
        if other_type is None:
            return self
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(other_type):
            for t in other_type.types:
                self._add(t)
            return self

        other_type = UnionType._wrap_type(other_type)

        # Do the current UnionType contain the passed type, then we do not add it again
        for t in self.types:
            if t == other_type:
                return self

        self.types.append(other_type)

        return self

    # ############################### PYTHON METHODS ################################

    def __repr__(self):
        """
        Visual representation of the UnionType
        :return:
        """
        return self.__str__()

    def __str__(self):
        """
        Visual representation of the UnionType
        :return:
        """
        the_str = ""
        for i in range(len(self.types)):
            the_str += str(self.types[i])
            if i < len(self.types) - 1:
                the_str += " \/ "
        return the_str

    def __iter__(self):
        """
        Iterator interface, to iterate through the contained types
        :return:
        """
        for elem in self.types:
            yield elem

    def __contains__(self, item):
        """
        The in operator, to determine if a type is inside a UnionType
        :param item: Type to test. If it is another UnionType and this passed UnionType types are all inside the
        current one, then the method returns true
        :return: bool
        """
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(item):
            for elem in item:
                if elem not in self.types:
                    return False
            return True
        else:
            if isinstance(item, undefined_type_copy.UndefinedType):
                found = False
                for elem in self.types:
                    if isinstance(elem, undefined_type_copy.UndefinedType):
                        found = True
                return found
            else:
                return item in self.types

    def __eq__(self, other):
        """
        The == operator, to compare UnionTypes

        :param other: Another UnionType (used in type inference code) or a list of types (used in unit testing)
        :return: True if the passed UnionType or list contains exactly the same amount and type of types that the
        passed entities
        """
        if isinstance(other, list):
            type_list = other
        else:
            if isinstance(other, UnionType):
                type_list = other.types
            else:
                return False

        if not len(self.types) == len(type_list):
            return False

        for type_ in self.types:
            if isinstance(type_, TypeError):
                for type_2 in type_list:
                    if type(type_2) is TypeError:
                        continue
            if type_ not in type_list:
                return False

        return True

    def __getitem__(self, item):
        """
        The [] operator, to obtain individual types stored within the union type

        :param item: Indexer
        :return:
        """
        return self.types[item]

    # ############################## MEMBER TYPE GET / SET ###############################

    def get_type_of_member(self, localization, member_name):
        """
        For all the types stored in the union type, obtain the type of the member named member_name, returning a
        Union Type with the union of all the possible types that member_name has inside the UnionType. For example,
        if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and
        Class2.attr: str, this method will return int \/ str.
        :param localization: Caller information
        :param member_name: Name of the member to get
        :return All the types that member_name could have, examining the UnionType stored types
        """
        result = []

        # Add all the results of get_type_of_member for all stored typs in a list
        for type_ in self.types:
            temp = type_.get_type_of_member(localization, member_name)
            result.append(temp)

        # Count errors
        errors = filter(lambda t: isinstance(t, TypeError), result)
        # Count correct invocations
        types_to_return = filter(lambda t: not isinstance(t, TypeError), result)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(result):
            return TypeError(localization, "None of the possible types ('{1}') has the member '{0}'".format(
                member_name, self.types))
        else:
            # If there is an error, it means that the obtained member could be undefined in one of the contained objects
            if len(errors) > 0:
                types_to_return.append(undefined_type_copy.UndefinedType())

            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log, as there are combinations of types
            # that are valid
            # - ErrorTypes are eliminated from the error collection.
            for error in errors:
                error.turn_to_warning()

        # Calculate return type: If there is only one type, return it. If there are several types, return a UnionType
        # with all of them contained
        if len(types_to_return) == 1:
            return types_to_return[0]
        else:
            ret_union = None
            for type_ in types_to_return:
                ret_union = UnionType.add(ret_union, type_)

            return ret_union

    @staticmethod
    def __parse_member_value(destination, member_value):
        """
        When setting a member of a UnionType to a certain value, each one of the contained types are assigned this
        member with the specified value (type). However, certain values have to be carefully handled to provide valid
        values. For example, methods have to be handler in order to provide valid methods to add to each of the
        UnionType types. This helper method convert a method to a valid method belonging to the destination object.

        :param destination: New owner of the method
        :param member_value: Method
        :return THe passed member value, either transformed or not
        """
        if inspect.ismethod(member_value):
            # Each component of the union type has to have its own method reference for model consistency
            met = types.MethodType(member_value.im_func, destination)
            return met

        return member_value

    def set_type_of_member(self, localization, member_name, member_value):
        """
        For all the types stored in the union type, set the type of the member named member_name to the type
        specified in member_value. For example,
        if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and
        Class2.attr: str, this method, if passsed a float as member_value will turn both classes "attr" to float.
        :param localization: Caller information
        :param member_name: Name of the member to set
        :param member_value New type of the member
        :return None or a TypeError if the member cannot be set. Warnings are generated if the member of some of the
        stored objects cannot be set
        """

        errors = []

        for type_ in self.types:
            final_value = self.__parse_member_value(type_, member_value)
            temp = type_.set_type_of_member(localization, member_name, final_value)
            if temp is not None:
                errors.append(temp)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            return TypeError(localization, "None of the possible types ('{1}') can set the member '{0}'".format(
                member_name, self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return None

    # ############################## MEMBER INVOKATION ###############################

    def invoke(self, localization, *args, **kwargs):
        """
        For all the types stored in the union type, invoke them with the provided parameters.
        :param localization: Caller information
        :param args: Arguments of the call
        :param kwargs: Keyword arguments of the call
        :return All the types that the call could return, examining the UnionType stored types
        """
        result = []

        for type_ in self.types:
            # Invoke all types
            temp = type_.invoke(localization, *args, **kwargs)
            result.append(temp)

        # Collect errors
        errors = filter(lambda t: isinstance(t, TypeError), result)

        # Collect returned types
        types_to_return = filter(lambda t: not isinstance(t, TypeError), result)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(result):
            for error in errors:
                TypeError.remove_error_msg(error)
            params = tuple(list(args) + kwargs.values())
            return TypeError(localization, "Cannot invoke {0} with parameters {1}".format(
                format_function_name(self.types[0].name), params))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated from the error collection.
            for error in errors:
                error.turn_to_warning()

        # Return type
        if len(types_to_return) == 1:
            return types_to_return[0]
        else:
            ret_union = None
            for type_ in types_to_return:
                ret_union = UnionType.add(ret_union, type_)

            return ret_union

    # ############################## STRUCTURAL REFLECTION ###############################

    def delete_member(self, localization, member):
        """
        For all the types stored in the union type, delete the member named member_name, returning None or a TypeError
        if no type stored in the UnionType supports member deletion.
        :param localization: Caller information
        :param member: Member to delete
        :return None or TypeError
        """
        errors = []

        for type_ in self.types:
            temp = type_.delete_member(localization, member)
            if temp is not None:
                errors.append(temp)

        # If all types contained in the union fail to delete this member, the whole operation is an error.
        if len(errors) == len(self.types):
            return TypeError(localization, "The member '{0}' cannot be deleted from none of the possible types ('{1}')".
                             format(member, self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return None

    def supports_structural_reflection(self):
        """
        Determines if at least one of the stored types supports structural reflection.
        """
        supports = False

        for type_ in self.types:
            supports = supports or type_.supports_structural_reflection()

        return supports

    def change_type(self, localization, new_type):
        """
        For all the types stored in the union type, change the base type to new_type, returning None or a TypeError
        if no type stored in the UnionType supports a type change.
        :param localization: Caller information
        :param new_type: Type to change to
        :return None or TypeError
        """
        errors = []

        for type_ in self.types:
            temp = type_.change_type(localization, new_type)
            if temp is not None:
                errors.append(temp)

        # If all types contained in the union do not support the operation, the whole operation is an error.
        if len(errors) == len(self.types):
            return TypeError(localization, "None of the possible types ('{1}') can be assigned a new type '{0}'".
                             format(new_type, self.types))
        else:
            # If not all the types return an error when changing types, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return None

    def change_base_types(self, localization, new_types):
        """
        For all the types stored in the union type, change the base types to the ones contained in the list new_types,
        returning None or a TypeError if no type stored in the UnionType supports a supertype change.
        :param localization: Caller information
        :param new_types: Types to change its base type to
        :return None or TypeError
        """
        errors = []

        for type_ in self.types:
            temp = type_.change_base_types(localization, new_types)
            if temp is not None:
                errors.append(temp)

        # Is the whole operation an error?
        if len(errors) == len(self.types):
            return TypeError(localization, "None of the possible types ('{1}') can be assigned new base types '{0}'".
                             format(new_types, self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return None

    def add_base_types(self, localization, new_types):
        """
        For all the types stored in the union type, add to the base types the ones contained in the list new_types,
        returning None or a TypeError if no type stored in the UnionType supports a supertype change.
        :param localization: Caller information
        :param new_types: Types to change its base type to
        :return None or TypeError
        """
        errors = []

        for type_ in self.types:
            temp = type_.change_base_types(localization, new_types)
            if temp is not None:
                errors.append(temp)

        # Is the whole operation an error?
        if len(errors) == len(self.types):
            return TypeError(localization, "The base types of all the possible types ('{0}') cannot be modified".
                             format(self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return None

    # ############################## TYPE CLONING ###############################

    def clone(self):
        """
        Clone the whole UnionType and its contained types
        """
        result_union = self.types[0].clone()
        for i in range(1, len(self.types)):
            if isinstance(self.types[i], Type):
                result_union = UnionType.add(result_union, self.types[i].clone())
            else:
                result_union = UnionType.add(result_union, copy.deepcopy(self.types[i]))

        return result_union

    def can_store_elements(self):
        temp = False
        for type_ in self.types:
            temp |= type_.can_store_elements()

        return temp

    def can_store_keypairs(self):
        temp = False
        for type_ in self.types:
            temp |= type_.can_store_keypairs()

        return temp

    def get_elements_type(self):
        errors = []

        temp = None
        for type_ in self.types:
            res = type_.get_elements_type()
            if isinstance(res, TypeError):
                errors.append(temp)
            else:
                temp = UnionType.add(temp, res)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            return TypeError(None, "None of the possible types ('{1}') can invoke the member '{0}'".format(
                "get_elements_type", self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return temp

    def set_elements_type(self, localization, elements_type, record_annotation=True):
        errors = []

        temp = None
        for type_ in self.types:
            res = type_.set_elements_type(localization, elements_type, record_annotation)
            if isinstance(res, TypeError):
                errors.append(temp)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
                "set_elements_type", self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return temp

    def add_type(self, localization, type_, record_annotation=True):
        errors = []

        temp = None
        for type_ in self.types:
            res = type_.add_type(localization, type_, record_annotation)
            if isinstance(res, TypeError):
                errors.append(temp)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
                "add_type", self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return temp

    def add_types_from_list(self, localization, type_list, record_annotation=True):
        errors = []

        temp = None
        for type_ in self.types:
            res = type_.add_types_from_list(localization, type_list, record_annotation)
            if isinstance(res, TypeError):
                errors.append(temp)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
                "add_types_from_list", self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return temp

    def get_values_from_key(self, localization, key):
        errors = []

        temp = None
        for type_ in self.types:
            res = type_.get_values_from_key(localization, key)
            if isinstance(res, TypeError):
                errors.append(temp)
            else:
                temp = UnionType.add(temp, res)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
                "get_values_from_key", self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return temp

    def add_key_and_value_type(self, localization, type_tuple, record_annotation=True):
        errors = []

        for type_ in self.types:
            temp = type_.add_key_and_value_type(localization, type_tuple, record_annotation)
            if temp is not None:
                errors.append(temp)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            return TypeError(localization, "None of the possible types ('{1}') can invoke the member '{0}'".format(
                "add_key_and_value_type", self.types))
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return None


class OrderedUnionType(UnionType):
    """
    A special type of UnionType that maintain the order of its added types and admits repeated elements. This will be
    used in the future implementation of tuples.
    """

    def __init__(self, type1=None, type2=None):
        UnionType.__init__(self, type1, type2)
        self.ordered_types = []

        if type1 is not None:
            self.ordered_types.append(type1)

        if type2 is not None:
            self.ordered_types.append(type2)

    @staticmethod
    def add(type1, type2):
        if type1 is None:
            return UnionType._wrap_type(type2)

        if type2 is None:
            return UnionType._wrap_type(type1)

        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_undefined_type(type1):
            return UnionType._wrap_type(type2)
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_undefined_type(type2):
            return UnionType._wrap_type(type1)

        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type1):
            return type1._add(type2)
        if stypy_copy.python_lib.python_types.type_introspection.runtime_type_inspection.is_union_type(type2):
            return type2._add(type1)

        if UnionType._wrap_type(type1) == UnionType._wrap_type(type2):
            return UnionType._wrap_type(type1)

        return OrderedUnionType(type1, type2)

    def _add(self, other_type):
        ret = UnionType._add(self, other_type)
        self.ordered_types.append(other_type)
        return ret

    def get_ordered_types(self):
        """
        Obtain the stored types in the same order they were added, including repetitions
        :return:
        """
        return self.ordered_types

    def clone(self):
        """
        Clone the whole OrderedUnionType and its contained types
        """
        result_union = self.types[0].clone()
        for i in range(1, len(self.types)):
            if isinstance(self.types[i], Type):
                result_union = OrderedUnionType.add(result_union, self.types[i].clone())
            else:
                result_union = OrderedUnionType.add(result_union, copy.deepcopy(self.types[i]))

        return result_union
