#!/usr/bin/env python
# -*- coding: utf-8 -*-

import inspect
import types

from stypy import invokation
from stypy import type_inference_programs
from stypy import types as stypy_types
from stypy.errors.type_error import StypyTypeError
from stypy.reporting.localization import Localization
from stypy.reporting.output_formatting import format_function_name
from stypy.types.undefined_type import UndefinedType
from type_wrapper import TypeWrapper


class UnionType(TypeWrapper):
    """
    A Union type manages those cases in which a certain variable may have any type within a list of possible types.
    For example x: int U str means that, depending on the program execution flow, x may have int or str type. This is
    an integral part of the SSA algorithm implemented in stypy, as SSA branches typically produces union types.
    """
    declared_members = None

    def __init__(self, type1=None, type2=None, base_class=None):
        """
        Initializes a union type with the types type1 and type2
        :param type1:
        :param type2:
        :param base_class: This parameter is not used in normal union type construction. It is only initialized to a
        value if a class has a union type as its parent. In this case, union type construction is different, as builds
        the type with the passed contents.
        """

        # Normal union type construction
        if base_class is None:
            self.types = list()
            self.wrapped_type = None
            self.__add_type(type1)
            self.__add_type(type2)
        else:
            # Building a class contents when this object is its parent
            self.types = type2[0].types
            for name, type_ in base_class.items():
                if inspect.isfunction(type_) and name is not "__init__":
                    method_ = types.MethodType(type_, self)
                    exec ("self." + name + " = method_")

    def can_store_elements(self):
        """
        Determines if any of the types in the union type can store elements
        :return:
        """
        for t in self.types:
            if stypy_types.type_containers.can_store_elements(t):
                return True

    def can_store_keypairs(self):
        """
        Determines if any of the types in the union type can store key pairs
        :return:
        """
        for t in self.types:
            if stypy_types.type_containers.can_store_keypairs(t):
                return True

    def get_contained_type(self):
        """
        Get the contained types of all the types in the union
        :return:
        """
        temp = None
        for t in self.types:
            temp = UnionType.add(temp, stypy_types.type_containers.get_contained_elements_type(t))

        return temp

    def get_contained_type_for_key(self, key):
        """
        Get the contained types of all the types in the union that belong to a certain key
        :return:
        """
        temp = None
        for t in self.types:
            temp = UnionType.add(temp, stypy_types.type_containers.get_contained_elements_type_for_key(t, key))

        return temp

    def set_contained_type(self, type_):
        """
        Sets the contained types of all the types in the union
        :return:
        """
        errors = []
        for t in self.types:
            ret = stypy_types.type_containers.set_contained_elements_type(t, type_)
            if isinstance(ret, StypyTypeError):
                errors.append(ret)

        if len(errors) == len(self.types):
            for e in errors:
                StypyTypeError.remove_error_msg(e)
            return StypyTypeError(Localization.get_current(),
                                  "None of the possible types is able to store elements of type {0}".format(type_))
        else:
            for e in errors:
                e.turn_to_warning()

    def set_contained_type_for_key(self, key, type_):
        """
        Sets the contained types of all the types in the union that belong to a certain key
        :return:
        """
        errors = []
        for t in self.types:
            ret = stypy_types.type_containers.set_contained_elements_type_for_key(t, key, type_)
            if isinstance(ret, StypyTypeError):
                errors.append(ret)

        if len(errors) == len(self.types):
            for e in errors:
                StypyTypeError.remove_error_msg(e)
            return StypyTypeError(Localization.get_current(),
                                  "None of the possible types is able to store "
                                  "key pairs of type ({0}, {1})".format(key, type_))
        else:
            for e in errors:
                e.turn_to_warning()

    def del_contained_type(self, type_):
        """
        Delete the contained type type_ of all the types in the union
        :return:
        """
        errors = []
        for t in self.types:
            ret = stypy_types.type_containers.del_contained_elements_type(t, type_)
            if isinstance(ret, StypyTypeError):
                errors.append(ret)

        if len(errors) == len(self.types):
            for e in errors:
                StypyTypeError.remove_error_msg(e)
            return StypyTypeError(Localization.get_current(),
                                  "None of the possible types is able to delete elements of type {0}".format(type_))
        else:
            for e in errors:
                e.turn_to_warning()

    def append_type(self, type_):
        """
        Adds a type to the union type
        :param type_:
        :return:
        """
        self.types.append(type_)


    @staticmethod
    def __is_same_base_type(a, b):
        base_types = [int, long, float, str]

        for t in base_types:
            if type(a) is t and type(b) is t:
                return True

        return False

    @staticmethod
    def __is_operator(element, list_):
        """
        Executes the is operator with all the types in the union, passed as a list
        :param element:
        :param list_:
        :return:
        """
        for elem in list_:
            if type(element) is invokation.type_rules.type_groups.type_groups.DynamicType and type(elem) is \
                    invokation.type_rules.type_groups.type_groups.DynamicType:
                return True
            if element is elem:
                return True

            if UnionType.__is_same_base_type(element, elem):
                return True

            if isinstance(element, TypeWrapper) and isinstance(elem, TypeWrapper):
                if element == elem:
                    return True
                else:
                    # Tuples with the same types are considered equal
                    if isinstance(elem.wrapped_type, tuple) and isinstance(element.wrapped_type, tuple):
                        if UnionType.compare_tuple_contents(elem, element):
                            return True

            from stypy.invokation.handlers import call_utilities
            if call_utilities.is_numpy_array(element) and call_utilities.is_numpy_array(elem):
                if type(element.contained_types) == type(elem.contained_types):
                    return True
            # if type(element).__name__ == 'ndarray' and type(elem).__name__ == 'ndarray':
            #     return element.tolist() == elem.tolist()

        return False

    def __add_type(self, type_to_add):
        """
        Adds a type to the union type
        :param type_to_add:
        :return:
        """

        # If the type to add is also a union type, append these types that are not present in the current one
        if is_union_type(type_to_add):
            for type_ in type_to_add.types:
                if not self.__is_operator(type_, self.types):
                    self.append_type(type_)
            return self

        # Append passed types, if it don't exist yet
        if type_to_add is not None:
            if not self.__is_operator(type_to_add, self.types):
                self.append_type(type_to_add)

        return self

    @staticmethod
    def create_from_type_list(type_list):
        """
        Creates a union type that holds all the types in the list. No checking is done for duplicate types
        :param type_list:
        :return:
        """
        union = UnionType()
        union.types = list(type_list)
        return union

    @staticmethod
    def isinstance(type1, type2):
        """
        Determines if type1 is an instance of type2
        :param type1:
        :param type2:
        :return:
        """
        return object.__getattribute__(type1, "__class__") == type2

    @staticmethod
    def __errors_in_parameters(param1, param2):
        """
        Determines if either param1 or param2 are type errors
        :param param1:
        :param param2:
        :return:
        """
        return stypy_types.type_inspection.is_error(param1) or stypy_types.type_inspection.is_error(param2)

    @staticmethod
    def __add_error(type1, type2):
        """
        Deals with the special behavior of the add operation when dealing with
        errors. If joining an error and a non-error type, the error is not added to the union and is turned into a
        warning. This means that at least one of the types of the variable is valid and therefore we continue operating
        with it. If both types are errors, these are merged into a single one.
        :param type1: Type to add
        :param type2: Type to add
        :return: A UnionType
        """
        if stypy_types.type_inspection.is_error(type1) and stypy_types.type_inspection.is_error(type2):
            if type1 == type2:
                return type1  # Equal errors are not added
            else:
                type1.error_msg += "\n" + type2.error_msg
                StypyTypeError.remove_error_msg(type2)
                return type1

        if stypy_types.type_inspection.is_error(type1):
            type1.turn_to_warning()
            return UnionType.add(None, type2)
        if stypy_types.type_inspection.is_error(type2):
            type2.turn_to_warning()
            return UnionType.add(type1, None)

        return UnionType.add(type1, type2)

    @staticmethod
    def tuple_contains_type(t, type):
        for e in t:
            if not UnionType.__is_same_base_type(e, type):
                return False

        return True

    @staticmethod
    def compare_tuple_contents(t1, t2):
        has_contained1 = hasattr(t1, 'contained_types')
        has_contained2 = hasattr(t2, 'contained_types')

        if has_contained1 != has_contained2:
            return False

        # Both are empty tuples
        if not has_contained1 and not has_contained2:
            return True

        if has_contained1 and has_contained2:
            if t1.contained_types is t2.contained_types:
                return True

            if isinstance(t1.contained_types, TypeWrapper) and isinstance(t2.contained_types, TypeWrapper):
                try:
                    if len(t1.contained_types) != len(t2.contained_types):
                        return False

                    for t in t1.contained_types:
                        # If a type in the first tuple is not contained in the second one, they are not considered equal
                        if not UnionType.tuple_contains_type(t2.contained_types, t):
                            return False
                    return True
                except:
                    pass

        return False


    @staticmethod
    def add(type1, type2):
        """
        Adds two types, potentially creating a new union type if they are different and not union types. If both are
        union types, they are mergued deleting duplicates. If only one is a union type, the other is added to its
        contained types if not already present.
        :param type1:
        :param type2:
        :return:
        """
        if stypy_types.type_inspection.is_str(type(type1)):
            # Nullify possible values of strings when added to the union type to prevent multiple strings in the union
            type1 = type(type1)()
        if stypy_types.type_inspection.is_str(type(type2)):
            # Nullify possible values of strings when added to the union type to prevent multiple strings in the union
            type2 = type(type2)()

        if type1 is None or stypy_types.type_inspection.is_recursive_call_result(type1):
            return type2

        if type2 is None or stypy_types.type_inspection.is_recursive_call_result(type2):
            return type1

        # If one of the types is an error (or both) a special treatment is executed
        if UnionType.__errors_in_parameters(type1, type2):
            return UnionType.__add_error(type1, type2)

        if is_union_type(type1):
            return type1.__add_type(type2)
        if is_union_type(type2):
            return type2.__add_type(type1)

        # Numbers have problems when using the is operator: Same values are reported as different.
        if invokation.type_rules.type_groups.type_group_generator.Number == type(type1):
            if type(type1) is type(type2):
                return type1

        if type1 is type2:
            return type1

        if isinstance(type1, TypeWrapper) and isinstance(type2, TypeWrapper):
            if type1 == type2:
                return type1
            else:
                # Tuples with the same types are considered equal
                if isinstance(type1.wrapped_type, tuple) and isinstance(type2.wrapped_type, tuple):
                    if UnionType.compare_tuple_contents(type1, type2):
                        return type1

        # Numpy ndarrays have a different treatment
        if type(type1).__name__ == 'ndarray' and type(type2).__name__ == 'ndarray':
            if type1.tolist() == type2.tolist():
                return type1

        if UnionType.can_be_mergued(type1, type2):
            mergue = UnionType.mergue(type1, type2)
            if mergue is not None:
                return mergue

        return UnionType(type1, type2)

    def get_types(self):
        """
        Gets the types of the union
        :return:
        """
        return self.types

    def has_member(self, name):
        """
        Determines if any of the types in the union has the passed member name
        :param name:
        :return:
        """
        for type_ in self.types:
            type_ = stypy_types.type_intercession.get_member(Localization.get_current(), type_, name)
            if not isinstance(type_, StypyTypeError):
                return True

        return False

    def get_type_of_member(self, name):
        """
        Get the type of the member name of the types of the union (if different types has the same member name with
        different types, a union type with all the possible types is returned).
        :param name:
        :return:
        """
        types_to_return = []
        errors = []

        # Add all the results of get_type_of_member for all stored types in a list
        for type_ in self.types:
            type_ = stypy_types.type_intercession.get_member(Localization.get_current(), type_, name)
            if isinstance(type_, StypyTypeError):
                errors.append(type_)
            else:
                types_to_return.append(type_)

        all_errors = len(errors) == len(self.types)

        if all_errors:
            return StypyTypeError.no_type_has_member_error(Localization.get_current(), self.types, name)
        else:
            # If there is an error, it means that the obtained member could be undefined in one of the contained
            # objects
            # if len(errors) > 0:
            #     types_to_return.append(UndefinedType)

            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log, as there are combinations of types
            # that are valid
            # - ErrorTypes are eliminated from the error collection.
            for error in errors:
                error.turn_to_warning()

        # Calculate return type: If there is only one type, return it. If there are several types, return a
        # UnionType with all of them contained
        if len(types_to_return) == 1:
            return types_to_return[0]
        else:
            ret_union = None
            for type_ in types_to_return:
                ret_union = UnionType.add(ret_union, type_)

            return ret_union

    def __getattribute__(self, name):
        """
            For all the types stored in the union type, obtain the type of the member named member_name, returning a
            Union Type with the union of all the possible types that member_name has inside the UnionType. For example,
            if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and
            Class2.attr: str, this method will return int \/ str.
            :param name: Name of the member to get
            :return All the types that member_name could have, examining the UnionType stored types
        """
        if name in object.__getattribute__(self, "declared_members"):
            return object.__getattribute__(self, name)
        else:
            return self.get_type_of_member(name)

    def set_type_of_member(self, name, value):
        """
        Set the type of the member name of all the types in the union that declare it to value.
        :param name:
        :param value:
        :return:
        """
        errors = []

        for type_ in self.types:
            type_ = stypy_types.type_intercession.set_member(Localization.get_current(), type_, name, value)
            if isinstance(type_, StypyTypeError):
                errors.append(type_)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            for error in errors:
                StypyTypeError.remove_error_msg(error)

            return StypyTypeError.no_type_can_set_member_error(Localization.get_current(), self.types, name)
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return None

    def __setattr__(self, name, value):
        """
            For all the types stored in the union type, set the type of the member named member_name to the type
            specified in member_value. For example,
            if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and
            Class2.attr: str, this method, if passsed a float as member_value will turn both classes "attr" to float.
            :param name: Name of the member to set
            :param value New type of the member
            :return None or a TypeError if the member cannot be set. Warnings are generated if the member of some of the
            stored objects cannot be set
        """
        # Class is writable over the contained members
        if name in object.__getattribute__(self, "declared_members") and not name == "__class__":
            try:
                object.__setattr__(self, name, value)
            except Exception as exc:
                return StypyTypeError.member_cannot_be_set_error(Localization.get_current(), self, name, value, exc)
        else:
            return self.set_type_of_member(name, value)

    def __delattr__(self, name):
        """
            For all the types stored in the union type, set the type of the member named member_name to the type
            specified in member_value. For example,
            if a UnionType has the types Class1 and Class2, both with the member "attr" so Class1.attr: int and
            Class2.attr: str, this method, if passsed a float as member_value will turn both classes "attr" to float.
            :param name: Name of the member to set
            :return None or a TypeError if the member cannot be set. Warnings are generated if the member of some of the
            stored objects cannot be set
        """
        # Class is writable over the contained members
        if name in object.__getattribute__(self, "declared_members") and not name == "__class__":
            try:
                object.__delattr__(self, name)
            except Exception as exc:
                return StypyTypeError.member_cannot_be_deleted_error(Localization.get_current(), self, name, exc)
        else:
            return self.del_member(name)

    def del_member(self, name):
        """
        Deletes the member name from all the types in the union that declare it
        :param name:
        :return:
        """
        errors = []

        for type_ in self.types:
            type_ = stypy_types.type_intercession.del_member(Localization.get_current(), type_, name)
            if isinstance(type_, StypyTypeError):
                errors.append(type_)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(self.types):
            for error in errors:
                StypyTypeError.remove_error_msg(error)

            return StypyTypeError.no_type_can_delete_member_error(Localization.get_current(), self.types, name)
        else:
            # If not all the types return an error when accessing the members, the policy is different:
            # - Notified errors are turned to warnings in the general error log
            # - ErrorTypes are eliminated.
            for error in errors:
                error.turn_to_warning()

        return None

    def is_declared_member(self, name):
        """
        Checks if a member name is declared in this class
        :param name:
        :return:
        """
        return name in self.declared_members

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
            if type_ is UndefinedType:
                result.append(StypyTypeError(localization.get_current(), "Trying to perform a call over an undefined type"))
                continue
            # Invoke all types
            temp = type_inference_programs.stypy_interface.invoke(localization, type_, *args, **kwargs)
            result.append(temp)

        # Collect errors
        errors = filter(lambda t: isinstance(t, StypyTypeError), result)

        # Collect returned types
        types_to_return = filter(lambda t: not isinstance(t, StypyTypeError), result)

        # If all types contained in the union do not have this member, the whole access is an error.
        if len(errors) == len(result):
            for error in errors:
                StypyTypeError.remove_error_msg(error)
            params = tuple(list(args) + kwargs.values())
            # return StypyTypeError(localization.get_current(), "Cannot invoke {0} with parameters {1}".format(
            #     format_function_name(type(self.types[0]).__name__), params))
            return StypyTypeError(localization, "Cannot invoke {0} with parameters {1}".format(
                format_function_name(type(self.types[0]).__name__), params))
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

    def __repr__(self):
        """
        String representation of this union type
        :return:
        """
        the_str = ""
        cont = 0

        for type_ in self.types:
            if hasattr(type_, '__name__'):
                the_str += type_.__name__
            else:
                the_str += type(type_).__name__

            if cont < len(self.types) - 1:
                the_str += " U "
            cont += 1

        return the_str

    def __str__(self):
        """
        String representation of this union type
        :return:
        """
        return self.__repr__()

    def __eq__(self, other):
        """
        Union type equality. It compares all the types contained in the union types to be compared
        :param other:
        :return:
        """
        if not isinstance(other, UnionType):
            return False

        other_types = other.types
        if len(self.types) != len(other_types):
            return False

        for type_ in self.types:
            if type_ not in other_types:
                return False
        return True

    def __contains__(self, item_):
        """
        Checks if a union type contains a certain type
        :param item_:
        :return:
        """
        return item_ in self.types

    def duplicate(self):
        """
        Duplicates this union type to another instance that contains the same types (no duplicate of the contained
        types is created)
        :return:
        """
        dup = UnionType()
        for t in self.types:
            dup.types.append(t)

        return dup

    @staticmethod
    def can_be_mergued(type1, type2):
        """
        Checks if two union types can be mergued into one
        :param type1:
        :param type2:
        :return:
        """
        import types as python_types
        from stypy.invokation.handlers import call_utilities
        if type(type1) is python_types.InstanceType and type(type2) is python_types.InstanceType:
            return type1.__class__ == type2.__class__

        if call_utilities.is_numpy_array(type1) and call_utilities.is_numpy_array(type2):
            return type(type1.contained_types) == type(type2.contained_types)

        return False

    @staticmethod
    def mergue(type1, type2):
        """
        Merque two union types into one if possible
        :param type1:
        :param type2:
        :return:
        """
        from stypy.invokation.handlers import call_utilities
        if call_utilities.is_numpy_array(type1) and call_utilities.is_numpy_array(type2):
            return type1

        import types as python_types
        try:
            dir_t1 = dir(type1)
            dir_t2 = dir(type2)
            if len(dir_t2) != len(dir_t1):
                return None

            for member in dir_t2:
                value_t2 = getattr(type2, member)
                if member in dir_t1:
                    if inspect.ismethod(value_t2) or inspect.isfunction(value_t2):
                        continue
                    if getattr(type1, member) is not value_t2:
                        setattr(type1, member, UnionType.add(getattr(type1, member), getattr(type2, member)))
                    else:
                        setattr(type1, member, UnionType.add(python_types.NoneType, getattr(type2, member)))
            return type1
        except:
            return None


# One-time initialization on first load
if UnionType.declared_members is None:
    UnionType.declared_members = [item for item in dir(UnionType) if item not in dir(object)] + \
                                 ["types", "__repr__", "__mro__", "invoke", "__class__", "wrapped_type", "__hash__"]


def is_union_type(type_):
    """
    Check if the passed object is a union type
    :param type_:
    :return:
    """
    return UnionType.isinstance(type_, UnionType)


# ############################################ UNION TYPE CLASSES #############################################

"""
When changing an instance class or inheritance hierarchy to a flow-sensitive type (more than one class) a union type
instance is created as the new value for the __class__ or __bases__ property. This is not acceptable by Python, as only
classes can be valid values for these two properties.

As the new supertype may hold the members of all the classes it holds, we created the UnionTypeClass as a decorator
that allow any union type to behave as an standard class so it can be assigned to these properties while giving access
to all the contained union type elements.

As classes can be either old-style or new-style, a version for each class type is provided.

"""


def get_name(obj):
    if hasattr(obj, "__name__"):
        return obj.__name__

    return type(obj).__name__


old_style_class_txt = """
class {0}:
    stored_union_type = None

    def __getattr__(self, key):
        if key == '__class__' or key == 'stored_union_type':
            return object.__getattribute__(self, key)

        member = getattr(self.stored_union_type, key)
        if isinstance(member, UnionType):
            ret_union = None
            for m in member.types:
                if inspect.ismethod(m):
                    # Copy method code into a function
                    new_method = types.FunctionType(m.func_code, m.func_globals, name = m.func_name,
                       argdefs = m.func_defaults,
                       closure = m.func_closure)
                    try:
                        new_method.__dict__.update(m.__dict__)
                        # Create a bound method from the function
                        new_method = types.MethodType(new_method, self, type(self))

                        # Compose the new union type of bound methods
                        ret_union = UnionType.add(ret_union, new_method)
                    except Exception as e:
                        pass
                        # print e
            if ret_union is not None:
                # Assign (each call overwrite previous values to respond to class updates)
                setattr(self, key, ret_union)
                return ret_union

        return member

    def __setattr__(self, key, value):
        if key == 'stored_union_type':
            object.__setattr__(self, key, value)
        return setattr(self.stored_union_type, key, value)
"""

new_style_class_txt = """
class {0}(object):
    stored_union_type = None

    def __getattribute__(self, key):
        if key == '__class__' or key == 'stored_union_type':
            return object.__getattribute__(self, key)

        member = getattr(self.stored_union_type, key)
        if isinstance(member, UnionType):
            ret_union = None
            for m in member.types:
                if inspect.ismethod(m):
                    # Copy method code into a function
                    new_method = types.FunctionType(m.func_code, m.func_globals, name = m.func_name,
                       argdefs = m.func_defaults,
                       closure = m.func_closure)
                    new_method.__dict__.update(m.__dict__)
                    # Create a bound method from the function
                    new_method = types.MethodType(new_method, self, type(self))

                    # Compose the new union type of bound methods
                    ret_union = UnionType.add(ret_union, new_method)
            if ret_union is not None:
                # Assign (each call overwrite previous values to respond to class updates)
                setattr(self, key, ret_union)
                return ret_union

        return member

    def __setattr__(self, key, value):
        if key == 'stored_union_type':
            object.__setattr__(self, key, value)
        return setattr(self.stored_union_type, key, value)
"""


def create_union_type_class(union_type, old_style=True):
    """
    Creates a union type class of the provided style to handle the situations explained before
    :param union_type:
    :param old_style:
    :return:
    """
    class_name = ""
    for t in union_type.types:
        class_name += get_name(t) + "_"

    if old_style:
        exec (old_style_class_txt.format(class_name))
    else:
        exec (new_style_class_txt.format(class_name))

    ret_type = eval(class_name)
    setattr(ret_type, 'stored_union_type', union_type)

    return ret_type
