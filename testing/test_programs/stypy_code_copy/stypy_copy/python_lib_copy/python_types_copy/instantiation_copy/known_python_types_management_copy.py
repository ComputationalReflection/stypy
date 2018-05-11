from ....python_lib_copy.python_types_copy.instantiation_copy.known_python_types_copy import *
import types

"""
Several functions that work with known Python types, placed here to facilitate the readability of the code.
Altough any list of known types may be used, the functions are aimed to manipulate the
known_python_type_typename_samplevalues type list defined in known_python_types.py

Automatic type rule generation makes heavy usage of these functions to create type rules by invoking members using
known types
"""


def get_known_types(type_table=known_python_type_typename_samplevalues):
    """
    Obtains a list of the known types in the known_python_type_typename_samplevalues list (used by default)
    """
    return type_table.keys()


def add_known_type(type_, type_name, type_value, type_table=known_python_type_typename_samplevalues):
    """
    Allows to add a type to the list of known types in the known_python_type_typename_samplevalues list (used by
    default)
    """
    type_table[type_] = (type_name, type_value)


def remove_known_type(type_, type_table=known_python_type_typename_samplevalues):
    """
    Delete a type to the list of known types in the known_python_type_typename_samplevalues list (used by default)
    """
    del type_table[type_]


def is_known_type(type_, type_table=known_python_type_typename_samplevalues):
    """
    Determines if a type or a type name is in the list of known types in the known_python_type_typename_samplevalues
    list (used by default)
    """
    # Is a type name instead of a type?
    if isinstance(type_, str):
        for table_entry in type_table:
            if type_ == type_table[table_entry][0]:
                return True

        return False
    else:
        return type_ in type_table


def get_known_types_and_values(type_table=known_python_type_typename_samplevalues):
    """
    Obtains a list of the library known types and a sample value for each one from the
    known_python_type_typename_samplevalues  list (used by default)
    """
    list_ = type_table.items()
    ret_list = []
    for elem in list_:
        ret_list.append((elem[0], elem[1][1]))

    return ret_list


def get_type_name(type_, type_table=known_python_type_typename_samplevalues):
    """
    Gets the type name of the passed type as defined in the known_python_type_typename_samplevalues
    list (used by default)
    """
    if type_ == types.NotImplementedType:
        return "types.NotImplementedType"

    try:
        return type_table[type_][0]
    except (KeyError, TypeError):
        if type_ is __builtins__:
            return '__builtins__'

        if hasattr(type_, "__name__"):
            if hasattr(ExtraTypeDefinitions, type_.__name__):
                return "ExtraTypeDefinitions." + type_.__name__

            if type_.__name__ == "iterator":
                return "ExtraTypeDefinitions.listiterator"

            return type_.__name__
        else:
            return type(type_).__name__


def get_type_sample_value(type_, type_table=known_python_type_typename_samplevalues):
    """
    Gets a sample value of the passed type from the known_python_type_typename_samplevalues
    list (used by default)
    """
    return type_table[type_][1]


def get_type_from_name(name, type_table=known_python_type_typename_samplevalues):
    """
    Gets the type object of the passed type name from the known_python_type_typename_samplevalues
    list (used by default)
    """
    if "NotImplementedType" in name:
        return "types.NotImplementedType"

    keys = type_table.keys()
    for key in keys:
        if name == type_table[key][0]:
            return key
