from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.types_of_parameters.type_rule import TypeRule
from stypy.python_lib.python_types.instantiation.known_python_types_management import get_type_name
from stypy.python_lib.type_rules.type_groups.type_groups import AnyType
import types
import inspect

"""
Number of parameters accepted by some known function calls. This is used additionaly of calculating them.
"""
__predefined_additional_types_of_parameters = {

}


def generate_rule_for_member(type_, member_name):
    type_base = get_type_name(type_).split('.')[-1]

    if "sort" in member_name and type_ == list:
        return [
            TypeRule(type_, type_base + '.sort', (type_,), None),
            ]

    if "__getslice__" in member_name and type_ == list:
        return [
            TypeRule(type_, type_base + '.__getslice__', (type_, slice), list),
            ]

    if "__getattribute__" in member_name:
        return [
            TypeRule(type_, type_base + '.__getattribute__', (type_, str), None),
            TypeRule(type_, type_base + '.__getattribute__', (type_, unicode), None)
        ]

    if "__delattr__" in member_name:
        return [
            TypeRule(type_, type_base + '.__delattr__', (type_, str), None),
            TypeRule(type_, type_base + '.__delattr__', (type_, unicode), None)
        ]

    if "__setslice__" in member_name and inspect.isclass(type_):
        return [
            TypeRule(type_, type_base + '.__setslice__', (type_, int, int, AnyType), None),
            TypeRule(type_, type_base + '.__setslice__', (type_, int, long, AnyType), None),
            TypeRule(type_, type_base + '.__setslice__', (type_, int, bool, AnyType), None),

            TypeRule(type_, type_base + '.__setslice__', (type_, long, int, AnyType), None),
            TypeRule(type_, type_base + '.__setslice__', (type_, long, long, AnyType), None),
            TypeRule(type_, type_base + '.__setslice__', (type_, long, bool, AnyType), None),

            TypeRule(type_, type_base + '.__setslice__', (type_, bool, int, AnyType), None),
            TypeRule(type_, type_base + '.__setslice__', (type_, bool, long, AnyType), None),
            TypeRule(type_, type_base + '.__setslice__', (type_, bool, bool, AnyType), None),

            TypeRule(type_, type_base + '.__setslice__', (type_, int, int, AnyType), None),
            TypeRule(type_, type_base + '.__setslice__', (type_, int, int, AnyType), None),
            TypeRule(type_, type_base + '.__setslice__', (type_, int, int, AnyType), None),
        ]
    #if "iterator" in type_base and "next" in member_name:

    return None


def get_additional_type_rules(type_, member_name):
    generated_rules = generate_rule_for_member(type_, member_name)
    if not generated_rules is None:
        return generated_rules

    try:
        #Try with type_.member_name
        code_to_invoke = getattr(type_, member_name)
        if code_to_invoke in __predefined_additional_types_of_parameters:
            return __predefined_additional_types_of_parameters[code_to_invoke]
        else:
            #Try with "type_.member_name" (some types are not hashable, so we cannot add them directly)
            id = get_type_name(type_) + "." + member_name
            if id in __predefined_additional_types_of_parameters:
                return __predefined_additional_types_of_parameters[id]
    except:
        #Try with "type_.member_name" (some types are not hashable, so we cannot add them directly)
        id = get_type_name(type_) + "." + member_name
        if id in __predefined_additional_types_of_parameters:
            return __predefined_additional_types_of_parameters[id]

    return None

# import sys
# print get_predefined_type_rules(sys.modules["__builtin__"], "dir")
# print get_predefined_type_rules(list, "__new__")
# print get_predefined_type_rules(list, "__subclasshook__")