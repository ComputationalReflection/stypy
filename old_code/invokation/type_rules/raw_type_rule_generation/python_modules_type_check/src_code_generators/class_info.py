
#Code generation constants
python_type_attribute_name = "stypy_python_type"
iterated_structure_attribute_name = "iterated_structure"

special_member_prefix = "stypy__"
class_postfix = "__type"


#Default superclasses for generated data structures
__superclasses = {
    "__builtins__.list": ("stypy.type_expert_system.types.library.python_wrappers.python_data_structures",
           "PythonIndexableDataStructure"),

    "__builtins__.tuple": ("stypy.type_expert_system.types.library.python_wrappers.python_data_structures",
            "PythonIndexableDataStructure"),

    "__builtins__.dict": ("stypy.type_expert_system.types.library.python_wrappers.python_data_structures",
           "PythonDictionary"),
}

"""
These member names are special in the sense that it cannot be directly redefined, as they have special meaning
in stypy or even the Python language behavior. They will be generated, but with the __stypy postfix
"""
__special_member_names = [
    "__init__",
    "__getattribute__",
    "__setattr__",
    "len",
    "__repr__",
    "__str__",
    "__eq__",
    "__cmp__",
    "isinstance"
]


def get_superclass_info(module_name, type_name):
    try:
        return __superclasses[module_name+"."+type_name]
    except KeyError:
        return "stypy.type_expert_system.types.library.python_wrappers.python_type", "PythonType"


def get_member_name(member_name):
    #print "member_name = ", member_name, member_name in __special_member_names
    if member_name in __special_member_names:
        return special_member_prefix + member_name

    return member_name


def is_stypy_attribute(member_name):
    return python_type_attribute_name == member_name or iterated_structure_attribute_name == member_name