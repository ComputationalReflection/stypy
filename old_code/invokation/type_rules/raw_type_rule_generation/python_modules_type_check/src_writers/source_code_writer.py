
import source_code_models
from stypy.python_lib.python_types.instantiation.known_python_types_management import get_type_name


def write_source_code_for_module(code_file, module_name, attribute_members, callable_members):
    """
    Function to write python module wrapper source code to a file.
    :param code_file: Destination file to write python code
    :param module_name: Name of the module to generate
    :param attribute_members: Attributes of the module
    :param callable_members: Functions of the module
    """
    with open(code_file, "w") as fcode:
        fcode.write(source_code_models.code_model_imports(module_name,
                                                          "stypy.type_expert_system.types.python.type_management.type_checker",
                                                          "module_type_check"))

        for member in attribute_members:
            fcode.write(source_code_models.code_model_attribute(member, attribute_members[member].return_type_name))

        for member in callable_members:
            fcode.write(source_code_models.code_model_function(module_name, member,
                                                               "module_type_check"))


def write_source_code_for_class(code_file, data_structure, class_name, superclass_package, superclass_name, attribute_members, callable_members, extra_imports=list()):
    """
    Function to write python module wrapper source code to a file.
    :param code_file: Destination file to write python code
    :param class_name: Name of the module to generate
    :param attribute_members: Attributes of the module
    :param callable_members: Functions of the module
    """
    with open(code_file, "w") as fcode:
        fcode.write(source_code_models.code_model_imports(get_type_name(data_structure),
                                                          "stypy.type_expert_system.types.python.type_management.type_checker",
                                                          "class_type_check", extra_imports))

        fcode.write(source_code_models.code_model_class(class_name, superclass_package, superclass_name))

        for member in attribute_members:
            fcode.write(source_code_models.code_model_attribute(member, attribute_members[member].return_type_name,
                                                                indent="    "))

        for member in callable_members:
            fcode.write(source_code_models.code_model_method(class_name, member, "class_type_check"))

