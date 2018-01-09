
def write_type_rules_for_type_members(code_file, type_source, type_member_rules):
    return
    # """
    # Function to write python module wrapper source code to a file.
    # :param code_file: Destination file to write python code
    # :param module_name: Name of the module to generate
    # :param attribute_members: Attributes of the module
    # :param callable_members: Functions of the module
    # """
    # with open(code_file, "w") as fcode:
    #     if not "__builtin" in type_source:
    #         fcode.write("import {0}\n".format(type_source))
    #
    #     fcode.write("from stypy.type_expert_system.types.library.known_types import type_instantiation\n")
    #     fcode.write("from stypy.type_expert_system.types.library.known_types import known_python_types_handling\n")
    #     fcode.write("\n\n")
    #
    #     fcode.write("type_member_rules = {\n")
    #     for rule in type_member_rules:
    #         fcode.write("    {0}: ({1}, {2}),\n".format(rule, type_member_rules[rule][0], type_member_rules[rule][1]))
    #
    #     fcode.write("}")
