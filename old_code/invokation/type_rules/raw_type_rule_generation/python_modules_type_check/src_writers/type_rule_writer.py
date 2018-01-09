from stypy.python_lib.python_types.instantiation.known_python_types_management import ExtraTypeDefinitions


def __write_rule_file_header(frules, type_name, extra_imports=list()):
    """
    Writes a standard header for type rule files
    :param frules: File to write to
    :param type_name: Name of the type rule file
    """
    frules.write("# ----------------------------------\n")
    frules.write("# Python {0} members type rules\n".format(type_name))
    frules.write("# ----------------------------------\n\n\n")

    frules.write("import types\n")
    frules.write(
        "from stypy.python_lib.python_types.instantiation.known_python_types_management import ExtraTypeDefinitions\n")
    frules.write("from stypy.python_lib.python_types.type_inference.undefined_type import UndefinedType\n")
    frules.write("from stypy.python_lib.type_rules.type_groups.type_group_generator import *\n")
    # frules.write("from stypy.type_expert_system.types.library.type_inference.any_type import AnyType\n")
    # frules.write("from stypy.type_expert_system.types.library.type_inference.vararg_type import VarArgType\n")
    # frules.write(
    #     "from stypy.code_generation.python_modules_type_check.type_rules.types_of_parameters.conditional_type_rules_functions import *\n")
    # frules.write(
    #     "from stypy.code_generation.python_modules_type_check.docstring_analysis.docstring_analyzer import DocStringInfo\n")

    for import_ in extra_imports:
        if import_.startswith("from"):
            frules.write("{0}\n".format(import_))
        else:
            frules.write("import {0}\n".format(import_))

    if type_name in ExtraTypeDefinitions.__dict__:
        frules.write("\n# We cannot directly name this type to use its members, to we have to extract it from this "
                     "class")
        frules.write("\n{0} = ExtraTypeDefinitions.{0}\n\n".format(type_name))

    frules.write("\n\n\n")


def write_rules_to_file(rule_file, type_name, attribute_type_rules, arities, callable_type_rules, extra_imports=list()):
    """
    Writes type rules to a Python source code file. A type rule file includes two dictionaries:
    - arities: That indicate the acceptable amount of number of parameters admitted by each function
    - type_rules_of_members: A dictionary with a key for each member, associated with a dictionary of
    <tuple>:<return type>. This associated a combination of parameters with the expected return type when
    calling the function with these parameter types
    :param rule_file: File to write code to
    :param type_name: Type whose code will be written
    :param attribute_type_rules: Type rules for those elements that are attributes (pieces of data)
    :param arities: A list of tuples (<callable member name>:<list of admitted parameter arities>)
    :param callable_type_rules: A dictionary of <callable member name>: <dictionary of <tuple>:<return type>
    """
    with open(rule_file, "w") as frules:
        __write_rule_file_header(frules, type_name, extra_imports)

        # Docstrings of callable members
        # frules.write("\n# Docstring information of all the members of this Python type or module\n")
        # frules.write("docstrings_of_callable_members = {\n")
        # for member_arity in arities:
        #     if not type_name is "__builtins__":
        #         temp = "    '{0}': DocStringInfo('{0}', {1}.{0}.__doc__),\n".format(member_arity[0], type_name)
        #     else:
        #         temp = "    '{0}': DocStringInfo('{0}', {0}.__doc__),\n".format(member_arity[0])
        #     frules.write(temp)
        # frules.write("}\n\n")

        # Parameter arities of callable members
        # frules.write("\n# Callable member parameter arities for type-checking procedures\n")
        # frules.write("arities_of_callable_members = {\n")
        # for member_arity in arities:
        #     frules.write("    '" + member_arity[0] + "': " + str(member_arity[1]) + ", \n")
        # frules.write("}\n\n")

        # frules.write("\n# Type rules of all the members of this type or module\n")
        frules.write("type_rules_of_members = {\n")
        # Attributes
        # for member_name in attribute_type_rules:
        #     type_rule = attribute_type_rules[member_name]
        #     frules.write("    # " + member_name + " is an attribute\n")
        #     frules.write("    '" + member_name + "': " + type_rule.return_type_name + ", \n")
        #
        # frules.write("\n\n")

        # Callables
        for member_arity in arities:
            member_name = member_arity[0]
            # Write only the current member type rules
            member_type_rules = callable_type_rules[member_name]

            # frules.write("    # " + member_name + " can be invoked with the following number of parameters: "
            #              + str(member_arity[1]) + "\n")
            frules.write("    '" + member_name + "': [\n")
            for type_rule in member_type_rules:
                frules.write("        (" + str(type_rule.param_type_names).replace("'", "") + ", "
                             + type_rule.return_type_name + "), \n")
            frules.write("    ],\n")

        frules.write("}\n")
