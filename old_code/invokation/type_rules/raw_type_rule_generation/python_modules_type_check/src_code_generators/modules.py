import inspect
import os

from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.types_of_parameters import \
    types_of_parameters
from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.src_writers import type_rule_writer, \
    error_writer, \
    type_member_rule_writer
from src_file_management import prepare_destination_files, prepare_destination_files_for_type_member
from stypy.python_lib.python_types.instantiation import known_python_types_management
from stypy.python_lib.python_types.instantiation import type_instantiation
from stypy.errors.type_error import TypeError
from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.types_of_parameters.type_rule import \
    TypeRule
import class_info


def __is_builtin_module(module_name):
    return "__builtin" in module_name


def __load_module(module_name):
    try:
        exec ("import " + module_name)
    except:
        pass

    # Builtin is a special module
    if not __is_builtin_module(module_name):
        module = eval(module_name)
    else:
        import sys

        module = sys.modules["__builtin__"]

    return module


def generate_code_for_type_member(module_name, type_member_name,
                                  superclass_package="stypy.type_expert_system.types.library.python_wrappers.python_type",
                                  superclass_name="PythonType"):
    module = __load_module(module_name)
    type_member = getattr(module, type_member_name)

    rule_file, class_file, error_file = prepare_destination_files_for_type_member("modules", module_name,
                                                                                  type_member_name)

    if os.path.exists(rule_file): # and os.path.exists(class_file) and os.path.exists(error_file):
        print "Code for {0} of module {1} has been already generated. Skipping...".format(type_member_name, module_name)
        return None

    try:
        instance = known_python_types_management.get_type_sample_value(type_member)
    except:
        instance = type_instantiation.get_type_sample_value(type_member)

    if isinstance(instance, TypeError):
        print "WARNING: Type {0} is not instantiable. Skipping code generation...".format(type_member_name)
        return None

    print "Generating attribute member rules for {0}".format(type_member_name)
    attribute_rules = types_of_parameters.get_type_rules_of_attribute_members(type_member)

    print "Generating callable member rules for {0}".format(type_member_name)

    custom_params = [instance, 1, 2, 3, 4]

    if inspect.isclass(type_member):
        has_self = True
    else:
        has_self = False

    type_added = False

    # Add the type to the list of known types
    if not known_python_types_management.is_known_type(type_member):
        if not __is_builtin_module(module_name):
            type_to_add_name = module_name + "." + type_member_name
        else:
            type_to_add_name = type_member_name

        known_python_types_management.add_known_type(type_member, type_to_add_name, custom_params[0])
        type_added = True
    else:
        type_to_add_name = known_python_types_management.get_type_name(type_member)

    callable_rules, errors = types_of_parameters.get_type_rules_of_callable_members(type_member, custom_params,
                                                                                    maximum_arity=3, has_self=has_self)

    print "Generating arities from type rules of {0}".format(type_member_name)
    arities = types_of_parameters.get_callable_members_parameter_arity_from_type_rules(callable_rules)

    # Add the special "python_type" attribute for type checking purposes
    type_rule = TypeRule(type_member, class_info.python_type_attribute_name, None, type_to_add_name)
    type_rule.set_return_type_name(type_to_add_name)

    attribute_rules[class_info.python_type_attribute_name] = type_rule

    if not __is_builtin_module(module_name):
        extra_imports = [module_name, "from {0} import {1}".format(module_name, type_member_name)]
    else:
        extra_imports = list()

    # Write rules
    print "Writing type rules for {0} ({1})".format(type_member_name, rule_file)

    #Eliminate firts parameter (own type) and leave only non-duplicate rules
    for member_name in callable_rules.keys():
        rule_list = callable_rules[member_name]
        for rule in rule_list:
            rule.delete_first_parameter()
        final_rule_list = []
        for rule in rule_list:
            if not rule in final_rule_list:
                final_rule_list.append(rule)
        callable_rules[member_name] = final_rule_list

    type_rule_writer.write_rules_to_file(rule_file, type_member_name, attribute_rules, arities, callable_rules,
                                         extra_imports)

    # Write class code
    # print "Writing python code for {0} ({1})".format(type_member_name, class_file)
    # source_code_writer.write_source_code_for_class(class_file, type_member, type_member_name,
    #                                                superclass_package, superclass_name, attribute_rules,
    #                                                callable_rules, extra_imports)

    # Write errors
    error_writer.write_error_rules_to_file(error_file, type_member_name, errors)

    if type_added and __is_builtin_module(module_name):
        # This is needed for performance reasons (we cannot allow the table of known types to grow without control). For
        # that, new builtin types are not allowed to remain into the known type table for the whole type generation
        # process for the module. Other modules do not follow that rule.
        known_python_types_management.remove_known_type(type_member)
    else:
        if type_added:
            return type_member

    return None


def generate_code_for_module(module_name, maximum_arity=4, excluded_members=None, only_for_members=None):
    if not excluded_members == None and not only_for_members == None:
        raise Exception("Exclusion of members and generation for specific members cannot be used at the same time")

    module = __load_module(module_name)

    if not only_for_members == None:
        # File preparing
        rule_file, class_file, error_file, type_members_file = prepare_destination_files("modules",
                                                                                         module_name + "_specific_members")
    else:
        rule_file, class_file, error_file, type_members_file = prepare_destination_files("modules", module_name)

    # print "Generating attribute member rules for module {0}".format(module_name)
    attribute_rules = types_of_parameters.get_type_rules_of_attribute_members(module)

    # if not module_name in __modules_excluded_from_type_member_generation:
    print "Generating type member rules for module '{0}'".format(module_name)
    type_member_rules = types_of_parameters.get_type_rules_of_type_members(module)

    print "Detected {0} types inside module '{1}'".format(len(type_member_rules), module_name)
    #type_member_rule_writer.write_type_rules_for_type_members(type_members_file, module_name, type_member_rules)

    added_types = []
    cont = 1
    type_members = types_of_parameters.get_type_members(module)
    for type_member in type_members:
        print "Writing code for type '{0}' of module {1} ({2}/{3})".format(type_member, module_name, cont,
                                                                         len(type_member_rules))

        superclass_package, superclass_name = class_info.get_superclass_info(module_name, type_member)

        # added_type = generate_code_for_type_member(module_name, "list", superclass_package, superclass_name)
        # return

        added_type = generate_code_for_type_member(module_name, type_member, superclass_package, superclass_name)
        if not added_type is None:
            added_types.append(added_type)

        cont += 1

    if os.path.exists(rule_file): #and os.path.exists(class_file) and os.path.exists(error_file):
        print "Code for module '{0}' has been already generated. Skipping...".format(module_name)
        return None

    print "Generating callable member rules for module '{0}'".format(module_name)
    callable_rules, errors = types_of_parameters.get_type_rules_of_callable_members(module, maximum_arity=maximum_arity,
                                                                                    excluded_members=excluded_members,
                                                                                    only_for_members=only_for_members)

    #print "Generating arities from type rules of module {0}".format(module_name)
    arities = types_of_parameters.get_callable_members_parameter_arity_from_type_rules(callable_rules)

    if not __is_builtin_module(module_name):
        extra_imports = [module_name]
    else:
        extra_imports = list()

    # Write rules
    print "Writing type rules for module '{0}' ({1})".format(module_name, rule_file)
    type_rule_writer.write_rules_to_file(rule_file, module_name, attribute_rules, arities, callable_rules,
                                         extra_imports)

    # Write class code
    #print "Writing python code for module {0} ({1})".format(module_name, class_file)
    #source_code_writer.write_source_code_for_module(class_file, module_name, attribute_rules, callable_rules)

    # Write errors
    error_writer.write_error_rules_to_file(error_file, module_name, errors)

    # Types of certain modules are deleted at the end of the module generation for performance reasons.
    for added_type in added_types:
        known_python_types_management.remove_known_type(added_type)
