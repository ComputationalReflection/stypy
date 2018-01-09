from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.types_of_parameters import types_of_parameters
from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.src_writers import type_rule_writer, error_writer, \
    source_code_writer
from src_file_management import prepare_destination_files_for_class
from stypy.python_lib.type_rules.raw_type_rule_generation.python_modules_type_check.type_rules.types_of_parameters.type_rule import TypeRule

import os
import class_info


def generate_code_for_class(class_instance, module_name, entity_name, class_name, type_to_mask_name=None,
                            superclass_package="stypy.type_expert_system.types.library.python_wrappers.python_type",
                            superclass_name="PythonType", extra_imports=list()):
    rule_file, class_file, error_file = prepare_destination_files_for_class(module_name, entity_name, class_name)

    if os.path.exists(rule_file) and os.path.exists(class_file) and os.path.exists(error_file):
        print "Code for {0} of module {1} has been already generated. Skipping...".format(class_name, module_name)
        return None

    print "Generating attribute member rules for {0}".format(class_name)
    attribute_rules = types_of_parameters.get_type_rules_of_attribute_members(class_instance)

    print "Generating callable member rules for {0}".format(class_name)

    custom_params = [class_instance, 1, 2, 3, 4]
    has_self = True

    callable_rules, errors = types_of_parameters.get_type_rules_of_callable_members(class_instance, custom_params,
                                                                                    maximum_arity=3, has_self=has_self)
    print "Generating arities from type rules of {0}".format(class_name)
    arities = types_of_parameters.get_callable_members_parameter_arity_from_type_rules(callable_rules)

    if not type_to_mask_name is None:
        # Add the special "python_type" attribute for type checking purposes
        type_rule = TypeRule(class_instance, class_info.python_type_attribute_name, None, type_to_mask_name)
        type_rule.set_return_type_name(type_to_mask_name)

        attribute_rules[class_info.python_type_attribute_name] = type_rule

    # Write rules
    print "Writing type rules for {0} ({1})".format(class_name, rule_file)
    type_rule_writer.write_rules_to_file(rule_file, class_name, attribute_rules, arities, callable_rules, extra_imports)

    #Write class code
    print "Writing python code for {0} ({1})".format(class_name, class_file)
    source_code_writer.write_source_code_for_class(class_file, class_instance, class_name,
                                                   superclass_package, superclass_name, attribute_rules,
                                                   callable_rules, extra_imports)
    #Write errors
    error_writer.write_error_rules_to_file(error_file, class_name, errors)

    return None