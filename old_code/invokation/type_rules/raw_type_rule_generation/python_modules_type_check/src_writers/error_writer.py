

def write_error_rules_to_file(rule_file, type_name, error_rules):
    """
    Function used to write error type rules to a log file. Format is the same as the python str output when applied
    to a list
    :param rule_file: File to write
    :param type_name: Type name of the owner of the error rules
    :param error_rules: Error rules
    :return:
    """
    with open(rule_file, "w") as frules:
        frules.write("# ----------------------------------\n")
        frules.write("# Python {0} members error rules\n".format(type_name))
        frules.write("# ----------------------------------\n\n\n")

        for error_rule in error_rules:
            frules.write(str(error_rule)+"\n")

