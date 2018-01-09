from stypy.python_lib.type_rules.type_groups.type_group_generator import RuleGroupGenerator

import inspect

class TypeRuleGroupGenerator:
    def __init__(self, raw_rules):
        self.raw_rules = raw_rules

    # def consolidate_rule_params(self, rules):
    #     param_list = []
    #
    #     if len(rules) > 0:
    #         ret_type = rules.values()[0]
    #     else:
    #         ret_type = None
    #     for (key, value) in rules.items():
    #         for i in range(len(key)):
    #             if len(param_list) <= i:
    #                 param_list.append([key[i]])
    #             else:
    #                 if not key[i] in param_list[i]:
    #                     param_list[i].append(key[i])
    #
    #     return (param_list, ret_type)

    def separate_rules_by_different_return_types(self, rules):
        classifier = dict()

        for (key, value) in rules.items():
            if inspect.isfunction(value):
                value = value.func_name
            if not value in classifier.keys():
                classifier[value] = list()
            classifier[value].append(key)

        return classifier

    def separate_rules_by_different_param_number(self, rules):
        classifier = dict()

        for element in rules:
            value = len(element)
            if not value in classifier.keys():
                classifier[value] = list()
            classifier[value].append(element)

        return classifier

    def analyze(self):
        classification_by_rt = self.separate_rules_by_different_return_types(self.raw_rules)

        for item in classification_by_rt:
            classification_by_rt[item] = self.separate_rules_by_different_param_number(classification_by_rt[item])
#            print item,": ", classification_by_rt[item]

        total_rules = []
        for rt in classification_by_rt:
            for param_number in classification_by_rt[rt]:
                if param_number > 0:
                    param_rule = ([RuleGroupGenerator()]*param_number, rt)
                    aggregated_rule = ([None]*param_number, rt)
                    for params in classification_by_rt[rt][param_number]:
                        for i in xrange(len(params)):
                            param_rule[0][i].add_type(params[i])

                    for i in xrange(len(param_rule[0])):
                        aggregated_rule[0][i] = param_rule[0][i].get_rule_group()
                else:
                    aggregated_rule = ([], rt)

                formatted_rule = eval("self.format_grouped_rules_{0}(aggregated_rule)".format(param_number))
                total_rules.append(formatted_rule)

        return total_rules

    def format_grouped_rules_0(self, grouped_rules):
        formatted_rules = []

        ret_type = grouped_rules[-1]
        params = grouped_rules[0]

        rule = ((), ret_type)
        formatted_rules.append(rule)

        return formatted_rules

    def format_grouped_rules_1(self, grouped_rules):
        formatted_rules = []

        ret_type = grouped_rules[-1]
        params = grouped_rules[0]

        for i in xrange(len(params[0])):
            rule = (params[0][i], ret_type)
            formatted_rules.append(rule)

        return formatted_rules

    def format_grouped_rules_2(self, grouped_rules):
        formatted_rules = []

        ret_type = grouped_rules[-1]
        params = grouped_rules[0]

        for i in xrange(len(params[0])):
            for j in xrange(len(params[1])):
                rule = (params[0][i], params[1][j], ret_type)
                formatted_rules.append(rule)

        return formatted_rules

    def format_grouped_rules_3(self, grouped_rules):
        formatted_rules = []

        ret_type = grouped_rules[-1]
        params = grouped_rules[0]

        for i in xrange(len(params[0])):
            for j in xrange(len(params[1])):
                for k in xrange(len(params[2])):
                    rule = (params[0][i], params[1][j], params[2][j], ret_type)
                    formatted_rules.append(rule)

        return formatted_rules

from stypy.python_lib.type_rules.raw_type_rule_generation.modules.math.math__type_rules import \
    type_rules_of_members as math_type_rules

# trg = TypeRuleGroupGenerator(math_type_rules['pow'])
# grouped_rules = trg.analyze()
# for rule in grouped_rules:
#     print rule
#
# print("")
#
# trg = TypeRuleGroupGenerator(math_type_rules['cosh'])
# grouped_rules = trg.analyze()
# for rule in grouped_rules:
#     print rule


print("")

trg = TypeRuleGroupGenerator(math_type_rules['ldexp'])
grouped_rules = trg.analyze()
for rule in grouped_rules:
    print rule