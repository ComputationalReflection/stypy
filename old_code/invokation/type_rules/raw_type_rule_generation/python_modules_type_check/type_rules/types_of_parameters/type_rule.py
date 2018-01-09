from stypy.python_lib.python_types.instantiation.known_python_types_management import get_type_name
from stypy.python_lib.python_types.instantiation.known_python_types import known_python_type_typename_samplevalues


class TypeRule:
    """
    A TypeRule contains all the information of a TypeRule that represent a valid (or erroneous) call to any
    Python callable. It stores information such as the owner of the call, the type name of the owner, the
    member to be called, the param types (and type names) to be used in a call and its return type (and name).
    If the TypeRule contains an erroneous call, the type and message of the thrown exception are also stored
    in the TypeRule
    """

    def get_number_of_params(self):
        return len(self.param_types)

    def __init__(self, owner_obj, member_name, param_types, return_obj, is_error_rule=False, is_conditional_rule=False):
        """
        :param owner_obj: Owner of the call
        :param member_name: Member to be called
        :param param_types: Types of parameters to perform the call
        :param return_obj: Return type once the call is made
        :param is_error_rule: Indicates if the rule represent an erroneous call or not.
        :return:
        """
        self.owner_obj = owner_obj
        self.owner_name = get_type_name(self.owner_obj)
        self.member_name = member_name
        self.param_types = param_types

        self.recalculate_param_type_names()

        self.return_obj = return_obj
        self.is_error = is_error_rule
        self.is_conditional_rule = is_conditional_rule

        if not self.is_error:
            try:
                if return_obj in known_python_type_typename_samplevalues:
                    self.return_type_name = get_type_name(return_obj)
                else:
                    self.return_type_name = get_type_name(type(return_obj))
            except:
                self.return_type_name = get_type_name(type(return_obj))

        else:
            self.exception_type = type(return_obj)
            self.exception_msg = str(self.exception_type) + ": " + str(return_obj)

        if is_conditional_rule:
            self.return_type_name = return_obj

    def clone(self):
        clone = TypeRule(self.owner_obj, self.member_name, self.param_types, self.return_obj, self.is_error,
                         self.is_conditional_rule)

        clone.owner_obj = self.owner_obj
        clone.owner_name = self.owner_name
        clone.member_name = self.member_name
        clone.param_types = self.param_types

        clone.return_obj = self.return_obj
        clone.is_error = self.is_error
        clone.is_conditional_rule = self.is_conditional_rule

        clone.return_type_name = self.return_type_name

        if self.is_error:
            clone.exception_type = self.exception_type
            clone.exception_msg = self.exception_msg

        clone.return_type_name = self.return_type_name

        return clone

    def recalculate_param_type_names(self):
        param_type_names = []
        if not self.param_types is None:
            for param_type in self.param_types:
                if not param_type == '*':  # Placeholder, not a real type
                    param_type_names.append(get_type_name(param_type))
                else:
                    param_type_names.append('*')

        self.param_type_names = tuple(param_type_names)

    def delete_first_parameter(self):
        if len(self.param_types) > 0:
            self.param_types = self.param_types[1:]
            self.recalculate_param_type_names()

    def __call__(self):
        """
        Obtain a list with the representation of the different elements of a type rule
        :return: list
        """
        if self.is_error:
            return [self.owner_name, self.member_name, self.param_type_names, self.exception_msg]
        else:
            return [self.owner_name, self.member_name, self.param_type_names, self.return_type_name]

    def __repr__(self):
        return str(self())

    def __str__(self):
        return self.__repr__()

    def __eq__(self, other):
        result = self.owner_name == other.owner_name
        result = result and (self.member_name == other.member_name)

        result = result and (self.param_type_names == other.param_type_names)
        result = result and (self.is_error == other.is_error)
        result = result and (self.is_conditional_rule == other.is_conditional_rule)

        if not self.is_error:
            result = result and (self.return_type_name == other.return_type_name)
        else:
            result = result and (self.exception_msg == other.exception_msg)

        if self.is_conditional_rule:
            result = result and (self.return_type_name == other.return_type_name)

        return result

    def set_function_return_type(self, function_call):
        self.is_conditional_rule = True
        self.return_type_name = function_call

    def set_return_type_name(self, return_type_name):
        self.return_type_name = return_type_name
