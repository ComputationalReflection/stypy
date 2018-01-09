import types
import inspect


class DocStringCallInfo:
    def __init__(self):
        self.call_line = ""
        self.member = ""
        self.return_type_str = ""
        self.mandatory_params = []
        self.optional_params = []
        self.defaults = {}
        self.kwargs = {}
        self.has_varargs = False
        self.has_kwargs = False
        self.invalid = True

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "Member '{0}' analysis for call line: \n{1}\n\n".format(self.member, self.call_line)

        if self.invalid:
            s += "INVALID CALL INFO: The docstring could not be analyzed.\n"
        else:
            s += "Mandatory parameters = {0}\n".format(str(self.mandatory_params))
            s += "Optional parameters = {0}\n".format(str(self.optional_params))
            s += "Defaults = {0}\n".format(str(self.defaults))
            s += "Kwargs = {0}\n".format(str(self.kwargs))
            s += "Has variable arguments = {0}\n".format(str(self.has_varargs))
            s += "Has keyword arguments = {0}\n".format(str(self.has_kwargs))
            s += "Return type description = {0}\n".format(self.return_type_str)

        return s


class DocStringInfo:
    invalid_name_chars = [" ", ":", "-", "(", ")", '\'', '\"']
    return_specifiers = ["-->", "--", "->", "returns", "return", "<==>"]

    def __init__(self, member, docstring, ismethod=False):
        self.ismethod = ismethod
        self.member = member
        self.original_docstring = docstring
        self.call_infos = []
        self.analyze_docstring()

        self.analyze_docstring()

    def __format_param_name(self, param_name):
        param_name = param_name.strip()

        for char in self.invalid_name_chars:
            param_name = param_name.replace(char, "_")

        return param_name

    def __extract_call_lines(self):
        result_lines = []

        if self.original_docstring is None:
            return []

        lines = self.original_docstring.split('\n')

        for line in lines:
            if not self.ismethod:
                if line.startswith(self.member + "(") and not "etc." in line:
                    result_lines.append(line)
            else:
                temp = "." + self.member + "("
                if temp in line and not "etc." in line:
                    # Leave the line without the class specifier
                    line = line[line.index(".") + 1:]
                    result_lines.append(line)
                elif line.startswith(self.member + "(") and not "etc." in line:
                    result_lines.append(line)
        return result_lines

    def __analyze_mandatory_parameters(self, arg_str):
        param_names = []
        defaults = {}
        kwargs = {}
        keywords_dict = defaults
        rest_of_arg_str = arg_str
        use_kwargs = False
        use_varargs = False

        params = arg_str.split(",")
        for param in params:
            if "..." in param:
                keywords_dict = kwargs
                rest_of_arg_str = rest_of_arg_str[len(param) + 1:]
                use_kwargs = True
                use_varargs = True
                continue

            if "=" in param:
                param_name, param_type = param.split('=')
                try:
                    param_type = type(eval(param_type))
                except:
                    if "func" in param_type:
                        param_type = types.FunctionType
                    else:
                        if "." in param_type:
                            try:
                                __import__(param_type.split('.')[0])
                                param_type = type(eval(param_type))
                            except:
                                pass

                keywords_dict[self.__format_param_name(param_name)] = param_type
                if not use_kwargs:
                    param_names.append(self.__format_param_name(param_name))
                rest_of_arg_str = rest_of_arg_str[len(param) + 1:]
            else:
                if "[" in param:
                    param_name = param.split("[")[0]
                    if not param_name == "":
                        param_names.append(self.__format_param_name(param_name))
                    rest_of_arg_str = rest_of_arg_str[len(param_name):]
                    break
                else:
                    if not param == '':
                        param_names.append(self.__format_param_name(param))
                        rest_of_arg_str = rest_of_arg_str[len(param) + 1:]

        return param_names, defaults, kwargs, use_varargs, rest_of_arg_str

    def __analyze_optional_params(self, arg_str):
        optional_params = arg_str.split(",")
        result = []
        use_varargs = False
        defaults = {}

        for op_param in optional_params:
            if "[...]" in op_param:
                op_param = op_param.strip("[...]")
                op_param = op_param.strip()
                if not op_param == "":
                    result.append(self.__format_param_name(op_param))
                use_varargs = True
                continue

            if "...]" in op_param:
                op_param = op_param.strip("...]")
                op_param = op_param.strip()
                if not op_param == "":
                    result.append(self.__format_param_name(op_param))
                use_varargs = True
                continue

            if "=" in op_param:
                param_name, param_type = op_param.split('=')
                try:
                    param_type = type(eval(param_type))
                except:
                    if "func" in param_type:
                        param_type = types.FunctionType
                    else:
                        if "." in param_type:
                            try:
                                __import__(param_type.split('.')[0])
                                param_type = type(eval(param_type))
                            except:
                                pass
                defaults[self.__format_param_name(param_name)] = param_type
                op_param = param_name

            op_param = op_param.strip()
            op_param = op_param.strip("[")
            op_param = op_param.strip("]")

            if op_param == '':
                continue
            result.append(op_param)

        return result, defaults, use_varargs

    def __split_args_from_return_description(self, arg_str):
        call_params = arg_str
        return_str = ""

        for specifier in self.return_specifiers:
            if specifier in call_params:
                call_params, return_str = call_params.split(specifier)
                return call_params, return_str

        return call_params, return_str

    def __add_call_info(self, return_str="", call_line="", params=[], optional_params=[], defaults={}, kwargs={},
                        has_varargs=False, invalid=True):

        call_info = DocStringCallInfo()

        call_info.call_line = call_line
        call_info.member = self.member
        call_info.return_type_str = return_str
        call_info.mandatory_params = params
        call_info.optional_params = optional_params
        call_info.defaults = defaults
        call_info.kwargs = kwargs
        call_info.has_varargs = has_varargs
        call_info.has_kwargs = len(kwargs) > 0
        call_info.invalid = invalid

        self.call_infos.append(call_info)

    def analyze_docstring(self):
        calls = self.__extract_call_lines()
        if len(calls) == 0:
            self.__add_call_info(call_line=self.original_docstring)
            return

        for call in calls:
            call_params, return_str = self.__split_args_from_return_description(call)

            args_str = call_params.strip().strip(self.member).strip("(")
            args_str = args_str.strip(")")

            params, defaults, kwargs, has_varargs, args_str = self.__analyze_mandatory_parameters(args_str)

            has_optional_params = not args_str.find("[") == -1
            if has_optional_params:
                optional_params, optional_defaults, has_varargs_in_op_params = self.__analyze_optional_params(args_str)
                has_varargs = has_varargs or has_varargs_in_op_params
                for elem in optional_defaults:
                    defaults[elem] = optional_defaults[elem]
            else:
                optional_params = []

            self.__add_call_info(return_str, call, params, optional_params, defaults, kwargs,
                                 has_varargs, invalid=False)

    def __repr__(self):
        return_str = ""

        for call_info in self.call_infos:
            return_str += str(call_info)

        return return_str

    def __str__(self):
        return self.__repr__()


def __callable_filter(target, m):
    try:
        return callable(getattr(target, m)) and not inspect.isclass(getattr(target, m))
    except AttributeError:
        return False


def callable_members_of(module_or_class):
    return filter(lambda member: __callable_filter(module_or_class, member), dir(module_or_class))


def extract_docstring_info_from_single_member(member, ismethod=False):
    return DocStringInfo(member, member.__doc__, ismethod)


def extract_docstrings_from(element):
    if inspect.isclass(element):
        ismethod = True
    else:
        ismethod = False

    doc_strings = sorted(
        map(lambda attribute: DocStringInfo(attribute, getattr(element, attribute).__doc__, ismethod),
            callable_members_of(element)), key=lambda docstring: docstring.member)

    doc_strings_map = {}

    def add_to_dict(dic, name, elem):
        dic[name] = elem
        return dic

    reduce(lambda x, y: add_to_dict(x, y.member, y), doc_strings, doc_strings_map)

    return doc_strings_map