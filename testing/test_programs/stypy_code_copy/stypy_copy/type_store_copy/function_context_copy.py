import copy
from stypy_copy.python_lib_copy.python_types_copy.type_copy import Type
from stypy_copy.reporting_copy import print_utils_copy
from stypy_copy.errors_copy.type_error_copy import TypeError

class FunctionContext:
    """
    Models a function/method local context, containing all its variables and types in a dictionary. A type store holds
    a stack of function contexts, one per called function, tracking all its local context. This class also have the
    optional feature of annotating types to create type-annotated programs, allowing the type annotation inside
    functions code
    """
    annotate_types = True

    def __init__(self, function_name, is_main_context=False):
        """
        Initializes the function context for function function_name
        :param function_name: Name of the function
        :param is_main_context: Whether it is the main context or not. There can be only a function context in the
        program.
        :return:
        """

        # Types of local variables/parameters (name: type)
        self.types_of = {}

        # Function name
        self.function_name = function_name

        # Global variables applicable to the function
        self.global_vars = []

        # Aliases of variables aplicable to the function
        self.aliases = dict()

        self.is_main_context = is_main_context

        # Context information
        # Declared named argument list
        self.declared_argument_name_list = None

        # Declared varargs variable name (if any)
        self.declared_varargs_var = None

        # Declared keyword arguments variable name (if any)
        self.declared_kwargs_var = None

        # Declared defaults for parameters (if any)
        self.declared_defaults = None

        # Position of the function inside the source code
        self.declaration_line = -1
        self.declaration_column = -1

        # Return type of the function
        self.return_type = None

    def get_header_str(self):
        """
        Obtains an appropriate str to pretty-print the function context, formatting the header of the represented
        function.
        :return: str
        """
        txt = ""
        arg_str = ""
        if self.declared_argument_name_list is not None:
            for arg in self.declared_argument_name_list:
                arg_str += str(arg) + ": " + str(self.get_type_of(arg)) + ", "

            if arg_str is not "":
                arg_str = arg_str[:-2]

        if self.declared_varargs_var is not None:
            if arg_str is not "":
                arg_str += ", "
            str_varargs = "*" + str(self.declared_varargs_var) + ": " + str(self.get_type_of(self.declared_varargs_var))

            arg_str += str_varargs

        if self.declared_kwargs_var is not None:
            if arg_str is not "":
                arg_str += ", "
            str_kwargs = "**"+str(self.declared_kwargs_var) + ": " + str(self.get_type_of(self.declared_kwargs_var))

            arg_str += str_kwargs

        txt += str(self.function_name) + "(" + arg_str + ") -> " + print_utils.get_type_str(self.return_type)

        return txt

    def __repr__(self):
        """
        String representation of the function context
        :return: str
        """
        txt = ""
        if self.is_main_context:
            txt += "Program '" + str(self.function_name) + "'\n"
        else:
            if self.declaration_line is not -1:
                txt = self.get_header_str()
                txt += " (Line: " + str(self.declaration_line) + ", Column: " + str(self.declaration_column) + ")\n"

        for name in self.types_of:
            type_ = self.types_of[name]
            if isinstance(type_, TypeError):
                txt += "\t" + name + ": TypeError\n"
            else:
                txt += "\t" + name + ": " + str(type_) + "\n"

        return txt

    def __str__(self):
        """
        String representation of the function context
        :return: str
        """
        return self.__repr__()

    def __contains__(self, item):
        """
        in operator, to determine if the function context contains a local variable
        :param item:
        :return:
        """
        return item in self.types_of.keys()

    def add_alias(self, alias_name, variable_name):
        """
        Adds an alias to the alias storage of this function context
        :param alias_name: Name of the alias
        :param variable_name: Name of the aliased variable
        :return:
        """
        self.aliases[alias_name] = variable_name

    def get_type_of(self, variable_name):
        """
        Returns the type of a variable or parameter in the local context
        :param variable_name: Name of the variable in the context
        :return: The variable type or None if the variable do not belong to this context locally
        """
        if variable_name in self.aliases.keys():
            variable_name = self.aliases[variable_name]

        if variable_name in self.types_of:
            return self.types_of[variable_name]

        return None

    def set_type_of(self, name, type_, localization):
        """
        Sets the type of name to type in this local context
        :param name: Name to search
        :param type: Type to assign to name
        """
        if self.annotate_types:
            self.annotation_record.annotate_type(localization.line, localization.column, name, type_)

        if name in self.aliases.keys():
            name = self.aliases[name]
        self.types_of[name] = type_

    def del_type_of(self, variable_name):
        """
        Deletes the type of a variable or parameter in the local context
        :param variable_name: Name of the variable in the context
        """
        if variable_name in self.types_of:
            del self.types_of[variable_name]

        return None

    def __iter__(self):
        """
        Allows iteration through all the variable names stored in the context.
        :return: Each variable name stored in the context
        """
        for variable_name in self.types_of:
            yield variable_name

    def __getitem__(self, item):
        """
        Allows the usage of the [] operator to access variable types by variable name
        :param item: Variable name
        :return: Same as get_type_of
        """
        return self.get_type_of(item)

    def __len__(self):
        """
        len operator, returning the amount of stored local variables
        :return:
        """
        return len(self.types_of)

    def clone(self):
        """
        Clones the whole function context. The returned function context is a deep copy of the current one
        :return: Cloned function context
        """
        cloned_obj = FunctionContext(self.function_name)

        cloned_obj.global_vars = copy.deepcopy(self.global_vars)

        for key, value in self.types_of.iteritems():
            if isinstance(value, Type):
                new_obj = value.clone()
            else:
                new_obj = copy.deepcopy(value)

            cloned_obj.types_of[key] = new_obj

        cloned_obj.aliases = copy.deepcopy(self.aliases)
        cloned_obj.annotation_record = self.annotation_record
        cloned_obj.is_main_context = self.is_main_context

        # Context information
        cloned_obj.declared_argument_name_list = self.declared_argument_name_list
        cloned_obj.declared_varargs_var = self.declared_varargs_var
        cloned_obj.declared_kwargs_var = self.declared_kwargs_var
        cloned_obj.declared_defaults = self.declared_defaults

        cloned_obj.declaration_line = self.declaration_line
        cloned_obj.declaration_column = self.declaration_column

        cloned_obj.return_type = self.return_type

        return cloned_obj

    def copy(self):
        """
        Copies this function context into a newly created one and return it. The copied function context is a shallow
        copy.
        :return: Copy of this function context
        """
        copied_obj = FunctionContext(self.function_name)

        copied_obj.global_vars = self.global_vars
        copied_obj.types_of = self.types_of

        copied_obj.aliases = self.aliases
        copied_obj.annotation_record = self.annotation_record
        copied_obj.is_main_context = self.is_main_context

        # Context information
        copied_obj.declared_argument_name_list = self.declared_argument_name_list
        copied_obj.declared_varargs_var = self.declared_varargs_var
        copied_obj.declared_kwargs_var = self.declared_kwargs_var
        copied_obj.declared_defaults = self.declared_defaults

        copied_obj.declaration_line = self.declaration_line
        copied_obj.declaration_column = self.declaration_column

        copied_obj.return_type = self.return_type

        return copied_obj

