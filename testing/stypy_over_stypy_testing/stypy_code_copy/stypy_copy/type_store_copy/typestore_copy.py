import os

from stypy_copy import python_interface_copy
from stypy_copy.errors_copy.type_error_copy import TypeError
from stypy_copy.errors_copy.undefined_type_error_copy import UndefinedTypeError
from stypy_copy.errors_copy.type_warning_copy import TypeWarning, UnreferencedLocalVariableTypeWarning
from function_context_copy import FunctionContext
from stypy_copy.python_lib_copy.python_types_copy import non_python_type_copy
from stypy_copy.python_lib_copy.python_types_copy.type_inference_copy import type_inference_proxy_copy, localization_copy
from type_annotation_record_copy import TypeAnnotationRecord
from stypy_copy import stypy_parameters_copy


class TypeStore(non_python_type_copy.NonPythonType):
    """
    A TypeStore contains all the registered variable, function names and types within a particular file (module).
    It functions like a central storage of type information for the file, and allows any program to perform type
    queries for any variable within the module.

    The TypeStore allows flow-sensitive type storage, as it allows us to create nested contexts in which
    [<variable_name>, <variable_type>] pairs are stored for any particular function or method. Following Python
    semantics a variable in a nested context shadows a same-name variable in an outer context. If a variable is not
    found in the topmost context, it is searched in the more global ones.

    Please note that the TypeStore abstracts away context search semantics, as it only allows the user to create and
    destroy them.
    """

    type_stores_of_modules = dict()

    @staticmethod
    def get_type_store_of_module(module_name):
        """
        Obtains the type store associated with a module name
        :param module_name: Module name
        :return: TypeStore object of that module
        """
        try:
            return TypeStore.type_stores_of_modules[module_name]
        except:
            return None

    def __load_predefined_variables(self):
        self.set_type_of(localization_copy.Localization(self.program_name, 1, 1), '__file__', str)
        self.set_type_of(localization_copy.Localization(self.program_name, 1, 1), '__doc__', str)
        self.set_type_of(localization_copy.Localization(self.program_name, 1, 1), '__name__', str)
        self.set_type_of(localization_copy.Localization(self.program_name, 1, 1), '__package__', str)

    def __init__(self, file_name):
        """
        Creates a type store for the passed file name (module)
        :param file_name: file name to create the TypeStore for
        :return:
        """
        file_name = file_name.replace("\\", "/")
        self.program_name = file_name.replace(stypy_parameters_copy.type_inference_file_postfix, "")
        self.program_name = self.program_name.replace(stypy_parameters_copy.type_inference_file_directory_name + "/", "")

        # At least every module must have a main function context
        main_context = FunctionContext(file_name, True)

        # Create an annotation record for the program, reusing the existing one if it was previously created
        main_context.annotation_record = TypeAnnotationRecord.get_instance_for_file(self.program_name)

        # Initializes the context stack
        self.context_stack = [main_context]

        # Teared down function contexts are stored for reporting variables created during the execution for debugging
        # purposes
        self.last_function_contexts = []

        # Configure if some warnings are given
        self.test_unreferenced_var = stypy_parameters_copy.ENABLE_CODING_ADVICES

        # External modules used by this module have its own type store. These secondary type stores are stored here
        # to access them when needed.
        self.external_modules = []
        self.__load_predefined_variables()

        file_cache = os.path.abspath(self.program_name).replace('\\', '/')
        # Register ourselves in the collection of created type stores
        TypeStore.type_stores_of_modules[file_cache] = self

    def add_external_module(self, stypy_object):
        """
        Adds a external module to the list of modules used by this one
        :param stypy_object:
        :return:
        """
        self.external_modules.append(stypy_object)
        module_type_store = stypy_object.get_analyzed_program_type_store()
        module_type_store.last_function_contexts = self.last_function_contexts

    def get_all_processed_function_contexts(self):
        """
        Obtain a list of all the function context that were ever used during the program execution (active + past ones)
        :return: List of function contexts
        """
        return self.context_stack + self.last_function_contexts

    def set_check_unreferenced_vars(self, state):
        """
        On some occasions, such as when invoking methods or reading default values from parameters, the unreferenced
        var checks must be disabled to ensure proper behavior.
        :param state: bool value. However, if coding advices are disabled, this method has no functionality, they are
        always set to False
        :return:
        """
        if not stypy_parameters_copy.ENABLE_CODING_ADVICES:
            self.test_unreferenced_var = False
            return
        self.test_unreferenced_var = state

    def set_context(self, function_name="", lineno=-1, col_offset=-1):
        """
        Creates a new function context in the top position of the context stack
        """
        context = FunctionContext(function_name)
        context.declaration_line = lineno
        context.declaration_column = col_offset
        context.annotation_record = TypeAnnotationRecord.get_instance_for_file(self.program_name)

        self.context_stack.insert(0, context)

    def unset_context(self):
        """
        Pops and returns the topmost context in the context stack
        :return:
        """
        # Invariant
        assert len(self.context_stack) > 0

        context = self.context_stack.pop(0)
        self.last_function_contexts.append(context)

        return context

    def get_context(self):
        """
        Gets the current (topmost) context.
        :return: The current context
        """
        return self.context_stack[0]

    def mark_as_global(self, localization, name):
        """
        Mark a variable as global in the current function context
        :param localization: Caller information
        :param name: variable name
        :return:
        """
        ret = None
        self.set_check_unreferenced_vars(False)
        var_type = self.get_context().get_type_of(name)
        self.set_check_unreferenced_vars(True)
        if var_type is not None:
            ret = TypeWarning(localization,
                              "SyntaxWarning: name '{0}' is used before global declaration".format(name))
            if not self.get_context() == self.get_global_context():
                # Declaring a variable as global once it has a value promotes it to global
                self.get_global_context().set_type_of(name, var_type, localization)

        unreferenced_var_warnings = filter(lambda warn: isinstance(warn, UnreferencedLocalVariableTypeWarning) and
                                                        warn.name == name and warn.context == self.get_context(),
                                           TypeWarning.get_warning_msgs())

        if len(unreferenced_var_warnings) > 0:
            ret = TypeWarning(localization,
                              "SyntaxWarning: name '{0}' is used before global declaration".format(name))

        global_vars = self.get_context().global_vars

        if name not in global_vars:
            global_vars.append(name)
        return ret

    def get_global_context(self):
        """
        Gets the main function context, the last element in the context stack
        :return:
        """
        return self.context_stack[-1]

    def get_type_of(self, localization, name):
        """
        Gets the type of the variable name, implemented the mentioned context-search semantics
        :param localization: Caller information
        :param name: Variable name
        :return:
        """
        ret = self.__get_type_of_from_function_context(localization, name, self.get_context())

        # If it is not found, look among builtins as python does.
        if isinstance(ret, UndefinedTypeError):
            member = python_interface_copy.import_from(localization, name)

            if isinstance(member, TypeError):
                member.msg = "Could not find a definition for the name '{0}' in the current context. Are you missing " \
                             "an import?".format(name)

            if member is not None:
                # If found here, it is not an error any more
                TypeError.remove_error_msg(ret)
                # ret_member = type_inference_proxy.TypeInferenceProxy.instance(type(member.python_entity))
                # ret_member.type_of = member
                return member
                # return ret_member

        return ret

    def get_context_of(self, name):
        """
        Returns the function context in which a variable is first defined
        :param name: Variable name
        :return:
        """
        for context in self.context_stack:
            if name in context:
                return context

        return None

    def set_type_of(self, localization, name, type_):
        """
        Set the type of a variable using the context semantics previously mentioned.

        Only simple a=b assignments are supported, as multiple assignments are solved by AST desugaring, so all of them
        are converted to equivalent simple ones.
        """
        if not isinstance(type_, type_inference_proxy_copy.Type):
            type_ = type_inference_proxy_copy.TypeInferenceProxy.instance(type_)

        type_.annotation_record = TypeAnnotationRecord.get_instance_for_file(self.program_name)
        # Destination is a single name of a variable
        return self.__set_type_of(localization, name, type_)

    def set_type_store(self, type_store, clone=False):
        """
        Assign to this type store the attributes of the passed type store, cloning the passed
        type store if indicated. This operation is needed to implement the SSA algorithm
        :param type_store: Type store to assign to this one
        :param clone: Clone the passed type store before assigning its values
        :return:
        """
        if clone:
            type_store = TypeStore.__clone_type_store(type_store)

        self.program_name = type_store.program_name
        self.context_stack = type_store.context_stack
        self.last_function_contexts = type_store.last_function_contexts
        self.external_modules = type_store.external_modules
        self.test_unreferenced_var = type_store.test_unreferenced_var

    def clone_type_store(self):
        """
        Clone this type store
        :return: A clone of this type store
        """
        return TypeStore.__clone_type_store(self)

    def get_public_names_and_types(self):
        """
        Gets all the public variables within this type store function contexts and its types
        in a {name: type} dictionary
        :return: {name: type} dictionary
        """
        name_type_dict = {}
        cont = len(self.context_stack) - 1
        # Run through the contexts in inverse order (more global to more local) and store its name - type pairs. This
        # way local variables that shadows global ones take precedence.
        for i in range(len(self.context_stack)):
            ctx = self.context_stack[cont]

            for name in ctx.types_of:
                if name.startswith("__"):
                    continue
                name_type_dict[name] = ctx.types_of[name]

            cont -= 1

        return name_type_dict

    def get_last_function_context_for(self, context_name):
        """
        Gets the last used function context whose name is the one passed to this function
        :param context_name: Context name to search
        :return: Function context
        """
        context = None

        for last_context in self.last_function_contexts:
            if last_context.function_name == context_name:
                context = last_context

        if context is None:
            for context in self.context_stack:
                if context_name == context.function_name:
                    return context

        return context

    def add_alias(self, alias, member_name):
        """
        Adds an alias to the current function context
        :param alias: Alias name
        :param member_name: Aliased variable name
        :return:
        """
        self.get_context().add_alias(alias, member_name)

    def del_type_of(self, localization, name):
        """
        Delete a variable for the first function context that defines it (using the context
        search semantics we mentioned)
        :param localization:
        :param name:
        :return:
        """
        ret = self.__del_type_of_from_function_context(localization, name, self.get_context())

        return ret

    def store_return_type_of_current_context(self, return_type):
        """
        Changes the return type of the current function context
        :param return_type: Type
        :return:
        """
        self.get_context().return_type = return_type

    # ########################################## NON - PUBLIC INTERFACE ##########################################

    def __get_type_of_from_function_context(self, localization, name, f_context):
        """
        Search the stored function contexts for the type associated to a name.
        As we follows the program flow, a correct program ensures that if this query is performed the name actually HAS
        a type (it has been assigned a value previously in the previous executed statements). If the name is not found,
         we have detected a programmer error within the source file (usage of a previously undeclared name). The
         method is orthogonal to variables and functions.
        :param name: Name of the element whose type we want to know
        :return:
        """

        # Obtain the current context
        current_context = f_context
        # Get global context (module-level)
        global_context = self.get_global_context()

        # Is this global? (marked previously with the global keyword)
        if name in current_context.global_vars:
            # Search the name within the global context
            type_ = global_context.get_type_of(name)
            # If it does not exist, we cannot read it (no value was provided for it)
            if type_ is None:
                return TypeError(localization, "Attempted to read the uninitialized global '%s'" % name)
            else:
                # If it exist, return its type
                return type_

        top_context_reached = False

        # If the name is not a global, we run from the more local to the more global context looking for the name
        for context in self.context_stack:
            if context == f_context:
                top_context_reached = True

            if not top_context_reached:
                continue

            type_ = context.get_type_of(name)

            if type_ is None:
                continue

            """
            The type of name is found. In this case, we test if the name is also present into the global context.
            If it is, and was not marked as a global till now, we generate a warning indicating that if a write access
            is performed to name and it is still not marked as global, then Python will throw a runtime error
            complaining that name has been referenced without being assigned first. global have to be used to avoid
            this error.
            """
            # Not marked as global & defined in a non local context & we are not within the global context & is a var
            if self.test_unreferenced_var:
                if name not in current_context.global_vars and \
                        not context == self.get_context() \
                        and not current_context == global_context:
                    UnreferencedLocalVariableTypeWarning(localization, name, current_context)

            return type_

        return UndefinedTypeError(localization, "The variable '%s' does not exist" % str(name))

    def __set_type_of(self, localization, name, type_):
        """
        Cases:

        - Exist in the local context:
            Is marked as global: It means that the global keyword was used after one assignment ->
         assign the variable in the global context and remove from the local
            Is not marked as global: Update
        - Don't exist in the local context:
            Is global: Go to the global context and assign
            Is not global: Create (Update). Shadows more global same-name element
        """
        global_context = self.get_global_context()
        is_marked_as_global = name in self.get_context().global_vars
        exist_in_local_context = name in self.get_context()

        if exist_in_local_context:
            if is_marked_as_global:
                global_context.set_type_of(name, type_, localization)
                del self.get_context().types_of[name]
                return TypeWarning(localization, "You used the global keyword on '{0}' after assigning a value to it. "
                                                 "It is valid, but will throw a warning on execution. "
                                                 "Please consider moving the global statement before "
                                                 "any assignment is done to '{0}'".format(name))
            else:
                self.get_context().set_type_of(name, type_, localization)
        else:
            if is_marked_as_global:
                global_context.set_type_of(name, type_, localization)
            else:
                """Special case:
                    If:
                        - A variable do not exist in the local context
                        - This variable is not marked as global
                        - There exist unreferenced type warnings in this scope typed to this variable.
                    Then:
                        - For each unreferenced type warning found:
                            - Generate a unreferenced variable error with the warning data
                            - Delete warning
                            - Mark the type of the variable as ErrorType
                """
                unreferenced_type_warnings = filter(lambda warning:
                                                    warning.__class__ == UnreferencedLocalVariableTypeWarning,
                                                    TypeWarning.get_warning_msgs())

                if len(unreferenced_type_warnings) > 0:
                    our_unreferenced_type_warnings_in_this_context = filter(lambda warning:
                                                                            warning.context == self.get_context() and
                                                                            warning.name == name,
                                                                            unreferenced_type_warnings)

                    for utw in our_unreferenced_type_warnings_in_this_context:
                        TypeError(localization, "UnboundLocalError: local variable '{0}' "
                                                "referenced before assignment".format(name))
                        TypeWarning.warnings.remove(utw)

                    # Unreferenced local errors tied to 'name'
                    if len(our_unreferenced_type_warnings_in_this_context) > 0:
                        self.get_context().set_type_of(name, TypeError(localization, "Attempted to use '{0}' previously"
                                                                                     " to its definition".format(name)),
                                                       localization)
                        return self.get_context().get_type_of(name)

                contains_undefined, more_types_in_value = type_inference_proxy.TypeInferenceProxy. \
                    contains_an_undefined_type(type_)
                if contains_undefined:
                    if more_types_in_value == 0:
                        TypeError(localization, "Assigning to '{0}' the value of an undefined variable".
                                  format(name))
                    else:
                        TypeWarning.instance(localization,
                                             "Potentialy assigning to '{0}' the value of an undefined variable".
                                             format(name))

                self.get_context().set_type_of(name, type_, localization)
        return None

    @staticmethod
    def __clone_type_store(type_store):
        """
        Clones the type store; eventually it must also clone the values (classes)
        because they can be modified with intercession
        """

        cloned_obj = TypeStore(type_store.program_name)
        cloned_obj.context_stack = []
        for context in type_store.context_stack:
            cloned_obj.context_stack.append(context.clone())

        cloned_obj.last_function_contexts = type_store.last_function_contexts
        cloned_obj.external_modules = type_store.external_modules
        cloned_obj.test_unreferenced_var = type_store.test_unreferenced_var

        return cloned_obj

    # TODO: Remove?
    # def __get_last_function_context_for(self, context_name):
    #     context = None
    #     try:
    #         context = self.last_function_contexts[context_name]
    #     except KeyError:
    #         for context in self.context_stack:
    #             if context_name == context.function_name:
    #                 return context
    #
    #     return context

    def __del_type_of_from_function_context(self, localization, name, f_context):
        """
        Search the stored function contexts for the type associated to a name.
        As we follows the program flow, a correct program ensures that if this query is performed the name actually HAS
        a type (it has been assigned a value previously in the previous executed statements). If the name is not found,
         we have detected a programmer error within the source file (usage of a previously undeclared name). The
         method is orthogonal to variables and functions.
        :param name: Name of the element whose type we want to know
        :return:
        """

        # Obtain the current context
        current_context = f_context
        # Get global context (module-level)
        global_context = self.get_global_context()

        # Is this global? (marked previously with the global keyword)
        if name in current_context.global_vars:
            # Search the name within the global context
            type_ = global_context.get_type_of(name)
            # If it does not exist, we cannot read it (no value was provided for it)
            if type_ is None:
                return TypeError(localization, "Attempted to delete the uninitialized global '%s'" % name)
            else:
                # If it exist, delete it
                return global_context.del_type_of(name)

        top_context_reached = False

        # If the name is not a global, we run from the more local to the more global context looking for the name
        for context in self.context_stack:
            if context == f_context:
                top_context_reached = True

            if not top_context_reached:
                continue

            type_ = context.get_type_of(name)

            if type_ is None:
                continue

            return context.del_type_of(name)

        return UndefinedTypeError(localization, "The variable '%s' does not exist" % str(name))

    # ############################################# SPECIAL METHODS #############################################

    def __len__(self):
        """
        len operator, returning the number of function context stored in this type store
        :return:
        """
        return len(self.context_stack)

    def __iter__(self):
        """
        Iterator interface, to traverse function contexts
        :return:
        """
        for f_context in self.context_stack:
            yield f_context

    def __getitem__(self, item):
        """
        Returns the nth function context in the context stack
        :param item: Index of the function context
        :return: A Function context or an exception if the position is not valid
        """
        return self.context_stack[item]

    def __contains__(self, item):
        """
        in operator, to see if a variable is defined in a function context of the current context stack
        :param item: variable
        :return: bool
        """
        type_ = self.get_type_of(None, item)

        return not (type_.__class__ == TypeError)

    def __repr__(self):
        """
        Textual representation of the type store
        :return: str
        """
        txt = "Type store of file '" + str(self.program_name.split("/")[-1]) + "'\n"
        txt += "Active contexts:\n"

        for context in self.context_stack:
            txt += str(context)

        if len(self.last_function_contexts) > 0:
            txt += "Other contexts created during execution:\n"
            for context in self.last_function_contexts:
                txt += str(context)

        return txt

    def __str__(self):
        return self.__repr__()

    # ############################## MEMBER TYPE GET / SET ###############################

    def get_type_of_member(self, localization, member_name):
        """
        Proxy for get_type_of, to comply with NonPythonType interface
        :param localization: Caller information
        :param member_name: Member name
        :return:
        """
        return self.get_type_of(localization, member_name)

    def set_type_of_member(self, localization, member_name, member_value):
        """
        Proxy for set_type_of, to comply with NonPythonType interface
        :param localization: Caller information
        :param member_name: Member name
        :return:
        """
        return self.set_type_of(localization, member_name, member_value)

    # ############################## STRUCTURAL REFLECTION ###############################

    def delete_member(self, localization, member):
        """
        Proxy for del_type_of, to comply with NonPythonType interface
        :param localization: Caller information
        :param member: Member name
        :return:
        """
        return self.del_type_of(localization, member)

    def supports_structural_reflection(self):
        """
        TypeStores (modules) always support structural reflection
        :return: True
        """
        return True

    # ############################## TYPE CLONING ###############################

    def clone(self):
        """
        Proxy for clone_type_store, to comply with NonPythonType interface
        :return:
        """
        return self.clone_type_store()
