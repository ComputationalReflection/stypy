#!/usr/bin/env python
# -*- coding: utf-8 -*-
from stypy import ssa
from stypy.errors.advice import Advice
from stypy.errors.type_error import StypyTypeError
from stypy.errors.type_warning import TypeWarning, UnreferencedLocalVariableTypeWarning
from stypy.reporting.localization import Localization
from stypy.reporting.print_utils import print_context_contents
from stypy.type_inference_programs.stypy_interface import get_builtin_python_type, is_builtin_python_type, \
    get_builtin_python_type_instance
from stypy.types import type_inspection
from stypy.types import type_intercession
from stypy.types import union_type
from stypy.types.type_wrapper import TypeWrapper
from stypy.visitor.type_inference.visitor_utils.stypy_functions import default_function_ret_var_name, auto_var_name


class Context(object):
    """
    Represents a function context, used to contain the local variables and parameter of any function analyzed by stypy
    """

    # Parent context of this one
    module_parent_contexts = {}
    auto_var_counter = 0

    def __init__(self, parent_context, context_name, can_access_parent_contexts=True):
        """
        Creates a new context with the specified parent context
        :param parent_context:
        :param context_name: Normally the name of a function
        :param can_access_parent_contexts: Determines if when looking for the type of a variable we access parent
        context or just restrict to this one
        :return:
        """

        # This is used when handing special cases when some variables have not an inferred type, but a manually assigned
        # one.
        self.manual_type_vars = list()

        # Parent context of this context
        self.parent_context = parent_context

        # Name of the context (if present)
        self.context_name = context_name

        # Module context (no parent)
        if self.parent_context is None:
            Context.module_parent_contexts[context_name] = self
            # Teared down function contexts are stored for reporting variables created during the execution for
            # debugging purposes
            self.last_function_contexts = []

        self.next_context = None
        self.add_next_context()

        # types stored on this specific context
        self.types_of = dict()

        # Global variables
        self.globals = []

        # Unreferenced variable accesses (usage of global variables without global declaration)
        self.unreferenced_var_warnings = []

        # Aliases
        self.aliases = dict()

        self.can_access_parent_contexts = can_access_parent_contexts

        if parent_context is None:
            # Root contexts (modules) has these predefined variables:
            self.__load_predefined_variables()

    # ######################################### CONTEXT LINKING AND LOCALIZATION ######################################

    def __load_predefined_variables(self):
        """
        Puts into the Context a set of variables that every context have defined (__file__, __doc__, __name__ and
        __package__
        :return:
        """
        global_localization = Localization(self.context_name, 1, 1)

        self.get_global_context().set_type_of(global_localization, '__file__', self.context_name)
        self.get_global_context().set_type_of(global_localization, '__doc__',
                                              None)
        self.get_global_context().set_type_of(global_localization, '__name__',
                                              get_builtin_python_type_instance(global_localization, 'str'))
        self.get_global_context().set_type_of(global_localization, '__package__',
                                              None)
        import __builtin__
        self.get_global_context().set_type_of(global_localization, '__builtins__',
                                              __builtin__)

    def add_next_context(self):
        """
        Puts this context as the next context of its parent
        :return:
        """
        if self.parent_context is not None:
            self.parent_context.next_context = self

    def remove_next_context(self):
        """
        Removes this context as the next context of its parent
        :return:
        """
        if self.parent_context is not None:
            self.parent_context.next_context = None

    def get_current_active_context(self):
        """
        Returns the topmost context in the stack in which this context is placed
        :return:
        """
        current_context = self
        while True:
            temp = current_context.next_context
            if temp is None:
                return current_context
            current_context = temp

    @staticmethod
    def exist_context_for_module(module_name):
        """
        Determines if the passed module name has a context created
        :param module_name:
        :return:
        """
        return Context.module_parent_contexts.has_key(module_name)

    @staticmethod
    def get_current_active_context_for_module(module_name):
        """
        Determines the passed module name active context
        :param module_name:
        :return:
        """
        return Context.module_parent_contexts[module_name].get_current_active_context()

    # ############################################### TYPE / MEMBER GET/SET ############################################

    def __has_parent_context(self):
        """
        Determines if this context has a parent one
        :return:
        """
        return self.parent_context is not None and self.can_access_parent_contexts

    def __contains__(self, name):
        """
        In operator (working with type names)
        :param name: Name of the stored variable
        :return:
        """
        if name in self.types_of:
            return True

        if self.__has_parent_context():
            return name in self.parent_context
        else:
            global_context = self.get_global_context()
            if self is not global_context:
                return name in self.get_global_context()

        # Variable not found
        return False

    def __getitem__(self, item):
        """
        [] operator, equivalent to get_type_of call
        :param item: Name of the variable to locate its type
        :return:
        """
        return self.get_type_of(Localization.get_current(), item)

    def add_alias(self, alias_name, variable_name):
        """
        Adds an alias to the alias storage of this function context
        :param alias_name: Name of the alias
        :param variable_name: Name of the aliased variable
        :return:
        """
        self.aliases[alias_name] = variable_name

    def has_type_of(self, localization, name):
        """
        Gets the type of the specified variable name, looking on its parent context if not found on its own type store
        :param localization:
        :param name:
        :return:
        """

        if name in self.aliases.keys():
            name = self.aliases[name]

        # Is this a global?
        if name in self.globals:
            global_context = self.get_global_context()
            return global_context.has_type_of(localization, name)

        # If not, lookup inside our types
        if name in self.types_of:
            return name in self.types_of

        # If not found, lookup our parent context (if any) recursively
        if self.__has_parent_context():
            return self.parent_context.has_type_of(localization, name)
        else:
            global_context = self.get_global_context()
            if self is not global_context:
                return global_context.has_type_of(localization, name)

        # Variable not found
        return False

    def get_type_of(self, localization, name, check_unreferenced=True):
        """
        Gets the type of the specified variable name, looking on its parent context if not found on its own type store
        :param localization:
        :param name:
        :param check_unreferenced Tells the function to check for local access to a global variable without using
        the "global" keyword
        :return:
        """
        if name == 'None':
            return type(None)

        if name in self.aliases.keys():
            name = self.aliases[name]

        # Is this a global?
        if name in self.globals:
            global_context = self.get_global_context()
            return global_context.get_type_of(localization, name)

        # If not, lookup inside our types
        if name in self.types_of:
            return self.types_of[name]

        # If not found, lookup our parent context (if any) recursively
        if self.__has_parent_context():
            type_ = self.parent_context.get_type_of(localization, name)

            # It is not an error
            if not isinstance(type_, StypyTypeError):
                global_context = self.get_global_context()
                # Found in the global context
                if global_context.exist_name_in_local_context(name) and type(self) is FunctionContext:
                    # This is a potentially unreferenced variable (local access to a global variable without using
                    # the "global" keyword
                    if check_unreferenced:
                        if not self.has_type_of(localization,
                                                type_inspection.get_name(type_)) and self is not global_context:
                            self.unreferenced_var_warnings.append(
                                UnreferencedLocalVariableTypeWarning(localization, name, self))
            return type_
        else:
            global_context = self.get_global_context()
            if self is not global_context:
                type_ = global_context.get_type_of(localization, name)

                if not isinstance(type_, StypyTypeError):
                    # Found in the global context
                    if type(self) is FunctionContext:
                        # This is a potentially unreferenced variable (local access to a global variable without using
                        # the "global" keyword
                        if check_unreferenced:
                            if not self.has_type_of(localization,
                                                    name) and self is not global_context:
                                                  #  type_inspection.get_name(type_)) and self is not global_context:
                                # and not type_inspection.is_function(type_) and not type_inspection.is_method(type_):
                                self.unreferenced_var_warnings.append(
                                    UnreferencedLocalVariableTypeWarning(localization, name, self))
                return type_

        # If not found, lookup among Python builtin types
        if is_builtin_python_type(localization, name):
            return get_builtin_python_type(localization, name)

        # Variable not found
        return StypyTypeError.name_not_defined_error(localization, name)

    def set_type_of(self, localization, name, type_):
        """
        Adds / changes the type of the variable 'name' in this context
        :param localization:
        :param name:
        :param type_:
        :return:
        """
        if name in self.manual_type_vars:
            return  # Manual variables cannot be written

        # The special return type variable behaves differently
        if name == default_function_ret_var_name:
            if name in self.types_of:
                self.types_of[name] = union_type.UnionType.add(self.types_of[name], type_)
                return

        # Setting a value to the __all__ variable removes previous static declarations of exportable members
        if name is "__all__":
            if hasattr(self, 'exportable_members'):
                del self.exportable_members

        if name in self.aliases.keys():
            name = self.aliases[name]

        if name in self.globals:
            global_context = self.get_global_context()
            return global_context.set_type_of(localization, name, type_)

        # Have we detected potential unreferenced variable warnings
        unreferenced_warnings = filter(lambda warn: warn.name == name and warn.context == self,
                                       self.unreferenced_var_warnings)
        if len(unreferenced_warnings) > 0:
            for warn in unreferenced_warnings:
                self.unreferenced_var_warnings.remove(warn)
                TypeWarning.remove_warning_msg(warn)
                StypyTypeError.unbound_local_error(warn.localization, warn.name)
            return

        self.types_of[name] = type_

    def del_type_of(self, localization, name):
        """
        Del the type of the specified variable name, looking on its parent context if not found on its own type store
        :param localization:
        :param name:
        :return:
        """

        if name in self.aliases.keys():
            del self.aliases[name]
            return None

        # Is this a global?
        if name in self.globals:
            global_context = self.get_global_context()
            global_context.del_type_of(localization, name)
            return None

        # If not, lookup inside our types
        if name in self.types_of:
            del self.types_of[name]
            return None

        # If not found, lookup our parent context (if any) recursively
        if self.__has_parent_context():
            return self.parent_context.del_type_of(localization, name)

        # Variable not found
        return StypyTypeError.name_not_defined_error(localization, name)

    def has_member(self, localization, obj, name):
        """
        Gets the type of the specified member name for the object obj, looking on its parent context if not found
        on its own type store
        :param localization:
        :param obj:
        :param name:
        :return:
        """
        Localization.set_current(localization)
        if isinstance(obj, TypeWrapper) and obj.is_declared_member(name):
            try:
                obj = obj.get_type_of_member(name)
                return True
            except:
                return False

        return hasattr(obj, name)

    def get_type_of_member(self, localization, obj, name):
        """
        Gets the type of the specified member name for the object obj, looking on its parent context if not found
        on its own type store
        :param localization:
        :param obj:
        :param name:
        :return:
        """
        member = type_intercession.get_member_from_object(localization, obj, name)

        if type(member) is StypyTypeError and name != "__eq__":
            # Store the error in the current context to turn it into a warning if dealing with a union type
            self.set_type_of(localization, auto_var_name + str(Context.auto_var_counter), member)
            Context.auto_var_counter += 1

        return member

    def set_type_of_member(self, localization, obj, name, type_):
        """
        Sets the type of the specified member name for the object obj, looking on its parent context if not found
        on its own type store
        :param localization:
        :param obj:
        :param name:
        :param type_:
        :return:
        """
        return type_intercession.set_member_to_object(localization, obj, name, type_)

    def del_member(self, localization, obj, name):
        """
        Sets the type of the specified member name for the object obj, looking on its parent context if not found
        on its own type store
        :param localization:
        :param obj:
        :param name:
        :return:
        """
        return type_intercession.del_member_from_object(localization, obj, name)

    def set_exportable_members(self, all_contents):
        """
        If the analyzed module has an __all__ variable, its contents are stored unprocessed to determine what members
        of the module are truly exportable
        :param all_contents: Contents of the __all__ variable
        :return: None
        """
        self.exportable_members = all_contents

    def get_public_names_and_types(self):
        """
        Gets all the public variables within this type store function contexts and its types
        in a {name: type} dictionary
        :return: {name: type} dictionary
        """

        if hasattr(self, 'exportable_members'):
            return self.exportable_members

        name_type_dict = {}

        for name in self.types_of:
            if name.startswith("_"):
                continue
            name_type_dict[name] = self.types_of[name]

        return name_type_dict

    def add_manual_type_var(self, var_name):
        """
        Declares a variable as one whose type will be manually assigned
        :param var_name:
        :return:
        """
        self.manual_type_vars.append(var_name)

    # ############################################### GLOBALS MANAGEMENT ##############################################

    def exist_name_in_local_context(self, var_name):
        """
        Determines if a variable name is defined in this context
        :param var_name:
        :return:
        """
        return var_name in self.types_of

    def get_global_context(self):
        """
        Gets the global context corresponding to this one
        :return:
        """
        global_context = self

        while True:
            temp = global_context.parent_context
            if temp is None:
                break
            global_context = temp

        return global_context

    def declare_global(self, localization, var_name):
        """
        Declares a global variable for this context
        :param localization:
        :param var_name:
        :return:
        """
        global_context = self.get_global_context()
        if global_context is self:  # We are the global context
            # Use of "global x" after declaring x
            if global_context.exist_name_in_local_context(var_name):
                Advice.syntax_warning_name_assigned_before_global_advice(localization, var_name)
            else:
                # Use of "global x" before declaring x
                Advice.redeclared_without_usage_advice(localization, var_name)
        else:  # We are not the global context
            if not global_context.exist_name_in_local_context(var_name):
                if self.exist_name_in_local_context(var_name):
                    global_context.set_type_of(localization, var_name, self.get_type_of(localization, var_name))
                    Advice.syntax_warning_name_assigned_before_global_advice(localization, var_name)
                else:
                    Advice.global_not_defined_advice(localization, var_name)

            # Have we detected potential unreferenced variable warnings: the variable was used prior to global
            # declaration
            unreferenced_warnings = filter(lambda warn: warn.name == var_name and warn.context == self,
                                           self.unreferenced_var_warnings)
            if len(unreferenced_warnings) > 0:
                Advice.syntax_warning_name_used_before_global_advice(localization, var_name)

            self.globals.append(var_name)

    # ############################################### CONTEXT MANAGEMENT ##############################################

    def open_function_context(self, context_name, line=0, column=0, access_parent=True):
        """
        Open a new funtion context child of this one
        :param context_name:
        :param line:
        :param column:
        :param access_parent:
        :return:
        """
        return FunctionContext(self, context_name, line, column, access_parent)

    def close_function_context(self):
        """
        Closes this context and removes it from the context stack
        :return:
        """
        # Cleanup unreferenced variable warnings
        for unref_warn in self.unreferenced_var_warnings:
            TypeWarning.remove_warning_msg(unref_warn)
            unref_warn.turn_to_advice()

        root_context = self.get_global_context()
        root_context.last_function_contexts.append(self)

        self.remove_next_context()
        return self.parent_context

    def get_last_function_context_for(self, context_name):
        """
        Gets the last used function context whose name is the one passed to this function
        :param context_name: Context name to search
        :return: Function context
        """
        root_context = self.get_global_context()
        if root_context.context_name == context_name:
            return root_context

        last_expired_context = None
        # Lookup expired function contexts
        for last_context in root_context.last_function_contexts:
            if last_context.context_name == context_name:
                last_expired_context = last_context

        if last_expired_context is not None:
            return last_expired_context

        # Lookup active function context
        temp = root_context.next_context
        while temp is not None:
            if context_name == temp.context_name:
                return temp
            temp = temp.next_context

        return None

    def store_return_type_of_current_context(self, return_type):
        """
        Changes the return type of the current function context
        :param return_type: Type
        :return:
        """
        self.return_type = return_type

    # ############################################### CONTEXT PRINT ##############################################

    def __repr__(self):
        """
        Textual representation of the type store
        :return: str
        """
        txt = print_context_contents(self)

        if hasattr(self, 'last_function_contexts'):
            if len(self.last_function_contexts) > 0:
                txt += "Other contexts created during execution:\n"
                for context in self.last_function_contexts:
                    txt += print_context_contents(context)

        return txt

    def __str__(self):
        return self.__repr__()


class FunctionContext(Context):
    """
    Child class whose purpose is to model a function context
    """

    def __init__(self, parent_context, context_name=None, line=0, column=0, access_parent=True):
        """
        Build a new function context
        :param parent_context:  Context of the caller
        :param context_name: Name of the function
        :param line: Position in which this function is declared
        :param column: Position in which this function is declared
        :param access_parent: Can access its caller?
        """
        super(FunctionContext, self).__init__(parent_context, context_name, access_parent)
        self.line = line
        self.column = column
        if hasattr(self.parent_context, 'on_ssa'):
            self.on_ssa = self.parent_context.on_ssa
        self.manual_type_vars = list()

    def __exist_ssa_context_in_context_stack(self):
        """
        Determines wheter there is an SSA context open in the current call stack
        :return:
        """
        temp_context = self
        while True:
            if temp_context.parent_context is None:
                return False
            if isinstance(temp_context.parent_context, ssa.ssa_context.SSAContext):
                return True
            temp_context = temp_context.parent_context

    def __inside_ssa(self, localization, obj, name):
        """
        Determines if the parent context is inside an SSA branch
        :param localization:
        :param obj:
        :param name:
        :return:
        """
        if hasattr(self.parent_context, 'on_ssa'):
            on_ssa = self.parent_context.on_ssa
            if on_ssa:
                return self.parent_context.has_member(localization, obj, name)

        return False

    def has_member(self, localization, obj, name):
        """
        Method override

        :param localization:
        :param obj:
        :param name:
        :return:
        """
        if self.__inside_ssa(localization, obj, name):
            return self.parent_context.has_member(localization, obj, name)
        return super(FunctionContext, self).has_member(localization, obj, name)

    def get_type_of_member(self, localization, obj, name):
        """
        Method override

        :param localization:
        :param obj:
        :param name:
        :return:
        """
        if self.__inside_ssa(localization, obj, name):
            return self.parent_context.get_type_of_member(localization, obj, name)
        return super(FunctionContext, self).get_type_of_member(localization, obj, name)

    def set_type_of_member(self, localization, obj, name, type_):
        """
        Method override

        :param localization:
        :param obj:
        :param name:
        :param type_:
        :return:
        """
        if self.__inside_ssa(localization, obj, name):
            return self.parent_context.set_type_of_member(localization, obj, name, type_)
        return super(FunctionContext, self).set_type_of_member(localization, obj, name, type_)

    def del_member(self, localization, obj, name):
        """
        Method override

        :param localization:
        :param obj:
        :param name:
        :return:
        """
        if self.__inside_ssa(localization, obj, name):
            return self.parent_context.del_member(localization, obj, name)
        return super(FunctionContext, self).del_member(localization, obj, name)

    def close_function_context(self):
        """
        Method override

        :return:
        """
        # Cleanup unreferenced variable warnings
        for unref_warn in self.unreferenced_var_warnings:
            TypeWarning.remove_warning_msg(unref_warn)
            unref_warn.turn_to_advice()

        ssa_call = self.__exist_ssa_context_in_context_stack()
        if ssa_call:
            for key, value in self.types_of.items():
                if auto_var_name in key:
                    continue
                if isinstance(value, StypyTypeError):
                    value.turn_to_warning()

        root_context = self.get_global_context()
        root_context.last_function_contexts.append(self)

        self.remove_next_context()
        return self.parent_context