import os
import sys
import inspect

from ....python_lib_copy.member_call_copy.handlers_copy.call_handler_copy import CallHandler
from .... import stypy_parameters_copy


class FileTypeModifier(CallHandler):
    """
    Apart from type rules stored in python files, there are a second type of file called the type modifier file. This
    file contain functions whose name is identical to the member that they are attached to. In case a function for
    a member exist, this function is transferred the execution control once the member is called and a type rule is
    found to match with the call. Programming a type modifier is then a way to precisely control the return type of
     a member call, overriding the one specified by the type rule. Of course, not every member call have a type
     modifier associated, just those who need special treatment.
    """

    # Cache of found type modifiers
    modifiers_cache = dict()

    # Cache of not found type modifiers
    unavailable_modifiers_cache = dict()

    @staticmethod
    def __modifier_files(parent_name, entity_name):
        """
        For a call to parent_name.entity_name(...), compose the name of the type modifier file that will correspond to
        the entity or its parent, to look inside any of them for suitable modifiers to call
        :param parent_name: Parent entity (module/class) name
        :param entity_name: Callable entity (function/method) name
        :return: A tuple of (name of the rule file of the parent, name of the type rule of the entity)
        """
        parent_modifier_file = stypy_parameters_copy.ROOT_PATH + stypy_parameters_copy.RULE_FILE_PATH + parent_name + "/" \
                               + parent_name + stypy_parameters_copy.type_modifier_file_postfix + ".py"

        own_modifier_file = stypy_parameters_copy.ROOT_PATH + stypy_parameters_copy.RULE_FILE_PATH + parent_name + "/" \
                            + entity_name.split('.')[-1] + "/" + entity_name.split('.')[
                                -1] + stypy_parameters_copy.type_modifier_file_postfix + ".py"

        return parent_modifier_file, own_modifier_file

    def applies_to(self, proxy_obj, callable_entity):
        """
        This method determines if this type modifier is able to respond to a call to callable_entity. The modifier
        respond to any callable code that has a modifier file associated. This method search the modifier file and,
        if found, loads and caches it for performance reasons. Cache also allows us to not to look for the same file on
        the hard disk over and over, saving much time. callable_entity modifier files have priority over the rule files
        of their parent entity should both exist.

        Code of this method is mostly identical to the code that searches for rule files on type_rule_call_handler

        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param callable_entity: Callable entity
        :return: bool
        """
        # We have a class, calling a class means instantiating it
        if inspect.isclass(callable_entity):
            cache_name = proxy_obj.name + ".__init__"
        else:
            cache_name = proxy_obj.name

        # No modifier file for this callable (from the cache)
        if self.unavailable_modifiers_cache.get(cache_name, False):
            return False

        # There are a modifier file for this callable (from the cache)
        if self.modifiers_cache.get(cache_name, False):
            return True

        # There are a modifier file for this callable parent entity (from the cache)
        if proxy_obj.parent_proxy is not None:
            if self.modifiers_cache.get(proxy_obj.parent_proxy.name, False):
                return True

        # Obtain available rule files depending on the type of entity that is going to be called
        if inspect.ismethod(callable_entity) or inspect.ismethoddescriptor(callable_entity) or (
                    inspect.isbuiltin(callable_entity) and
                    (inspect.isclass(proxy_obj.parent_proxy.get_python_entity()))):
            try:
                parent_type_rule_file, own_type_rule_file = self.__modifier_files(
                    callable_entity.__objclass__.__module__,
                    callable_entity.__objclass__.__name__,
                )
            except:
                if inspect.ismodule(proxy_obj.parent_proxy.get_python_entity()):
                    parent_type_rule_file, own_type_rule_file = self.__modifier_files(
                        proxy_obj.parent_proxy.name,
                        proxy_obj.parent_proxy.name)
                else:
                    parent_type_rule_file, own_type_rule_file = self.__modifier_files(
                        proxy_obj.parent_proxy.parent_proxy.name,
                        proxy_obj.parent_proxy.name)
        else:
            parent_type_rule_file, own_type_rule_file = self.__modifier_files(proxy_obj.parent_proxy.name,
                                                                              proxy_obj.name)

        # Determine which modifier file to use
        parent_exist = os.path.isfile(parent_type_rule_file)
        own_exist = os.path.isfile(own_type_rule_file)
        file_path = ""

        if parent_exist:
            file_path = parent_type_rule_file

        if own_exist:
            file_path = own_type_rule_file

        # Load rule file
        if parent_exist or own_exist:
            dirname = os.path.dirname(file_path)
            file_ = file_path.split('/')[-1][0:-3]

            sys.path.append(dirname)
            module = __import__(file_, globals(), locals())
            entity_name = proxy_obj.name.split('.')[-1]
            try:
                # Is there a modifier function for the specific called entity? Cache it if it is
                method = getattr(module.TypeModifiers, entity_name)
                self.modifiers_cache[cache_name] = method
            except:
                # Not available: cache unavailability
                self.unavailable_modifiers_cache[cache_name] = True
                return False

        if not (parent_exist or own_exist):
            if proxy_obj.name not in self.unavailable_modifiers_cache:
                # Not available: cache unavailability
                self.unavailable_modifiers_cache[cache_name] = True

        return parent_exist or own_exist

    def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
        """
        Calls the type modifier for callable entity to determine its return type.

        :param proxy_obj: TypeInferenceProxy that hold the callable entity
        :param localization: Caller information
        :param callable_entity: Callable entity
        :param arg_types: Arguments
        :param kwargs_types: Keyword arguments
        :return: Return type of the call
        """
        if inspect.isclass(callable_entity):
            cache_name = proxy_obj.name + ".__init__"
        else:
            cache_name = proxy_obj.name

        modifier = self.modifiers_cache[cache_name]

        # Argument types passed for the call
        argument_types = tuple(list(arg_types) + kwargs_types.values())
        return modifier(localization, proxy_obj, argument_types)
