import abc


class CallHandler:
    """
    Base abstract class for all the call handlers
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def applies_to(self, proxy_obj, callable_entity):
        """
        This method determines if this call handler can respond to a call to this entity.
        """
        return False

    @abc.abstractmethod
    def __call__(self, proxy_obj, localization, callable_entity, *arg_types, **kwargs_types):
        """
        This method calls callable_entity(localization, *arg_types, **kwargs_types) with the call handler strategy
        modeled by its subclasses.
        :param proxy_obj:
        :param localization:
        :param callable_entity:
        :param arg_types:
        :param kwargs_types:
        :return:
        """
        pass

    @staticmethod
    def compose_type_modifier_member_name(name):
        return "type_modifier_" + name
