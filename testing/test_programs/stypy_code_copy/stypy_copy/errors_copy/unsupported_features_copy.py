from type_error_copy import TypeError


def create_unsupported_python_feature_message(localization, feature, description):
    """
    Helper function to create a TypeError to indicate the usage of an stypy unsupported feature
    :param localization: Caller information
    :param feature: Used feature name
    :param description: Description of why this feature is unsupported
    :return: A TypeError with a custom message
    """
    unsupported_error = TypeError(localization, "Unsupported feature '{0}': '{1}'".format(feature, description))
    TypeError.usage_of_unsupported_feature = True
    return unsupported_error
