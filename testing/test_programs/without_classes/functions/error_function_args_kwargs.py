def functionargs(*args):
    return args[0]  # Should warn about None


r1 = functionargs("hi")

x1 = r1.thisdonotexist()  # Unreported


def functionkw(**kwargs):
    return kwargs[0]  # Accepts anyting as key, even if we know that kwargs has always str keys


def functionkw2(**kwargs):
    return kwargs["val"]  # Accepts anyting as key, even if we know that kwargs has always str keys


r2 = functionkw(val="hi")

x2 = r2.thisdonotexist()  # Unreported

r3 = functionkw2(val="hi")

x3 = r2.thisdonotexist()  # Unreported

r4 = functionkw2(not_exist="hi")