global_var = "hi"

def joinseq(seq):
    if len(seq) == 1:
        return '(' + seq[0] + ',)'
    else:
        return '(' + ', '.join(seq) + ')'


def strseq(object, convert, join=joinseq):
    if type(object) in [list, tuple]:
        return join([strseq(_o, convert, join) for _o in object])
    else:
        return convert(object)


def foo(par):
    return '*' + par

def foo2():
    par = None
    return '*' + par

def formatargspec(args, varargs=None, varkw=None, defaults=None,
                 formatarg=str,
                 formatvarargs=lambda name: '*' + name,
                 formatvarkw=lambda name: '**' + name,
                 formatvalue=lambda value: '=' + repr(value),
                 join=joinseq):
    specs = []
    if defaults:
        firstdefault = len(args) - len(defaults)
        for i in range(len(args)):
            spec = strseq(args[i], formatarg, join)
            if defaults and i >= firstdefault:
                spec = spec + formatvalue(defaults[i - firstdefault])
            specs.append(spec)
            if varargs is not None:
                specs.append(formatvarargs(varargs))
                foo(varargs)
                foo2()
            if varkw is not None:
                specs.append(formatvarkw(varkw))
        return '(' + ', '.join(specs) + ')'


r = formatargspec(('a', 'b'), None, None, (3, 4))
print r
