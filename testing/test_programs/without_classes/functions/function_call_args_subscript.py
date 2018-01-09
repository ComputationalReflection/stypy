dic = {"a": 1, "b": 2}
tup = ([3], dic)


def function(*args, **kwargs):
    print args
    print kwargs

function(*[3], **dic)
function(*tup[0], **tup[1])
