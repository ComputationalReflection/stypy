
a = (3,4)

(b, c) = a


(test1, (Type1, instance1, _)), (test2, (Type2, instance2, _)) = [('a', (3, 4, 5)), ('b', (10.1, 11.2, 12.3))]

def getlist():
    return [('a', (3, 4, 5)), ('b', (10.1, 11.2, 12.3))]

(test1b, (Type1b, instance1b, _)), (test2b, (Type2b, instance2b, _)) = getlist()


def func():
    (test1b, (Type1b, instance1b, _)), (test2b, (Type2b, instance2b, _)) = getlist()

func()

