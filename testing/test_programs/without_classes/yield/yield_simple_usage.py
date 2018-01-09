

def createGenerator():
    mylist = range(3)
    for i in mylist:
       yield i*i


x = createGenerator()

