

def createGenerator():
    mylist = range(3)
    for i in mylist:
       yield


x = createGenerator()
for e in x:
    print e

