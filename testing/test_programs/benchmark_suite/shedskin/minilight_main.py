from ml import minilight
import os

'''
  Copyright (c) 2008, Harrison Ainsworth / HXA7241 and Juraj Sukop.
  http://www.hxa7241.org/minilight/
'''


def Relative(path):
    return os.path.join(os.path.dirname(__file__), path)


class MinilightRun:
    def main(self):
        minilight.main(Relative('ml/cornellbox.txt'))

    def run(self):
        self.main()
        return True


def run():
    m = MinilightRun()
    m.run()


run()
