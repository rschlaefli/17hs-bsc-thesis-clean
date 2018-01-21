from multiprocessing import Process
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f(name):
    # info('function f')
    print('hello', name)

if __name__ == '__main__':
    print('hello')
    # info('main line')
    p = Process(target=f, args=('bob',))
    p2 = Process(target=f, args=('fred',))
    p.start()
    p2.start()
    p.join()
    p2.join()
