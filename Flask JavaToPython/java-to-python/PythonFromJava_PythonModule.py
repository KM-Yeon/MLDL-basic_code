import sys

def call_for_java() :
    print ("java param " + sys.argv[1] + " " + sys.argv[2])

if __name__ == '__main__':
    call_for_java()