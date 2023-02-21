import time

def busy_wait_1_sec():	
    i=0
    while i < 15000000:
        i += 1

def foo():
    print("Foo")
    # time.sleep(1)
    busy_wait_1_sec()
    print("Foo, slept 1sec")
    # time.sleep(2)
    busy_wait_1_sec()
    busy_wait_1_sec()
    print("Foo, slept 3sec, exiting")

def bar():
    print("Bar")
    # time.sleep(2)
    busy_wait_1_sec()
    busy_wait_1_sec()
    print("Bar, slept 2sec")
    # time.sleep(2)
    busy_wait_1_sec()
    busy_wait_1_sec()
    print("Bar, slept 4sec, exiting")