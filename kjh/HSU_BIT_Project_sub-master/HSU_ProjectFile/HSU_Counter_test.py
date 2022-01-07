import time
import random
import numpy as np

q = list(0 for i in range(5))
qc = list(0 for i in range(5))

for i in range(20):

    if i<5:
        num1 = random.randrange(1, 101)
        q[i] = chr(num1)
        print(q)
    else:
        if q[4] == 0:
            num2 = random.randrange(1, 101)
            q[4] = chr(num2)
            print(q)
        elif q[4] != 0:
            for j in range(4):
                qc[j] = q[j+1]
            q[4] = 0
            q = qc

            print(q)
    time.sleep(0.3)


#for i in range(q.qsize()):
    #print(q.get(i))

#if q[4] != 0:
#    print('된다')
#else:
#    print('이상하다')