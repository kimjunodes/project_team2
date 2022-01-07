import random

q = list(0 for i in range(5))
qc = list(0 for i in range(5))

for i in range(20):
    num = random.randrange(1, 101)
    if i<5: #0~4까지 랜덤한수 넣기
        q[i] = chr(num)
        print(q)
    else:
        for j in range(4): #좌로 밀어주기
            qc[j] = q[j + 1]
        q[4] = chr(num)
        q = qc
        print(q)