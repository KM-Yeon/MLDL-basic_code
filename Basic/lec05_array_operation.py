import numpy as np
a = [1,2,3]
b = [4,5,6]
c = np.array(a)
d = np.array(b)

#방법들 중 하나만 정확하게 알면 된다.

#옆으로 합치기 [1,2,3] + [4,5,6]
print(np.r_[a,b])                    #[1 2 3 4 5 6]     #여기서 r은 read라 생각
print(np.hstack([a,b]))              #[1 2 3 4 5 6]
print(np.concatenate((a,b), axis=0)) #[1 2 3 4 5 6]
print(np.array(a+b))                 #[1 2 3 4 5 6]
print(a+b)                           #[1, 2, 3, 4, 5, 6] --> 리스트가 돼버림

# 위아래로 합치기 [1,2,3] +
#             [4,5,6]
print(np.r_[[a],[b]])
print(np.vstack([a,b]))
#print(np.concatenate((c,d), axis=1)) #---> 행렬로 와야 T가 가능하므로 오류

#세로줄 합치기
#[1,   4]
#[3,   5]
#[1,   2]
# matrix = [[1, 2, 3], [4, 5, 6]]
# print(np.array(matrix))
# print(np.array(matrix).T) #<--********
# print(np.array(matrix).transpose())
#
# print(np.c_[a,b])
# print(np.column_stack([a,b]))
# print(np.concatenate(c.T, d.T), axis=1)



