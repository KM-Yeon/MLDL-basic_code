#module name : lec02_pandas_numpy.py

import numpy as np
# 1. array ndtype [123]
# 2. List [1,2,3] -> 괄호 안에 점이 있으면 리스트로 땅땅..이므로 뒤에 , 없어도 리스트로 인식
# 3. Series Series([1,2,3,])
# 4. a=[1,] list -> 괄호 안에 점이 있으면 리스트
# 5. a=[1] array
#6. list = [[7733, 'kim', '111'], [8855, 'lee', '222']] -> 행렬(matrix), array 아님

vlist = [1,2,3]
print(vlist)

vlist = [[1,2,3],
         [1,2,3]
         ]

#리스트는 shape 없고 len 가능
print(len(vlist), len(vlist[0]))

#shape을 찍고 싶으면 array로 바꿔야 한다.
varr = np.array(vlist)
print(varr)

vmatrix = [[1,2,3], ['a', 'b', 'c']] #리스트 안에 리스트 넣은 것
varr2 = np.array(vmatrix)
print(varr2)
print(varr2.shape)
print(varr2[1][2]) #c뽑기

#list[] tupl() set dict{}6
# tupl = (1,2,3)     --> 한번 넣으면 변경 불가능
# list = [1,2]
# set  = (1,1,1,2,3) --> 중복 제거

dict = [{"id":"kim", "pw":"111"}, #리스트 안에 딕셔너리 넣은 것
        {"id":"park", "pw":"222"}]

print(dict)
print(dict[1]["pw"])

dict = {"record1" : {"id":"kim", "pw":"111"},
        "record2" : {"id":"park", "pw":"222"}}
print(type(dict))
print(dict["record2"]["pw"])

#데이터 프레임 --> 데이터 구조를 표 형태로 만듦
import pandas as pd
s = pd.Series([1,2,3])
s2 = pd.Series(['a','b','dfdf'])
print(s)
print(s.shape)
print(s.dtype) #int64, float64
print(s2.dtype) #글자는 object

#데이터 프레임 문법
# def __init__(
#         self,
#         data=None,
#         index: Optional[Axes] = None,
#         columns: Optional[Axes] = None,
#         dtype: Optional[Dtype] = None,
#         copy: bool = False,
#     ):
#data: dict np list series
list = [[7733, 'kim', '111'], [8855, 'lee', '222']]
df = pd.DataFrame(data=list,
                  columns=['eno', 'id', 'pw'], )
print(df)
print(df.shape)
print(df.head())


