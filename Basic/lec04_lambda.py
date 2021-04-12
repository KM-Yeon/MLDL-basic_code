#Module Name : lec04_lambda.py
import pandas as pd
import numpy as np

def add(x):
    res = 10 + x
    return res

res = add(7)
print(res)
#--------------------------------
변수 = lambda x : x+10
res = 변수(6)
print(res)

df = pd.read_csv("emp.csv", index_col="empno")
# df.apply(람다식) --> 컬럼이 여러개일 때 사용 가능 => 그냥 apply 사용해라~
# df.map(람다식)   --> 컬럼이 한 개일때만 사용 가능
# df.applymap(람다식)

#--------------------------------------
#update emp set deptno='십십' where deptno=10
df['deptno'] = df['deptno'].apply(lambda x : "aaa" if x == 10 else x ) #else x: 나머지는 그대로 둬
print(df.head(10))
#--------------------------------------

def depnoCheck(deptno):
    str =""
    # if df["depto"] == 10:
    if deptno == 10:
        str = "aaa"
    else:
        str = "bbb"
    return str

df['dummy'] = df['deptno'].apply(lambda x: depnoCheck(x)) #조건식 or 함수
print(df.head(10))
#--------------------------------------
data = {
        "uid":['kim','lee','park','hong'],
        "upw":['11','22','33','44'],
        "uname":['김','리','박','홍'],
        "addr":['서울','경기','서울','인천']
        }

#1. 데이터 프레임화
df1 = pd.DataFrame(data = data)
print(df1)

#2. [람다식 사용] : 서울인 경우: 02, 인천인 경우: 032, 경기: 031 / "locde" 컬럼에 적용
#df1['locde'] = df1['addr'].apply(lambda x: "02" if x == '서울' else "032" if x == '인천' else '031')
df1['locde'] = df1['addr'].apply(lambda x: "02" if x == '서울' else "032" if x == '인천' else '031' if x =='경기' else x)

#람다식은 elif 사용 불가능
print(df1)

#3. [람다식+함수 사용] : 서울인 경우: 02, 인천인 경우: 032, 경기: 031 / "locde" 컬럼에 적용

def lonum(x):
    num = 0
    if x == '서울':
        num = '02'
    elif x == '인천':
        num = '032'
    else:
        num = '031'
    return num

df1['locde'] = df1['addr'].apply(lambda x: lonum(x))
print(df1)


