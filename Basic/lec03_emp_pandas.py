import numpy as np
import pandas as pd

df = pd.read_csv("emp.csv", index_col="empno")
print(df.info())
print(df.shape)
print(df.head())

aa = df[["ename"]] # = df.ename = df[["ename"]]
#aa = df[["ename", "sal"]] --> 괄호가 2개면 복수개 뽑을 수 있다.
print(aa)

#select job||ename sal from emp;
print(df["job"] + df["ename"])

#select sal, sal*10 from emp;
print(df['sal']*10)

list = [1,2,3,4,5]
df1 = pd.DataFrame(data=list, columns=["no"])
print(df1)

#   uid  upw
# 0 kim  11
# 1 lee  22
# 2 park 33

#list->dataframe
dict = {"uid":["kim", "lee", "park"],
        "upw":[11, 22, 33]}
df2 = pd.DataFrame(data=dict)
print(df2)

#array->dataframe
list = [1,2,3,4,5]
arr = np.array(list)
df3 = pd.DataFrame(data=arr, columns=["no"])
print(df3)

#series->dataframe
ss = pd.Series(["aa", "bb", "cc"])
print(ss)
df2['uname'] = ss
print(df2)

#select sal, ename from emp where deptno=10
dd = df[["sal", "ename"]] [ df["deptno"]==10 ]#조건절은 []안에 넣어라
print(dd)
print(df[ df["deptno"]==10] [["sal", "ename"]]) #조건절이 앞에 오는걸 더 많이 씀

#select sal, ename from emp where deptno=10 order by ename asc
print(df[["sal", "ename"]].sort_values(by="ename"))
print(df[["sal", "ename"]].sort_values(by="ename", ascending=False))

print(df[["sal", "ename"]] [ df["deptno"]==10 ].sort_values(by="ename", ascending=False))
print(df[df["deptno"]==10] [["sal", "ename"]].sort_values(by="ename", ascending=False))

dd['SC'] = df['sal'] + df['comm']
print(dd)

#null np.NAN is null --> 주석 처리한 부분은 오류 발생
#select ename, comm from emp where comm is null;
# print(df[df[['ename','comm']].isnull()])
# print(df[df['comm'].isnull()])
# print(df['comm'][df['comm'] == np.nan] )
# print(df[['ename','commm']][df['comm'] == np.nan] )
print(df[df['comm'].isnull()][['ename', 'comm']])
print(df[df['comm'].isnull()==False][['ename', 'comm']]) #not null인 경우

#select count(1) from emp where comm is null;
#order by --> sort_values()
#count(1) --> value_counts() // *은 한 줄씩 천천히 세는 것, 1은 한 줄씩 빠르게 씀 => 대용량에서는 1 씀
print(df['ename'][df['comm'].isnull()])
print(df['comm'].isnull().value_counts())

#Group 함수:
# count(1) --> value_counts()
# max      --> max()
# min      --> min()
# avg      --> mean()
# sum      --> sum()
#group by  --> groupby()
#select deptno, avg(sal) from emp
#group by deptno
#--order by deptno;
print(df.groupby(by = 'deptno', sort=True)['sal'].mean())
# = print(df.groupby(by = 'deptno')['sal'].mean())

#select deptno, min(sal), max(sal)
#from emp
#group by deptno;
# dic = {'sal':'max', 'sal':'min'}
# print(df.groupby(by="deptno").agg(dic))
print(df.groupby(by="deptno")['sal'].agg(['max', 'min']))
df.groupby(by='deptno')

#delete from emp where ename = 'smith'
#delete --> drop

#[가로]레코드 삭제: axis = 0, index = 7499
#[세로]job컬럼 삭제 : columns = ['job', 'mgrno']

#====== drop test 후 주석 처리
#delete from emp where empno=7499;
# dfcp = df.drop(axis=0, index=7499) #defalut가 inplace = False라 원본은 그대로임
# #dfcp = df.drop(axis=0, index=7499, inplace = True)
# print(dfcp)

df.drop(axis=1, columns=["job", "mgrno"], inplace=True)
print(df)

#delete from emp where ename='smith'; --> 가로줄은 무조건 index로 접근! 세로줄은 무조건 column으로 접근
idx = df[df['ename'] == 'SMITH'].index
df.drop(index = idx, axis=0, inplace=True) #index 생략 가능
print(df)

#### np.non --> NaN (결측치 데이터 제거*******)
dfcp = df
dfcp.dropna(inplace=True)
#dfcp['comm'].dropna(inplace=True)
print(dfcp)

dfcp = df
print(dfcp.isnull().sum())

#slicing : iloc[인덱스] loc[값]

#iloc[행, 열]  iloc[2,4]
#iloc[s:e, s:e]  iloc[0:3, 0:2]
# s<=I<e # e = -1의 경우 맨 끝을 의미
#loc[인덱스값, 컬럼명]
#loc[[,,,], s:e]

list = [[1,2,3,4], [5,6,7,8]]
print(len(list), len(list[0]))
arr = np.array(list)
print(arr.shape)

#8개 값을 4 덩어리 --> 1덩어리에 2개씩
arr = arr.reshape(4,-1) #자동계산된 어떠한 값을 의미 = reshape = -1 => 딱 떨어지지 않으면 에러 남
print(arr)

#8개 값을 ?덩어리 --> 1덩어리에 2개씩
arr = arr.reshape(-1,2)
print(arr.shape)
print(arr)


#------------------------insert
# insert into emp values(9999, 'MYNAME', 'JJJ', '', '', '', '', '')
list = ['MYNAME', 'JJJ', '', '', '']
col = df.columns
print(col)
#idx = df.index

#------------
#기존 list 데이터를 DataFrame[신규]로 만들고 합쳐야 하는 불편함
#Dataframe[기존] + Dataframe[신규]
insdf = pd.DataFrame(data=[list], index=[9999], columns=col)
print(insdf)
df = df.append(insdf) #, ignore_index=True)
print(df)

insdf = pd.DataFrame(data=[list], columns=col)
#df = df.reset_index() #새로운 인덱스가 생기고 기존 인덱스는 컬럼 안으로 들어간다.
print(df)

#dict로 넣기
dic = {"ename":"DICNAME", "JOB":"JJJ"}
df = df.append(dic, ignore_index=True)
print(df)

#loc /iloc으로 넣기
df.loc["9898"] = ['MYNAME', 'JJJ', '', '', '', '']
print(df)
#------------------------update
#------------------------lambda update
#------------------------map apply applymap
mylam = lambda x: x+10
print(mylam(6))

#------------------------ 선형대수 : 행렬
#------------------------ .T 전치(Transpose): 피벗화 = 행렬 바꾸다
