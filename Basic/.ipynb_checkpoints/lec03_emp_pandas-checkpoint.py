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

#null np.NAN is null
#select ename, comm from emp where comm is null;
# print(df[df[['ename','comm']].isnull()])
# print(df[df['comm'].isnull()])
# print(df['comm'][df['comm'] == np.nan] )
# print(df[['ename','commm']][df['comm'] == np.nan] )
print(df[df['comm'].isnull()][['ename', 'comm']])
print(df[df['comm'].isnull()==False][['ename', 'comm']]) #not null인 경우

#select count(1) from emp where comm is null;
#order by --> sort_values()
#count(1) --> value_counts()
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





