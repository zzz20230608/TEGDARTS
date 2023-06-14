f = open("text.txt", "r",encoding="utf-8")
s1=f.read().lower()#全小写

base="abcdefghijklmnopqrstuvwxyz"

def Encrypt(m,k):
    """加法密码加密"""
    m1=list(m)
    length=len(m1)
    for i in range(length):
        if m1[i] in base:
            x=ord(m[i])-ord('a')
            y=(x+k)%26
            m1[i]=chr(ord('a')+y)
    return "".join(m1)#转换回字符串

def strmodify():
    pass

def Decrypt(s,k):
    """加法密码解密"""
    s1=list(s)
    length=len(s1)
    for i in range(length):
        if s1[i] in base:
            x=ord(s[i])-ord('a')
            y=(x-k)%26
            s1[i]=chr(ord('a')+y)
    return "".join(s1)

s2=Encrypt(s1,2)#加密后的结果

de=[]#所有解密结果
de_dict={}
for key in range(1,26):#遍历所有可能的密钥(1-25)
    de.append(Decrypt(s2,key))
de_dict=de_dict.fromkeys(de)#创建字典
for s1 in de_dict:
    de_dict[s1]=s1.count("the")+s1.count("is")#统计is和the的次数和，作为排序

a = sorted(de_dict.items(), key=lambda x: x[1], reverse=True)#排序
n=int(input("请输入你想查看的个数(1-25):"))
for i in range(n):
    print(a[i][0])
