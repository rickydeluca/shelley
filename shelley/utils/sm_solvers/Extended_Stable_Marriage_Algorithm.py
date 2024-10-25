import numpy as np

class ExtendedStableMarriage:
    def __init__(self,malechoice,femalechoice,n,k):

        self.malechoice = malechoice 
        self.femalechoice = femalechoice 
        self.fc = np.random.randint(10, size=(k, n+1))
        self.count = 0
        self.male_counter=np.empty(n+1,dtype=int)
        self.marriage=np.empty(k,dtype=int)
        self.MOSM(n, k)

    def MOSM(self,n,k):
        for i in range (k):
            for j in range(n):
                self.fc[i,self.femalechoice[i,j]]=j
                self.marriage[i]=0
                self.fc[i,0]=n+1
        for i in range(n+1):
            self.male_counter[i]=0
        for i in range(n+1):
            self.Proposal(i,n,k)
        
        return self.marriage


    def Proposal(self,i,n,k):
        if i!=0 and self.male_counter[i]<k:
            self.count = self.count+1
            j = self.male_counter[i]
            self.male_counter[i] = j+1
            self.Refusal(i,self.malechoice[i,j],n,k)

    def Refusal(self,i,j,n,k):
        if self.fc[j,self.marriage[j]]>self.fc[j,i]:
            l=self.marriage[j]
            self.marriage[j]=i
            self.Proposal(l,n,k)
        else:
            self.Proposal(i,n,k)

    def Get_Male_Counter(self):
        return self.male_counter

