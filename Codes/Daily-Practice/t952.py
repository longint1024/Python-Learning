class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n
        self.n = n
        # 当前连通分量数目
        self.setCount = n
    
    def findset(self, x: int) -> int:
        if self.parent[x] == x:
            return x
        self.parent[x] = self.findset(self.parent[x])
        return self.parent[x]
    
    def unite(self, x: int, y: int) -> bool:
        x, y = self.findset(x), self.findset(y)
        if x == y:
            return False
        if self.size[x] < self.size[y]:
            x, y = y, x
        self.parent[y] = x
        self.size[x] += self.size[y]
        self.setCount -= 1
        return True
    
    def connected(self, x: int, y: int) -> bool:
        x, y = self.findset(x), self.findset(y)
        return x == y
class Solution:
    def largestComponentSize(self, A: List[int]) -> int:
        maxA = max(A)
        uf = UnionFind(maxA+1)
        setA = set(A)
        mA = maxA//2
        prime = [1 for i in range(mA+1)]
        r = floor(sqrt(len(prime)))+2
        for i in range(2,r):
            if prime[i]:
                for j in range(2,mA//i+1):
                    prime[j*i] = 0
        p = []
        for i in range(2,len(prime)):
            if prime[i]:
                p.append(i)
        for i in range(len(p)):
            for j in range(1,maxA//p[i]+1):
                if j*p[i] in setA:
                    for k in range(j+1,maxA//p[i]+1):
                        if k*p[i] in setA:
                            uf.unite(j*p[i],k*p[i])
                    break
        return max(uf.size)