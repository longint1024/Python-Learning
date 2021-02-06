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
    def minimumEffortPath(self, heights: List[List[int]]) -> int:
        m, n = len(heights), len(heights[0])
        dx = [0,1]
        dy = [1,0]
        def judge(x,y):
            if x<0 or y<0 or x>m-1 or y>n-1:
                return False
            return True
        def num(x,y):
            return y+x*n
        edge = []
        for i in range(m):
            for j in range(n):
                for k in range(2):
                    if judge(i+dx[k],j+dy[k]):
                        edge.append([abs(heights[i][j]-heights[i+dx[k]][j+dy[k]]),num(i,j),num(i+dx[k],j+dy[k])])
        edge.sort(key=lambda x:x[0])
        target = num(m-1,n-1)
        uf = UnionFind(target+1)
        if uf.connected(0,target):
            return 0
        for c,i,j in edge:
            uf.unite(i,j)
            if uf.connected(0,target):
                return c