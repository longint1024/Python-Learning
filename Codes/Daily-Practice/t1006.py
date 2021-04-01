class Solution:
    def clumsy(self, N: int) -> int:
        ans = 0
        if N <= 2:
            return N
        if N == 3:
            return 6
        n = N
        flag = 1
        for i in range(N//4):
            ans += flag*(n*(n-1)//(n-2))+n-3
            n -= 4
            flag = -1
        if n == 3:
            ans -= 6
        else:
            ans -= n
        return ans