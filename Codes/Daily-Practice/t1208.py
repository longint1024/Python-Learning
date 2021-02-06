class Solution:
    def equalSubstring(self, s: str, t: str, maxCost: int) -> int:
        n = len(s)
        dif = [0 for i in range(n)]
        for i in range(n):
            dif[i] = abs(ord(s[i])-ord(t[i]))
        slide, p, cost = 0, 0, 0
        while p+slide < n:
            if cost + dif[p+slide] <= maxCost:
                cost += dif[p+slide]
                slide += 1
            else:
                p += 1
                cost += (dif[p+slide-1] - dif[p-1])
        return slide