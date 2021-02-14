class Solution:
    def checkInclusion(self, s1: str, s2: str) -> bool:
        n1, n2 = len(s1), len(s2)
        if n1>n2:
            return False
        cnt, slide = [0 for _ in range(26)], [0 for _ in range(26)]
        def num(c):
            return ord(c) - ord('a')
        def judge(a,b):
            res = 0
            for i in range(len(a)):
                res += abs(a[i]-b[i])
            return res
        for c in s1:
            cnt[num(c)] += 1
        for c in s2[:n1]:
            slide[num(c)] += 1
        res = judge(slide,cnt)
        if res == 0:
            return True
        p = 0 
        for r in range(n1,n2):
            slide[num(s2[r-n1])] -= 1
            slide[num(s2[r])] += 1
            if judge(slide, cnt) == 0:
                return True
        return False