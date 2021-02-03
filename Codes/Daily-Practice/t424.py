class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        def c2n(c):
            return ord(c)-ord('A')
        if len(s)<=k:
            return len(s)
        slide, p = k, 0
        abc = [0 for i in range(26)]
        for i in range(k):
            abc[c2n(s[i])] += 1
        maxabc = max(abc)
        while p+slide-1<len(s)-1:
            index = c2n(s[p+slide])
            abc[index] += 1
            maxabc = max(abc[index],maxabc)
            if slide+1-maxabc>k:
                abc[c2n(s[p])] -= 1
                p += 1
            else:
                slide += 1
        return slide