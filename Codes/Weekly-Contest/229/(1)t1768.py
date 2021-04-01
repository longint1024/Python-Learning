class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        m, n = len(word1), len(word2)
        ans = ''
        for i in range(min(m,n)):
            ans += word1[i]
            ans += word2[i]
        if m<n:
            ans += word2[m:n]
        else:
            ans += word1[n:m]
        return ans