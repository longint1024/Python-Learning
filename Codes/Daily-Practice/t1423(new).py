class Solution:
    def maxScore(self, cardPoints: List[int], k: int) -> int:
        n = len(cardPoints)
        s = sum(cardPoints)
        tmp = 0
        for i in cardPoints[:n-k]:
            tmp += i
        ans = tmp
        for i in range(n-k, n):
            tmp += cardPoints[i]
            tmp -= cardPoints[i-n+k]
            if tmp < ans:
                ans = tmp
        return s-ans