class Solution:
    def longestOnes(self, A: List[int], K: int) -> int:
        left, right, cnt, ans = 0, 0, 0, 0
        while right<len(A):
            while right<len(A) and cnt<K:
                if not A[right]:
                    cnt += 1
                right += 1
            while right < len(A) and A[right]:
                right += 1
            if right - left > ans:
                ans = right - left
            if right ==  len(A) - 1:
                return ans
            while left<len(A) and A[left]:
                left += 1
            left += 1
            cnt -= 1
        return ans