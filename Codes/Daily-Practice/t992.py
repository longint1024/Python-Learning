class Solution:
    def subarraysWithKDistinct(self, A: List[int], K: int) -> int:
        def find(T):
            left, right, num, cnt, res, n = 0, 0, 0, collections.Counter(), 0, len(A)
            while right < n:
                if cnt[A[right]] == 0:
                    num += 1
                cnt[A[right]] += 1
                right += 1
                while num > T:
                    left += 1
                    cnt[A[left-1]] -= 1
                    if cnt[A[left-1]] == 0:
                        num -= 1
                res += right - left
                #print(left,right,res)
            return res
        return find(K)-find(K-1)