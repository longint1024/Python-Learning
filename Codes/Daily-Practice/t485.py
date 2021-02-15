class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        ans, MAX = 0, 0
        for i in nums:
            if i == 1:
                ans += 1
                if ans > MAX:
                    MAX = ans
            else:
                ans = 0
        return MAX