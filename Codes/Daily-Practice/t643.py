class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        s = sum(nums[:k])
        ans = s
        for i in range(k,len(nums)):
            s += nums[i]
            s -= nums[i-k]
            if s>ans:
                ans = s
        return ans/k