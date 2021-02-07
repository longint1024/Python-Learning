class Solution:
    def checkPossibility(self, nums: List[int]) -> bool:
        p, n = 0, len(nums)
        while p < n-1 and nums[p] <= nums[p+1]:
            p += 1
        if p >= n-2:
            return True
        p += 2
        if nums[p] < nums[p-1]:
            return False
        if nums[p] < nums[p-2]:
            if p>=3 and nums[p-1] < nums[p-3]:
                return False
        while p < n-1 and nums[p] <= nums[p+1]:
            p += 1
        return p == n-1