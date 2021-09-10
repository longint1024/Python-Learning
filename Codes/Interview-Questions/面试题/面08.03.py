class Solution:
    def findMagicIndex(self, nums: List[int]) -> int:
        n = len(nums)
        for i in range(n):
            if nums[i]==i:
                return i
        return -1