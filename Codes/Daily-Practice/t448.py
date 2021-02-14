class Solution:
    def findDisappearedNumbers(self, nums: List[int]) -> List[int]:
        n = len(nums)
        for i in nums:
            nums[(i-1)%n] += n
        ans = []
        for i in range(n):
            if nums[i]<=n:
                ans.append(i+1)
        return ans