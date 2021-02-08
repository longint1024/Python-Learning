class Solution:
    def numSubseq(self, nums: List[int], target: int) -> int: 
        nums.sort()
        table = [1] + [0]*(len(nums))
        M = 10**9 + 7
        for i in range(len(nums)):
            table[i+1] = table[i]*2 % M
        ans = 0
        for i in range(len(nums)):
            if nums[i]*2 > target:
                break
            j = bisect.bisect_right(nums,target-nums[i])
            ans += table[j-i-1]
            if ans>M:
                ans -= M
        return ans