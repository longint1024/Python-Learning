class Solution:
    def countNicePairs(self, nums: List[int]) -> int:
        M = 10**9+7
        def rev(n:int) ->int:
            temp = 0
            while n>0:
                temp *= 10
                temp += (n % 10)
                n = n // 10
            return temp
        count = collections.Counter()
        for i in range(len(nums)):
            count[nums[i] - rev(nums[i])] += 1
        ans = 0
        for i in count:
            k = count[i]
            ans += k*(k-1)//2
            ans = ans % M
        return ans