from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        if nums == []:
            return []
        dq, ans = deque(), []
        def clear(i):
            if dq and dq[0] == i-k:
                dq.popleft()
            while dq and nums[i]>nums[dq[-1]]:
                dq.pop()
        for i in range(k):
            clear(i)
            dq.append(i)
        ans.append(nums[dq[0]]) 
        for i in range(k, len(nums)):
            clear(i)
            dq.append(i)
            ans.append(nums[dq[0]])
        return ans