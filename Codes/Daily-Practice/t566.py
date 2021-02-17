class Solution:
    def matrixReshape(self, nums: List[List[int]], r: int, c: int) -> List[List[int]]:
        m, n = len(nums), len(nums[0])
        x, y = 0, 0
        if m*n != r*c:
            return nums
        ans = [[] for _ in range(r)]
        for i in range(r):
            for j in range(c):
                ans[i].append(nums[x][y])
                y += 1
                if y==n:
                    y = 0
                    x += 1
        return ans