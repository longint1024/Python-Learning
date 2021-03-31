class Solution:
    def findShortestSubArray(self, nums: List[int]) -> int:
        visited = {}
        for i in range(len(nums)):
            if nums[i] not in visited:
                visited[nums[i]] = [1,i]
            else:
                visited[nums[i]][0] += 1
        for i in range(len(nums)-1,-1,-1):
            if len(visited[nums[i]]) == 2:
                visited[nums[i]].append(i)
        ans, MAX = 10000000, 0
        for t in visited:
            k = visited[t]
            if k[0] > MAX:
                MAX = k[0]
                ans = k[2]-k[1]
            elif k[0] == MAX:
                if k[2]-k[1]<ans:
                    ans = k[2]-k[1]
        return ans+1