class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        count = [0 for _ in range(22)]
        ans = [[]]
        for i in nums:
            count[i+10]+= 1
        for i in range(22):
            temp = []
            for j in range(count[i]):
                add = [i-10]*(j+1)
                for s in ans:
                    tt = s[:]
                    #print(tt)
                    tt.extend(add)
                    temp.append(tt)
            ans.extend(temp)
        return ans