class Solution:
    def getRow(self, rowIndex: int) -> List[int]:
        if rowIndex == 0:
            return [1]
        ans = [1,1]
        if rowIndex == 1:
            return ans
        for i in range(rowIndex-1):
            tmp1 = 1
            for j in range(1,len(ans)):
                tmp2 = ans[j]
                ans[j] += tmp1
                tmp1 = tmp2
            ans.append(1)
        return ans