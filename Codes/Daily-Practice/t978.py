class Solution:
    def maxTurbulenceSize(self, arr: List[int]) -> int:
        if len(arr) == 1:
            return 1
        p, slide, ans, n = 0, 1, 1, len(arr)
        for i in range(n-1):
            if arr[i] == arr[i+1]:
                slide = 1
                continue
            if slide == 1:
                flag = (arr[i]<arr[i+1])
                slide = 2
            else:
                if flag == (arr[i]<arr[i+1]):
                    slide = 2
                    continue
                else:
                    flag = not flag
                    slide += 1
            if slide>ans:
                ans = slide
        return ans