class Solution:
    def minAbsoluteSumDiff(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        M = 10**9 + 7
        score = [0 for _ in range(n)]
        for i in range(n):
            score[i] = abs(nums1[i]-nums2[i])
        MAX = 0
        nums1.sort()
        def found(k,l,r):
            if r-l<3:
                MIN = 10000000
                for i in range(l,r+1):
                    if abs(nums1[i]-k)<MIN:
                        MIN = abs(nums1[i]-k)
                return MIN
            else:
                mid = (l+r)//2
                if k==nums1[mid]:
                    return 0
                if k<nums1[mid]:
                    return found(k,l,mid)
                if k>nums1[mid]:
                    return found(k,mid,r)
        for i in range(n):
            diff = found(nums2[i],0,n-1)
            if score[i]-diff>MAX:
                MAX = score[i]-diff
        ans = -MAX
        for i in score:
            ans += i
            if ans>M:
                ans -= M
        return ans