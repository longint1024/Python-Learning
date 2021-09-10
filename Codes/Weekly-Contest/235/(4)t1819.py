class Solution:
    def countDifferentSubsequenceGCDs(self, nums: List[int]) -> int:
        s = set(nums)
        MAX = -1
        for i in s:
            if i>MAX:
                MAX = i
        ans = len(s)
        def gcd(a:int,b:int)->int:
	        return gcd(b,a%b) if b else a
        for i in range(1,MAX):
            if i in s:
                continue
            q = []
            for j in range(2,MAX//i+1):
                if i*j in s:
                    q.append(j)
            flag = 0
            if len(q)>1:
                tt = q[0]
                for k in range(1,len(q)):
                    tt = gcd(tt,q[k])
                    if tt == 1:
                        flag = 1
                        break
            if flag:
                ans += 1
        return ans