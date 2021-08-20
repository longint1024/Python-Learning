class Solution:
    def reverseStr(self, s: str, k: int) -> str:
        ans = list(s)
        n = len(ans)
        t = (n-1)//(2*k)
        for i in range(t):
            temp = ans[i*2*k+k-1:i*2*k:-1]
            temp.append(ans[i*2*k])
            ans[i*2*k:i*2*k+k] = temp
        temp = ans[min(n,t*2*k+k)-1:t*2*k:-1]
        temp.append(ans[t*2*k])
        ans[t*2*k:min(n,t*2*k+k)] = temp
        return "".join(ans)