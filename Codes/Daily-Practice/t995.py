class Solution:
    def minKBitFlips(self, A: List[int], K: int) -> int:
        N = len(A)
        que = collections.deque()
        res = 0
        for i in range(N):
            if que and i >= que[0] + K:
                que.popleft()
            if len(que) % 2 == A[i]:
                if i +  K > N: return -1
                que.append(i)
                res += 1
        return res