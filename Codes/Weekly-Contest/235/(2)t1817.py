class Solution:
    def findingUsersActiveMinutes(self, logs: List[List[int]], k: int) -> List[int]:
        usingt = collections.Counter()
        ans = [0 for _ in range(k)]
        t = [[] for i in range(100004)]
        for i in logs:
            if not (i[0] in t[i[1]]):
                t[i[1]].append(i[0])
        for k in t:
            for j in k:
                usingt[j] += 1
        for i in usingt:
            ans[usingt[i]-1] += 1
        return ans