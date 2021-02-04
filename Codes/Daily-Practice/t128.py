class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        hash = set(nums)
        visited = set()
        ans = 0
        for i in nums:
            if i not in visited:
                visited.add(i)
                ll = 1
                tmp = i
                while 1:
                    tmp += 1
                    if tmp in hash:
                        ll += 1
                        visited.add(tmp)
                    else:
                        break
                tmp = i
                while 1:
                    tmp -= 1
                    if tmp in hash:
                        ll += 1
                        visited.add(tmp)
                    else:
                        break
                if ll>ans:
                    ans = ll
            else:
                continue
        return ans