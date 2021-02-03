class DualHeap:
    def __init__(self):
        self.small, self.large = [], []
        self.delay = collections.Counter()
        self.smallSize, self.largeSize = 0, 0
    def prune(self, heap: List[int]):
        while heap:
            num = heap[0]
            if heap is self.small:
                num = -num
            if num in self.delay:
                self.delay[num] -= 1
                if self.delay[num] == 0:
                    self.delay.pop(num)
                heapq.heappop(heap)
            else:
                break
    def makeBalance(self):
        if self.smallSize > self.largeSize + 1:
            num = -self.small[0]
            heapq.heappush(self.large, num)
            heapq.heappop(self.small)
            self.smallSize -= 1
            self.largeSize += 1
            self.prune(self.small)
        elif self.smallSize < self.largeSize:
            num = self.large[0]
            heapq.heappush(self.small, -num)
            heapq.heappop(self.large)
            self.smallSize += 1
            self.largeSize -= 1
            self.prune(self.large)
    def insert(self, num:int):
        if (not self.small) or num<= -self.small[0]:
            heapq.heappush(self.small, -num)
            self.smallSize += 1
        else:
            heapq.heappush(self.large, num)
            self.largeSize += 1
        self.makeBalance()
    def erase(self, num:int):
        self.delay[num] += 1
        if num <= -self.small[0]:
            self.smallSize -= 1
            if num == -self.small[0]:
                self.prune(self.small)
        else:
            self.largeSize -= 1
            if num == self.large[0]:
                self.prune(self.large)
        self.makeBalance()
    def mid(self):
        if self.smallSize>self.largeSize:
            return -self.small[0]
        else:
            return (self.large[0]-self.small[0])/2
class Solution:
    def medianSlidingWindow(self, nums: List[int], k: int) -> List[float]:
        dq = DualHeap()
        for i in nums[:k]:
            dq.insert(i)
        ans = [dq.mid()]
        for i in range(k,len(nums)):
            dq.insert(nums[i])
            dq.erase(nums[i-k])
            ans.append(dq.mid())
        return ans