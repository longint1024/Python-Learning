class Solution:
    def chalkReplacer(self, chalk: List[int], k: int) -> int:
        rest = k%(sum(chalk))
        for i in range(len(chalk)):
            if chalk[i]>rest:
                return i
            rest -= chalk[i]