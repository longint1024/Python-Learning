class Solution:
    def areSentencesSimilar(self, sentence1: str, sentence2: str) -> bool:
        if len(sentence1)<len(sentence2):
            s1, s2 = sentence2, sentence1
        else:
            s1, s2 = sentence1, sentence2
        l1, l2 = s1.split(' '), s2.split()
        n1, n2 = len(l1), len(l2)
        left, right, right1 = -1, n2, n1
        while left <= n2-2 and l1[left+1] == l2[left+1]:
            left += 1
        while right >0 and l1[right1-1] == l2[right-1]:
            right -= 1
            right1 -= 1
        #print(left, right)
        if left+1>=right:
            return True
        else:
            return False