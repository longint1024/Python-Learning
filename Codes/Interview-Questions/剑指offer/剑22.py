# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def getKthFromEnd(self, head: ListNode, k: int) -> ListNode:
        N, Now = 0, ListNode(0)
        Now = head
        while Now.next:
            Now = Now.next
            N += 1
        print(N)
        Now = head
        for i in range(N-k+1):
            Now = Now.next
        return Now