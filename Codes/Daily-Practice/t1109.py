class Solution:
    def corpFlightBookings(self, bookings: List[List[int]], n: int) -> List[int]:
        diff, ans = [0]*n, [0]*n
        for i in bookings:
            diff[i[0]-1] += i[2]
            if i[1]<n:
                diff[i[1]] -= i[2]
        ans[0] = diff[0]
        for i in range(1,n):
            ans[i] = ans[i-1]+diff[i]
        return ans