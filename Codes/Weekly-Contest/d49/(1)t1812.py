class Solution:
    def squareIsWhite(self, coordinates: str) -> bool:
        ss = ['b','d','f','h']
        nn = ['1','3','5','7']
        if coordinates[0] in ss:
            if coordinates[1] in nn:
                return True
            else:
                return False
        else:
            if coordinates[1] in nn:
                return False
            else:
                return True