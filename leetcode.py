class Solution(object):
    def findKthNumber(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: int
        """
        if n<10:
           return k


        digits=str(n)
        n_digits=len(digits)
        start_digit=int(digits[0])
        output=""
        if start_digit*10**(n_digits-1)==k:
            return k
        elif start_digit*10**(n_digits-1)>k:
            output=str(start_digit)+str(self.findKthNumber(int(digits[1:]),k))
            return int(output)
        else:
            output=self.findKthNumber(start_digit*10**(n_digits-1),k)
            return output

if __name__ == '__main__':
    sln=Solution()
    n=13
    k=2
    result=sln.findKthNumber(n,k)
    print(result)