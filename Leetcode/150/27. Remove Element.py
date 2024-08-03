class Solution:
    def removeElement(self, nums: List[int], val: int) -> int:
        i,j=0,1
        l=len(nums)
        if l==0:
            return 0

        while(i<l and j<l):
            print("a")
            if nums[i]==val:
                if nums[j]==val:
                    j+=1
                else:
                    nums[i],nums[j]=nums[j],nums[i]
                    print(nums[i],nums[j],i,j)
                    i+=1  
                    j=i+1
            else:
                i+=1
                j=i+1
        cnt=0
        for i in nums:
            if i!=val:
                cnt+=1
        return cnt

                