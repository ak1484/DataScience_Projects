class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        x=nums[0]
        i=0
        for j in range(1,len(nums)):
            if nums[j]!=x:
                i+=1
                nums[i]=nums[j]
                x=nums[j]
                
        return i+1
