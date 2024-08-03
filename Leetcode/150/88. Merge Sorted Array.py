class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        i,j=0,0
        nums=[]
        while(i<m and j<n):
            if nums1[i]>=nums2[j]:
                nums.append(nums2[j])
                j+=1
            else:
                nums.append(nums1[i])
                i+=1

        while(i<m):
            nums.append(nums1[i])
            i+=1
        while(j<n):
            nums.append(nums2[j])
            j+=1

        for i in range(m+n):
            nums1[i]=nums[i]
        return nums1
        