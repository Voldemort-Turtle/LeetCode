import collections
from idlelib.tree import TreeNode
from typing import List, Optional


def getConcatenation(nums: List[int]) -> List[int]:
    ans = []
    for i in range(2):
        for n in nums:
            ans.append(n)
    return ans

def canFinish(self,numCourses: int, prerequisites: List[List[int]]) -> bool:
    #map each course to prereq list
    preMap={ i:[] for i in range(numCourses)}
    for crs,pre in prerequisites:
        preMap[crs].append(pre)
    #visitSet = all courses along the curr DFS path
    visitSet = set()
    def dfs(crs):
        if crs in visitSet:
            return False
        if preMap[crs] ==[]:
            return True
        visitSet.add(crs)
        for pre in preMap[crs]:
            if dfs(pre): return False
        visitSet.remove(crs)
        preMap[crs] = []
        return True
    for crs in range(numCourses):
        if not dfs(crs): return False
    return True

def findTargetSumWays(self,nums:List[int],target:int) -> int:
    dp={}#(index,total) -> # of ways
    def backtrack(i,total):
        if i == len(nums):
            return 1 if total == target else 0
        if(i,total):
            return dp[(i,total)]
        dp[(i,total)]=(backtrack(i+1,total + nums[i])) +(backtrack(i+1,total - nums[i]))
        return dp[(i,total)]
    return backtrack(0,0)

def two_sum(self,nums:List[int],target:int) -> List[int]:
    prevMap= {} # val : index
    for i, n in enumerate(nums):
        diff = target - n
        if diff in prevMap:
            return [prevMap[diff], i]
        prevMap[n] = i
    return
def lengthLIS(self,nums:List[int]) -> int:
    LIS = [1] * len(nums)
    for i in range(len(nums)- 1, -1 ,-1):
        for j in range(i +1,len(nums)):
            if nums[i] < nums[j]:
                LIS[i] = max(LIS[i],1 +LIS[j])
    return max(LIS)

def levelorder(self,root:TreeNode) -> List[List[int]]:
    res = []
    q = collections.deque()
    q.append(root)
    while q:
        qLen = len(q)
        level = []
        for i in range(qLen):
            node = q.popleft()
            if node:
                level.append(node.val)
                q.append(node.left)
                q.append(node.right)
        if level:
            res.append(level)
    return res
def searchInsert(self,nums:List[int],target:int) -> int:
    #log(n)
    l,r=0,len(nums)-1
    while(l<=r):
        mid = (l+r)//2
        if target == nums[mid]:
            return mid
        if target > nums[mid]:
            l = mid + 1
        else:
            r = mid - 1
def findDifference(self,nums1:List[int],nums2:List[int]) -> List[List[int]]:
    nums1Set, nums2Set = set(nums1), set(nums2)
    res1,res2 = [],[]
    for n in nums1: #1,2,3 3
        if n not in nums2Set:
            res1.add(n)
    for n in nums2:  # 1,2,3 3
        if n not in nums1Set:
            res2.add(n)

    return [list(res1),list(res2)]

    return l
def cloneGraph(self,node:'Node') -> 'Node':
    oldToNew = {}
    def dfs(node):
        if node in oldToNew:
            return oldToNew[node]
        copy = node(node.val)
        oldToNew[node] = copy
        for nei in node.neighbors:
            copy.neighbors.append(dfs(nei))
        return copy
    return dfs(node) if node else None

def canJump(self,nums:List[int]) -> bool:
    goal = len(nums) - 1
    for i in range (len(nums)-1,-1,-1):
        if i + nums[i] >= goal:
            goal = i
    return True if goal == 0 else False
def candy(self,ratings: List[int]) -> int:
    arr = [1] * len(ratings)
    for i in range(1,len(ratings)):
        if ratings[i-1] < ratings[i]:
            arr[i] = arr[i-1] + 1
    for i in range(len(ratings) - 2,-1,-1):
        if ratings[i]>ratings[i+1]:
            arr[i] = max(arr[i], arr[i+1] + 1)
    return sum(arr)
def shortestPathBinaryMatrix(self,grid: List[List[int]]) -> int:
    N = len(grid)
    q = collections.deque([(0,0,1)])# r c length
    visit = set((0,0))
    while q:
        r,c,length = q.popleft()
        if min(r,c) < 0 or max(r,c):
            continue


class ListNode:
    pass


def reverseList(self,head: ListNode) -> ListNode:
    prev,curr = None,head
    while curr:
        nxt = curr.next
        curr.next = prev
        prev = curr
        curr = nxt
    return prev

def reverseList(self,head:ListNode) -> ListNode:
    if not head:
        return None
    newHead = head
    if head.next:
        newHead = self.reverseList(head.next)
        head.next.next = head
    head.next = None
    return newHead
def sortList(self,head:ListNode) -> ListNode:
    if not head or head.next:
        return head
    #split the list into two halfs
    left = head
    right = self.getMid()
    tmp = right.next
    right.next = None
    right = tmp
    self.sortList(left)
    self.sortList(right)
    return self.merge(left,right)
def getMid(self,head):
    slow, fast = head,head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow
def merge(self,left,right):
    tail = dummy = ListNode()
    while list1 and list2:
        if list1.val < list2.val:
            tail.next = list1
            list1 = list1.next
        else:
            tail.next = list2
            list2 = list2.next
        tail= tail.next
    if list1:
        tail.next = list1
    if list2:
        tail.next = list2
    return dummy.next

def insertionSortList(self,head:Optional[ListNode]) -> Optional[ListNode]:
    dummy = ListNode(0,head)
    prev,cur = head,head.next
    while cur:
        if cur.val >= prev.val:
            prev,cur = cur,cur.next
            continue
        tmp = dummy
        while cur.val > tmp.next.val:
            tmp = tmp.next
        prev.next = cur.next
        cur.next = tmp.next
        tmp.next = cur
        cur = prev.next
    return dummy.next


if __name__ == '__main__':
    print(getConcatenation([1,2,1]))





