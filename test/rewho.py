from collections import defaultdict

arr = [2,0,1,3,4]

count = defaultdict(int)
print(count,'\n')

ans = nonzero = 0

for x,y in zip(arr, sorted(arr)):
    print('begain:',ans,nonzero,x,y)
    count[x] += 1
    if count[x] == 0:
        nonzero -= 1
    if count[x] == 1:
        nonzero += 1
    
    count[y] -= 1
    if count[y] == -1:
        nonzero += 1
    if count[y] == 0:
        nonzero -= 1
    
    if nonzero == 0:
        ans += 1
    print('end   :',ans,nonzero,x,y,'\n')
    

print(ans)

"""
from collections import Counter

arr = [4,3,2,1]

count = Counter()
print(count)
counted = []

for x in arr:
    print(x)
    count[x] += 1
    counted.append((x,count[x]))

print(count,'\n')
    
ans, cur = 0, ()

for x,y in zip(counted, sorted(counted)):
    print('begain:',ans,cur,x,y)
    cur = max(cur, x)
    if cur == y:
        ans += 1
    print('end   :',ans,cur,x,y,'\n')
print('ans:',ans)
"""

"""
def maxChunksToSorted(arr):
        
        print(arr)
        i = 0
        j = len(arr)-1
        
        while i < j:
            print('Circle one begain:',i,j)
            if arr[i] > arr[i+1] and arr[i] < arr[j]:
                i += 1
            elif arr[i] > arr[j]:
                return 1
            else:
                break
            print('Circle one end:',i,j)
        while i < j:
            print('Circle two begain:',i,j)
            if arr[j] >= arr[j-1] and arr[j] >= arr[i]:
                j -= 1
            else:
                break
            print('Circle two end:',i,j)
        if i==j and j == len(arr)-1:
            return 1
        else:
            return maxChunksToSorted(arr[:i+1])+maxChunksToSorted(arr[i+1:len(arr)])

print(maxChunksToSorted([1,1,0,0,1]))
"""