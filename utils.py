#二分探索法の降順(binarysearch)
def bsearch(a, x, reverse_sorted=False):
	left, right = 0, len(a) - 1
    #下限と上限が逆転したら終了
	while left <= right:
		mid = left + (right - left) // 2
		if a[mid] == x:
			return mid
		if reverse_sorted:
			if a[mid] >= x:
				# 真ん中より右側に焦点を置く
				left = mid + 1
			else:
				right = mid - 1
		else:
			if a[mid] >= x:
				# 真ん中から左に焦点を置く
				right = mid - 1
			else:
				left = mid + 1
	return -1

def in_array(a, idx, reverse_sorted=False):
	i = bsearch(a, idx, reverse_sorted)
	if i >= 0:
		return True
	return False

def insert_sorted(a, x):
	i = 0
	while i < len(a) and x >= a[i]:
		while i < len(a) - 1 and a[i] == a[i+1]:
			i += 1
		i += 1
	a.insert(i, x)
