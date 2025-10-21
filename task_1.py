# task 1

def count_vowels(s):
    vowels = 'aeiou'
    return sum(1 for char in s.lower() if char in vowels)


# task 2

def unique_chars(s):
    return len(s) == len(set(s))


# task 3

def count_bits(n):
    return bin(n).count('1')


# task 4

def digit_multiplication_steps(n):
    if n < 10:
        return 0
    steps = 0
    while n >= 10:
        n = eval('*'.join(str(n)))
        steps += 1
    return steps


# task 5

def rmsd(v1, v2):
    if len(v1) != len(v2):
        return 0
    squared_diff = sum((x - y) ** 2 for x, y in zip(v1, v2))
    return (squared_diff / len(v1)) ** 0.5


# task 6

def mean_std(nums):
    n = len(nums)
    mean = sum(nums) / n
    variance = sum((x - mean) ** 2 for x in nums) / n
    return (mean, variance ** 0.5)


# task 7

def prime_factors(n):
    factors = []
    d = 2
    while n > 1:
        count = 0
        while n % d == 0:
            count += 1
            n //= d
        if count > 0:
            factors.append(f"{d}**{count}" if count > 1 else f"{d}")
        d += 1 if d == 2 else 2
    return ''.join(f"({f})" for f in factors)


# task 8

def network_addresses(ip, mask):
    def to_binary(s):
        return ''.join(format(int(x), '08b') for x in s.split('.'))

    def to_decimal(b):
        return '.'.join(str(int(b[i:i + 8], 2)) for i in range(0, 32, 8))

    ip_bin = to_binary(ip)
    mask_bin = to_binary(mask)
    network_bin = ''.join('0' if m == '0' else i for i, m in zip(ip_bin, mask_bin))
    broadcast_bin = ''.join('1' if m == '0' else i for i, m in zip(ip_bin, mask_bin))

    return to_decimal(network_bin), to_decimal(broadcast_bin)


# task 9

def pyramid_levels(n):
    k = 1
    total = 0
    while total < n:
        total += k * k
        if total == n:
            return k
        k += 1
    return "It is impossible"


# task 10

def is_balanced(n):
    s = str(n)
    length = len(s)
    if length == 1:
        return True
    mid = length // 2
    left = sum(int(d) for d in s[:mid - (length % 2 == 0)])
    right = sum(int(d) for d in s[mid + (length % 2 == 0):])
    return left == right


# task 11

def split_array(M, r):
    if not 0 < r < 1 or not M or not all(len(row) > 1 for row in M):
        return [], []

    classes = {}
    for row in M:
        cls = row[0]
        classes[cls] = classes.get(cls, 0) + 1

    total = len(M)
    split1_size = int(r * total)
    split2_size = total - split1_size

    result1 = []
    result2 = []
    cls_counts1 = {cls: int(r * count) for cls, count in classes.items()}  # Fixed: 'count' is from classes.items()
    cls_counts2 = {cls: count - cls_counts1[cls] for cls, count in classes.items()}  # Fixed and used cls_counts2

    for row in M:
        cls = row[0]
        if cls_counts1[cls] > 0:
            result1.append(row)
            cls_counts1[cls] -= 1
        else:
            result2.append(row)

    return result1, result2