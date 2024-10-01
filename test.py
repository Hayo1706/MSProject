def alternate_last_four(list1, list2):
    # Get the last 4 elements of each list, padded with 0s
    last_four_list1 = (list1[-4:] + [0]*4)[:4]  # Last 4 elements or padded with 0
    last_four_list2 = (list2[-4:] + [0]*4)[:4]  # Last 4 elements or padded with 0

    # Combine them in alternating order
    result = []
    for a, b in zip(last_four_list1, last_four_list2):
        result.append(a)
        result.append(b)

    return result

list1 = [1, 2, 3]
list2 = [11, 12, 13]

print(alternate_last_four(list1, list2))  # Output: [5, 15, 6, 16, 7, 17, 8, 18]