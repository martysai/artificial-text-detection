def construct_suffix_array(string):
    ta = construct_type_array(string)
    mapping = construct_alphabet_mapping(string)
    es = encode_string(string, mapping)
    buckets = construct_buckets(es, mapping)

    sa = initialize_sa_with_lms_guess(buckets, es, ta)
    sa = l_type_induced_sorting(sa, buckets, es, ta)
    sa = s_type_induced_sorting(sa, buckets, es, ta)

    summarized_string, original_indices = summarize(sa, es, ta)
    summarized_suffix_array = summary_suffix_array(summarized_string)

    sa = final_lms_sort(es, buckets, ta, summarized_suffix_array, original_indices)
    sa = l_type_induced_sorting(sa, buckets, es, ta)
    sa = s_type_induced_sorting(sa, buckets, es, ta)

    return sa


def initialize_sa_with_lms_guess(buckets, string, ta):
    sa = [-1] * len(string)
    tails = get_bucket_tails(buckets)

    for i in range(len(sa)):
        if is_lms_character(ta, i):
            c = string[i]
            sa[tails[c]] = i
            tails[c] -= 1

    return sa


def l_type_induced_sorting(sa, buckets, string, ta):
    heads = get_bucket_heads(buckets)

    for i in range(len(sa)):
        if sa[i] == -1 or sa[i] == 0:
            continue

        if ta[sa[i] - 1] == 'L':
            c = string[sa[i] - 1]
            sa[heads[c]] = sa[i] - 1
            heads[c] += 1

    return sa


def s_type_induced_sorting(sa, buckets, string, ta):
    tails = get_bucket_tails(buckets)

    for i in range(len(sa) - 1, -1, -1):
        if sa[i] == -1 or sa[i] == 0:
            continue

        if ta[sa[i] - 1] == 'S':
            c = string[sa[i] - 1]
            sa[tails[c]] = sa[i] - 1
            tails[c] -= 1

    return sa


def summarize(sa, string, ta):
    summarized_array = [-1] * len(string)

    summarized_array[len(string) - 1] = 1
    last_lms_index = sa[0]
    current_name = 1
    for i, sa_entry in enumerate(sa[1:], 1):
        if not is_lms_character(ta, sa_entry):
            continue

        if are_equal_lms_substrings(string, ta, last_lms_index, sa_entry):
            summarized_array[sa_entry] = current_name
        else:
            current_name += 1
            summarized_array[sa_entry] = current_name
        last_lms_index = sa_entry

    summarized_array.append(0)

    return [s for s in summarized_array if s != -1], [i for i, s in enumerate(summarized_array) if s != -1]


def summary_suffix_array(summarized_string):
    alphabet_size = len(set(summarized_string))
    if alphabet_size == len(summarized_string):
        # Bucket sorting.
        summarized_suffix_array = [-1] * (len(summarized_string))

        for i, c in enumerate(summarized_string):
            summarized_suffix_array[c] = i

    else:
        summarized_suffix_array = construct_suffix_array(summarized_string)

    return summarized_suffix_array


def final_lms_sort(string, buckets, ta, summarized_suffix_array, original_indices):
    sa = [-1] * len(string)
    tails = get_bucket_tails(buckets)

    for i in range(len(summarized_suffix_array) - 1, 0, -1):
        index_at_string = original_indices[summarized_suffix_array[i]]

        c = string[index_at_string]
        sa[tails[c]] = index_at_string

        tails[c] -= 1

    return sa


def construct_type_array(string):
    type_array = ['S']
    for i in range(len(string) - 2, -1, -1):
        if (string[i] < string[i + 1]) or (string[i] == string[i + 1] and type_array[-1] == 'S'):
            type_array.append('S')
        else:
            type_array.append('L')

    return list(reversed(type_array))


def is_lms_character(type_array, i):
    if i == 0:
        return False
    elif type_array[i] == 'S' and type_array[i - 1] == 'L':
        return True
    else:
        return False


def are_equal_lms_substrings(string, type_array, i1, i2):
    if i1 == len(string) or i2 == len(string):
        return False

    for offset, (c1, c2) in enumerate(zip(string[i1:], string[i2:])):
        is_lms_1 = is_lms_character(type_array, i1 + offset)
        is_lms_2 = is_lms_character(type_array, i2 + offset)

        if offset != 0 and is_lms_1 and is_lms_2:
            return True

        if (string[i1 + offset] != string[i2 + offset]) or (is_lms_1 != is_lms_2):
            return False


def construct_alphabet_mapping(string):
    alphabets = list(sorted(list(set([c for c in string]))))
    return dict([(c, i) for i, c in enumerate(alphabets)])


def encode_string(string, mapping):
    return [mapping[c] for c in string]


def construct_buckets(encoded_string, mapping):
    alphabet_size = len(mapping)

    buckets = [0] * alphabet_size
    for i in encoded_string:
        buckets[i] += 1
    return buckets


def get_bucket_heads(buckets):
    heads = [0]
    for bucket_size in buckets[:-1]:
        heads.append(heads[-1] + bucket_size)
    return heads


def get_bucket_tails(buckets):
    tails = [0]
    for bucket_size in buckets[1:]:
        tails.append(tails[-1] + bucket_size)
    return tails
