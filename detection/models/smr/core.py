import random
from detection.models.smr.sais import construct_suffix_array
from detection.models.const import ORD_UPPER_BOUND
from detection.utils import ord_cyrillic


class SuffixArray:
    def __init__(self, string):
        self.string = string + '$'
        self.sa = self._construct_suffix_array(self.string)

    def suffix_array(self):
        """Returns a suffix array."""
        return self.sa

    def match(self, pattern):
        """Returns an array of the string index where the pattern occurs."""
        pattern_len = len(pattern)

        low, high = 0, len(self.string) - 1
        while low < high:
            mid = (low + high) // 2

            suffix_index = self.sa[mid]
            pattern_is_bigger = self.string[suffix_index:suffix_index+pattern_len] < pattern

            if pattern_is_bigger:
                low = mid + 1
            else:
                high = mid

        if self.string[self.sa[low]:self.sa[low] + pattern_len] != pattern:
            return (-1, -1)
        else:
            lower_bound = low

        low, high = 0, len(self.string) - 1
        while low < high:
            mid = (low + high) // 2

            suffix_index = self.sa[mid]
            pattern_is_smaller = self.string[suffix_index:suffix_index+pattern_len] > pattern

            if pattern_is_smaller:
                high = mid
            else:
                low = mid + 1

        if self.string[self.sa[high]:self.sa[high] + pattern_len] != pattern:
            high -= 1

        upper_bound = high

        return list(sorted([self.sa[i] for i in range(lower_bound, upper_bound + 1)]))

    def longest_common_prefix(self):
        """Returns an array of longest common prefix(LCP).
        LCP[i] contains the length of common prefix between SA[i] and SA[i-1].
        """
        n = len(self.sa)
        phi = [-1] * n
        for i in range(1, n):
            phi[self.sa[i]] = self.sa[i-1]

        l = 0
        plcp = [0] * n
        for i in range(n):
            if phi[i] == -1:
                continue
            else:
                while self.string[i + l] == self.string[phi[i] + l]:
                    l += 1
                plcp[i] = l
                l = max(l-1, 0)

        return [plcp[self.sa[i]] for i in range(n)]

    def longest_repeated_substring(self):
        """Returns one of the longest repeated substrings within the string."""
        # Find index and length of the longest repeated substring.
        i, l = max(enumerate(self.longest_common_prefix()), key=lambda tup: tup[1])
        return self.string[self.sa[i]:self.sa[i]+l]

    def _construct_suffix_array(self, string):
        """Constructs suffix array in O(nlogn) time by sorting ranking pairs of suffixes."""
        string_len = len(string)
        suffix_array = list(range(string_len))
        rank_array = [ord_cyrillic(c) for c in string]

        k = 1
        # This sorting process will be repeated at most log(n) times.
        while k < string_len:
            # At first, sort suffixes with the first elements of ranking pairs.
            suffix_array = self._sort(suffix_array, rank_array, string_len, k)
            # Next, sort suffixes with the second elements of ranking pairs.
            suffix_array = self._sort(suffix_array, rank_array, string_len, 0)
            # Recompute rank of suffixes.
            rank_array = self._rerank(suffix_array, rank_array, k)
            k *= 2

        return suffix_array

    def _sort(self, suffix_array, rank_array, string_len, k):
        """Sorts suffixes by count-sorting rank array.
        Offset k is defined such that the value used when sorting suffix i corresponds to rank_array[i + k].
        """
        max_length = max(2**8 - 1, string_len)
        count = [0] * max_length

        for i in range(len(rank_array)):
            if i + k < string_len:
                count[rank_array[i + k]] += 1
            else:
                count[0] += 1

        cumsum = 0
        for i in range(max_length):
            t = count[i]
            count[i] = cumsum
            cumsum += t

        temp_suffix_array = [-1] * string_len
        for i in range(len(suffix_array)):
            if suffix_array[i] + k < string_len:
                target_index = rank_array[suffix_array[i] + k]
            else:
                target_index = 0

            temp_suffix_array[count[target_index]] = suffix_array[i]
            count[target_index] += 1

        return temp_suffix_array

    def _rerank(self, suffix_array, rank_array, k):
        """Recomputes rank of suffixes. When consecutive suffixes with identical ranking pairs are found,
        assigns same ranks to them.
        """
        temp_rank_array = [0] * len(rank_array)
        r, s = rank_array, suffix_array

        rank = 0
        for i in range(1, len(rank_array)):
            # When ranking pairs are identical, do not increment the rank.
            if r[s[i]] == r[s[i-1]] and r[s[i] + k] == r[s[i-1] + k]:
                temp_rank_array[s[i]] = rank
            else:
                rank += 1
                temp_rank_array[s[i]] = rank

        return temp_rank_array
