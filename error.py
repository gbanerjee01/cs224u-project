def wer(original, result):
    r"""
    The WER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more words (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the WER will always be 1 (N / N = 1).
    """
    # The WER is calculated on word (and NOT on character) level.
    # Therefore we split the strings into words first:
    original = original.split()
    result = result.split()
    return levenshtein(original, result) / float(len(original))

def cer(original, result):
    r"""
    The CER is defined as the editing/Levenshtein distance on word level
    divided by the amount of words in the original text.
    In case of the original having more chars (N) than the result and both
    being totally different (all N words resulting in 1 edit operation each),
    the CER will always be 1 (N / N = 1).
    """
    # The CER is calculated on character (and NOT on word) level.
    # Therefore we split the strings into characters first:
    original = list(original)
    result = list(result)
    return levenshtein(original, result) / float(len(original))

def ers(originals, results, func=wer):
    count = len(originals)
    rates = []
    mean = 0.0
    assert count == len(results)
    for i in range(count):
        rate = func(originals[i], results[i])
        mean = mean + rate
        rates.append(rate)
    return rates, mean / float(count)

# The following code is from: http://hetland.org/coding/python/levenshtein.py

# This is a straightforward implementation of a well-known algorithm, and thus
# probably shouldn't be covered by copyright to begin with. But in case it is,
# the author (Magnus Lie Hetland) has, to the extent possible under law,
# dedicated all copyright and related and neighboring rights to this software
# to the public domain worldwide, by distributing it under the CC0 license,
# version 1.0. This software is distributed without any warranty. For more
# information, see <http://creativecommons.org/publicdomain/zero/1.0>

def levenshtein(a,b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n,m)) space
        a,b = b,a
        n,m = m,n

    current = list(range(n+1))
    for i in range(1,m+1):
        previous, current = current, [i]+[0]*n
        for j in range(1,n+1):
            add, delete = previous[j]+1, current[j-1]+1
            change = previous[j-1]
            if a[j-1] != b[i-1]:
                change = change + 1
            current[j] = min(add, delete, change)

    return current[n]