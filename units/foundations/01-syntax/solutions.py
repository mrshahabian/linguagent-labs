"""Reference answers for 01 - syntax, values and control flow."""


def classify_number(n):
    """Return "negative", "zero" or "positive"."""
    if n < 0:
        return "negative"
    elif n == 0:
        return "zero"
    else:
        return "positive"


def fizzbuzz(n):
    """Return ["1", "2", "Fizz", "4", "Buzz", ...] up to and including n."""
    out = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


def count_vowels(word):
    """Count the vowels (a, e, i, o, u, either case) in word."""
    count = 0
    for ch in word:
        if ch in "aeiouAEIOU":
            count += 1
    return count


def grade_for_score(score):
    """Map a 0-100 score to a letter grade: A >= 90, B >= 80, C >= 70,
    D >= 60, else F."""
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"


def sum_up_to(n):
    """Sum every integer from 1 to n inclusive, using a while loop."""
    total = 0
    i = 1
    while i <= n:
        total += i
        i += 1
    return total


def is_leap_year(year):
    """A year is a leap year if divisible by 4, unless it's also divisible
    by 100 - in which case it isn't, unless it's also divisible by 400."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
