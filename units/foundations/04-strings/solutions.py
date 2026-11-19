"""Reference answers for 04 - strings and basic text processing."""


def shout(text):
    """Return text in all caps, with "!!!" appended."""
    return text.upper() + "!!!"


def count_words(sentence):
    """Count the words in sentence, splitting on whitespace."""
    return len(sentence.split())


def reverse_words(sentence):
    """Reverse the order of the words in sentence (not the letters).
    "the cat sat" -> "sat cat the"."""
    words = sentence.split()
    words.reverse()
    return " ".join(words)


def is_palindrome(text):
    """True if text reads the same forwards and backwards, ignoring case
    and spaces. "Never Odd or Even" -> True."""
    cleaned = text.lower().replace(" ", "")
    return cleaned == cleaned[::-1]


def initials(full_name):
    """"Ada Lovelace" -> "AL" - the first letter of each word, uppercase,
    no separators."""
    return "".join(word[0].upper() for word in full_name.split())


def slugify(title):
    """"Hello World" -> "hello-world" - lowercase, spaces become hyphens."""
    return title.lower().replace(" ", "-")
