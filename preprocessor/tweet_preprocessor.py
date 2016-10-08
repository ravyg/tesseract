#!/usr/bin/env python
# -*- coding: utf-8 -*- 
import HTMLParser
import re
# import contractions as c
import acronyms as a
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class tweet_preprocessor:
  """First Tweet Text processing/Cleaning Class"""

  # Initila object.
  def __init__(self, tweet_text):
    # super(ClassName, self).__init__()
    self.tweet_text = tweet_text

  # Preprosessor.  
  def preprocessor(self):
    # Excaping HTML characters.
    html_parser = HTMLParser.HTMLParser()
    tweet = html_parser.unescape(self.tweet_text)
    tweet = re.sub(r'@[^\s]+','Author', tweet)
     # Remove repeated Authors and other words.
    tweet = re.sub(r'\b(.+)\s+\1\b', r'\1', tweet)
    # Decoding data.
    tweet = tweet.decode("utf8").encode('ascii','ignore')

    # Some more processing.
    tweet = translate_contractions(tweet)
    tweet = remove_link_text(tweet)
    tweet = http_cleaner(tweet)
    tweet = translate_acronyms(tweet)
    tweet = translate_smileys(tweet)
    tweet = translate_punctuation(tweet)
    tweet = translate_whitespace(tweet)
    tweet = translate_shorthand(tweet)
    tweet = translate_numbers_simple(tweet)
    tweet = translate_ordinals(tweet)
    tweet = translate_unicode(tweet)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    #tweet = pattern.sub(' ', tweet)
    #tweet = re.sub(r'[^\x00-\x7F]+',' ', tweet)
    # Character repitation to 2 characters max.
    # @TODO: fix character repeat to max 2.
    tweet = re.sub(r'(.)\1\1+', r'\1\1', tweet)
    
    # repeated characters fix.
    tweet = re.sub('[.]+', '. ', tweet)
    tweet = re.sub('[ ]+', ' ', tweet)
    # for metamap.
    tweet = remove_all_punctuation(tweet)
    # @TODO: Fix this causes words to stick.
    #tweet = ignore_mm_special(tweet)
    return tweet


def remove_short_words(text, length=3):
    """Removes all words with length < 'length' param"""
    text = [w for w in text.split() if len(w) >= length]
    return " ".join(text)

def remove_link_text(text):
    """Attempts to match and remove hyperlink text"""
    text = re.sub(r"\S*https?://\S*", "", text)
    return text

def http_cleaner(text):
    """Removes mysteriously hanging 'http' instances"""
    text = re.sub(r"^http ", " ", text)
    text = re.sub(r" http ", " ", text)
    text = re.sub(r" http$", " ", text)
    return text

def translate_punctuation(text):
    """Translates some punctuation (not apostrophes) from text. Returns cleaned string"""
    PUN = a.punctuations
    for p in PUN:
        text = text.replace(p, PUN[p])
    return text

def remove_all_punctuation(text, keep_hashtags=False, keep_mentions=False):
    """
    Does a strict remove of anything that is not a character in the Unicode character
    properties database, any whitespace char, or optionally a hashtag or mention symbol. 
    Works now with flags=re.U (ie, Unicode) on any non-word char for most languages.
    """
    if keep_mentions and keep_hashtags:
        return re.sub(r"[^\w\s@#]", "", text, flags=re.U)
    elif keep_mentions:
        return re.sub(r"[^\w\s@]", "", text, flags=re.U)
    elif keep_hashtags:
        return re.sub(r"[^\w\s#]", "", text, flags=re.U)
    else:
        return re.sub(r"[^\w\s]", "", text, flags=re.U)

def basic_tokenize(text, lower=True, keep_hashtags=True, keep_mentions=True):
    """
    Basic space-and-punctuation-based tokenization of a string. Removes all non-word
    characters (can keep hashtag and mention characters optionally), returns list of
    resulting space-separated tokens. Optionally lower-cases (default true)
    """
    if lower:
        return remove_all_punctuation(text.lower(),
                                      keep_hashtags=keep_hashtags,
                                      keep_mentions=keep_mentions).split()
    else:
        return remove_all_punctuation(text,
                                      keep_hashtags=keep_hashtags,
                                      keep_mentions=keep_mentions).split()

def get_cleaned_tokens(text, lower=True, keep_hashtags=True, keep_mentions=True, rts=False,
    mts=False, https=False, stopwords=[]):
    """
    Tokenization function specially for cleaning tweet tokens. 
    All parameters represent "keep" parameters:
        lower, keep_hashtags, and keep_mentions are passed to 'basic_tokenize()'
        rts: keep token 'rt' if true, else discard
        mts: keep token 'mt' if true, else discard
        https: keep any token containing 'http' if true, else discard
    """
    tokens = basic_tokenize(text, lower, keep_hashtags, keep_mentions)
    if not rts:
        tokens = [t for t in tokens if t != "rt"]
    if not mts:
        tokens = [t for t in tokens if t != "mt"]
    if not https:
        tokens = [t for t in tokens if not re.search(r"http", t)]
    if stopwords:
        tokens = [t for t in tokens if t not in stopwords]
    return tokens

def remove_RT_MT(text):
    """Removes all hanging instances of 'RT' and 'MT'. NOTE: Expects lower case"""
    text = re.sub(r" rt ", " ", text)
    text = re.sub(r"^rt ", " ", text)
    text = re.sub(r" rt$", " ", text)

    text = re.sub(r" mt ", " ", text)
    text = re.sub(r"^mt ", " ", text)
    text = re.sub(r" mt$", " ", text)
    return text

def clean_whitespace(text):
    """Alternate method of cleaning whitespace"""
    return " ".join(text.split())

def translate_whitespace(text):
    """Replaces any non-single-space whitespace chars (tabs, newlines..). Returns cleaned string"""
    WTS = a.whitespaces
    for w in WTS:
        text = text.replace(w, WTS[w])
    return text

def translate_smileys(text):
    """Replaces any non-single-space whitespace chars (tabs, newlines..). Returns cleaned string"""
    SMY = a.smileys
    for smy in SMY:
        text = text.replace(smy, ' '+SMY[smy]+' ')
    return text 

def ignore_mm_special(text):
    """Replaces any non-single-space whitespace chars (tabs, newlines..). Returns cleaned string"""
    MM = a.ignore_mm
    for mm in MM:
        text = text.replace(mm, MM[mm])
    return text        

def csv_safe(text):
    """Makes text CSV-safe (no commas, tabs, newlines, etc). Returns cleaned string"""
    text = clean_whitespace(text)
    for c in csvsafe_trans:
        text = text.replace(c, csvsafe_trans[c])
    return text

def translate_shorthand(text):
    """Translate common shorthand terms into long form. Returns translated string"""
    STH = a.shorthands
    for s in STH:
        text = text.replace(s, STH[s])
    return text

def translate_numbers_simple(text):
    """Naive translation of digit characters to corresponding number words. No scaling."""
    NUM = a.numbers
    for key, rep in NUM.items():
        text = text.replace(key, rep)
    return text

def translate_ordinals(text):
    """Translate ordinal digit strings up two twelve(fth)"""
    ORD = a.ordinals
    for key, rep in ORD.items():
        text = re.sub(" {0}".format(key), " {0}".format(rep), text)
        text = re.sub("^{0} ".format(key), "{0} ".format(rep), text)
        text = re.sub(" {0}$".format(key), " {0}".format(rep), text)
    return text

def translate_acronyms(text):
    """Translate common acronyms. WARNING: watch out for words (use spaces). Returns translated string"""
    # acronym_trans is a (re, replacement) tuple
    ACR = a.acronyms
    for ac in ACR:
        text = ac[0].sub(ac[1], text)
    return text

def translate_unicode(text):
    """Translate some unicode characters into better equivalents. Returns translated string"""
    UCD = a.unicodes
    for u in UCD:
        text = text.replace(u, UCD[u])
    return text

def translate_contractions(text):
    """Translates contractions with expanded forms. Returns translated string"""
    CON = a.contractions
    for cn in CON:
        text = text.replace(cn, CON[cn])
    return text

def remove_stopwords(text):

    """
    Standard way to remove stopwords from text. Text is a string. Stopwords is
    a list of strings to remove. Returns cleaned string.
    """
    # text = " " + text + " "
    # for w in stopwords:
    #   text = text.replace(u" {0} ".format(w.decode("utf8")), u" ")
    # return text.strip()
    filtered_words = [word for word in text if word not in stopwords.words('english')]
    return filtered_words    

def remove_digit_words(text):
    """
    Remove all space-separated substrings of only digits. Return cleaned string.
    """
    return " ".join([w for w in text.split() if not w.isdigit()])

def remove_user_mentions(text):
    """
    Removes @mentions from tweets so that "hello @shlomo how are you?" becomes "hello how are you?."
    """
    text = re.sub(r"\S*@\w+\S*", "", text)
    return text
