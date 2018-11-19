import re
import string

import nltk


def get_bug_vector_comment(item):
    title = clean_item(item['title'])
    desc = clean_item(item['summary'])

    comments = []

    for comment in item['comments']:
        comments += clean_item(comment)

    return title + desc + comments


def get_bug_vector_list(item):
    # 1. Remove \r
    current_title = item['title'].replace('\r', ' ')
    current_desc = item['summary'].replace('\r', ' ')
    # 2. Remove URLs
    current_desc = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '', current_desc)
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title = re.sub(r'(\w+)0x\w+', '', current_title)
    # 5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    # 7. Strip trailing punctuation marks
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]
    # 8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = [x for x in current_data if x]

    return " ".join(current_data)


def get_bug_vector(item):
    # 1. Remove \r
    current_title = item['title'].replace('\r', ' ')
    current_desc = item['summary'].replace('\r', ' ')
    # 2. Remove URLs
    current_desc = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '', current_desc)
    # 3. Remove Stack Trace
    start_loc = current_desc.find("Stack trace:")
    current_desc = current_desc[:start_loc]
    # 4. Remove hex code
    current_desc = re.sub(r'(\w+)0x\w+', '', current_desc)
    current_title = re.sub(r'(\w+)0x\w+', '', current_title)
    # 5. Change to lower case
    current_desc = current_desc.lower()
    current_title = current_title.lower()
    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(current_desc)
    current_title_tokens = nltk.word_tokenize(current_title)
    # 7. Strip trailing punctuation marks
    current_desc_filter = [word.strip(string.punctuation) for word in current_desc_tokens]
    current_title_filter = [word.strip(string.punctuation) for word in current_title_tokens]
    # 8. Join the lists
    current_data = current_title_filter + current_desc_filter
    current_data = [x for x in current_data if x]

    return " ".join(current_data)


def clean_item(item):
    item = item.replace('\r', ' ')
    # 2. Remove URLs
    item = re.sub(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        '', item)
    # 3. Remove Stack Trace
    start_loc = item.find("Stack trace:")
    item = item[:start_loc]
    # 4. Remove hex code
    item = re.sub(r'(\w+)0x\w+', '', item)
    # 5. Change to lower case
    item = item.lower()
    # 6. Tokenize
    current_desc_tokens = nltk.word_tokenize(item)
    # 7. Strip trailing punctuation marks
    filtered = [word.strip(string.punctuation) for word in current_desc_tokens]

    return [x for x in filtered if x]


def get_bug_vector_from(title, summary):
    item = {
        'title': title,
        'summary': summary
    }

    return get_bug_vector_list(item)
