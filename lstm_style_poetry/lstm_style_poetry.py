
# coding: utf-8

# # Preprocessing of the poetry
# 
# ## Clean
# + Split using symbol ":", just the poetry itself
# + Sort the poetry according to its length
# + `Poetry` - cleaned data(5~79 words and no strange symbols)

# In[138]:

poetry_file = "poems.txt"
poetrys = []
with open(poetry_file, "r", encoding="utf-8") as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ','')
            if set('_(（《[') & set(content):
                continue
            if len(content) < 5 or len(content) > 79:
                continue
            content = 'B' + content + 'E'
            poetrys.append(content)
        except Exception as e:
            pass
        

print(poetrys[0])
print("lines:", len(poetrys))


# ## Dict and List Structrue for Post-use
# 
# + `words` - Sort word list in a descending way
# + `counter` - Dict {word:counts}
# + `word_int_map` - Dict {character:number}
# + Add `' '` into the `words`

# In[153]:

from collections import Counter

all_words = [word for poetry in poetrys for word in poetry]
counter = Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x:-x[1])
words, _ = zip(*count_pairs)
words = words[:len(words)] + (' ',) 
word_int_map = dict(zip(words,range(len(words))))


# ## Create peotry vector
# + `poetry_vectors` - convert words into a vector, convert all poetrys into vectors

# In[185]:

to_num = lambda word:word_int_map.get(word, len(words))
poetry_vectors = [list(map(to_num, poetry)) for poetry in poetrys]
print(poetry_vectors[0])


# ## Create batches
# + `x_batches` - the batched input
# + `y_batches` - the batched output
# + Poetry_vectors within a batch have the same length 

# In[198]:

import numpy as np

batch_size = 64
n_chunk = len(poetry_vectors) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
    start_index = i * batch_size
    end_index = start_index + batch_size
    batches = poetry_vectors[start_index:end_index]
    length = max(map(len, batches))
    xdata = np.full((batch_size, length), word_int_map[' '], np.int32)
    for row in range(batch_size):
        xdata[row, :len(batches[row])] = batches[row]
    ydata = np.copy(xdata)
    ydata[:, :-1] = xdata[:, 1:]
    """
    xdata             ydata
    [6,2,4,6,9]       [2,4,6,9,9]
    [1,4,2,8,5]       [4,2,8,5,5]
    """
    x_batches.append(xdata)
    y_batches.append(ydata)


# # Build the model

# In[ ]:



