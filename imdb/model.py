import os
import numpy as np

#categorizing reviews
#DATA PREPARATION

#linking to the folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#imdb_dir = 'C:\\Users\\HP\\Desktop\\AI\\Natural Language\\imdb\\'

train_dir = os.path.join(BASE_DIR, 'train')

# list consisting of 0s and 1s
labels = []
texts = []

for label_type in ['neg','pos']:
    dirname = os.path.join(train_dir, label_type)
    for filename in os.listdir(dirname):
        #selecting last 4 characters ending with .txt
        if filename[-4:] == '.txt':
            f = open(os.path.join(dirname,filename),encoding='utf8')
            texts.append(f.read())
            f.close()

            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1) 

#Tokenizer breaks the words into sentence tokens or individual words or characters
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences 

# when the words are upto 100 it should be cut
maxlen = 1000
training_samples = 200

#validation samples. The model sees it but doesnt learn from it
validation_samples = 1000 

#initializing the tokenizer
# max_words=1000 is the max number of words in the token
tokenizer = Tokenizer(num_words=1000)

#fitting the tokenizer to the texts to do the breaking into the tokens
tokenizer.fit_on_texts(texts)

#getting the tokens into sequence
sequences = tokenizer.texts_to_sequences(texts)

#getting the number of tokens
#print(len(tokenizer.word_index))
#print(len(sequences))

#sequence consists of tokens

#we are making the squence unique
# converting the sequence the actual data the computer or model will understand and it will be in a tuple
data = pad_sequences(sequences, maxlen=maxlen)

labels = np.asarray(labels)

print(data.shape)

indices = np.arange(data.shape[0])

#shuffling
np.random.shuffle(indices)

data = data[indices]
labels = labels[indices]

#Everyting till u get to the training samples of 200
features_train = data[:training_samples]
labels_train = labels[:training_samples]
validation_features = data[training_samples: validation_samples + training_samples]
validation_labels = labels[training_samples: validation_samples + training_samples]


from tensorflow.keras.models import Sequential
#Embedding serves as the input layer and convert every sequence to an integer
from tensorflow.keras.layers import Embedding, Dense, Flatten

model = Sequential([
    Embedding(1000,8, input_length=maxlen),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

#MODEL TRAINING
model.fit(features_train, labels_train, epochs=10, validation_data=(validation_features, validation_labels), batch_size=32)

print(model.evaluate(features_train, labels_train))

model.save('model.h5')

print(model.predict(features_train)[0])
print(labels_train[0])            