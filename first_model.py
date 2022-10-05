# -*- coding: utf-8 -*-
"""
Created on Fri Jan  1 14:39:44 2021

@author: carlo
"""
import os
import json
import pandas as pd
import tensorflow as tf
import numpy as np

n_exp = '05'
model_name = 'VQA'

#I set the seed
SEED = 1234
tf.random.set_seed(SEED)
np.random.seed(SEED)

#I get the current directory
cwd = os.getcwd() 
dataset_dir = os.path.join(cwd, 'VQA_Dataset')

#I read the json in a dataframe
json_train_q_a_dir = os.path.join(dataset_dir, 'train_questions_annotations.json')
with open(json_train_q_a_dir) as json_file:
    labels = json.load(json_file)
    
df = pd.DataFrame.from_dict(labels, orient='index')
df.columns = ['question', 'image_id', 'answer']
   
#I change the answer "monkey bars" to make it of 1 word only     
idx = df.index[df['answer'] == 'monkey bars']
df.loc[idx,'answer'] = 'monkeybars'

########################################################################################
#TOKENIZATION

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#I build the tokenizer for questions and answers
MAX_NUM_WORDS = 100000
answer_tokenizer = Tokenizer(num_words= MAX_NUM_WORDS)
question_tokenizer = Tokenizer(num_words=MAX_NUM_WORDS, oov_token='<UKN>')

#I extract questions and answers list
question_list = df['question'].tolist()
answer_list = df['answer'].tolist()

#I use the questions and the answers to fit the tokenizers
answer_tokenizer.fit_on_texts(answer_list)
question_tokenizer.fit_on_texts(question_list)

#I obtain answers and questions tokenized
answer_tokenized = answer_tokenizer.texts_to_sequences(answer_list)
question_tokenized = question_tokenizer.texts_to_sequences(question_list)

#I plot the histograms for answrsa and questions length frequency
import matplotlib.pyplot as plt
quest_len = np.array([len(i) for i in question_tokenized])

plt.hist(x=quest_len, bins=19, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.xlabel("Questions length")
plt.xticks([2,4,6,8,10,12,14,16,18,20])
plt.ylabel("Frequency")
plt.grid(True, axis='y')

plt.hist(x=np.array(answer_tokenized), bins=58, color='#0504aa',
                            alpha=0.7, rwidth=0.85)
plt.xlabel("Answers")
plt.ylabel("Frequency")
plt.grid(True, axis='y')

#99% percentile of questions length (it is 13)
len_99 = np.percentile(quest_len, 99)

#I build the vocabularies
answer_wtoi = answer_tokenizer.word_index
answer_itow = {v:k for k, v in answer_wtoi.items()}
question_wtoi = question_tokenizer.word_index

#sizes of the vocabularies
vocabulary_answer_size = len(answer_wtoi)+1
vocabulary_question_size = len(question_wtoi)+2   #+2 because there are padding and <UKN>

max_answer_length = max(len(sentence) for sentence in answer_tokenized)
max_question_length = max(len(sentence) for sentence in question_tokenized)
#I set the maximal question length to the 99% percentile of questions length
max_question_length = int(len_99)

#I pad the sequences
answer_encoder_inputs = pad_sequences(answer_tokenized, maxlen=max_answer_length)
question_encoder_inputs = pad_sequences(question_tokenized, maxlen=max_question_length, padding = 'pre')

################################################################################################################
#BUILD QUESTIONS ENCODER

EMBEDDING_SIZE = 256
encoder_input = tf.keras.Input(shape=[max_question_length])

#I create the embedding representation of input vector of the sentences, passing from {0,1}^N to (0,1)^m smaller representation
encoder_embedding_layer = tf.keras.layers.Embedding(input_dim=vocabulary_question_size,   #input dim, is the num of words in the dictionary + 1 (it is not MAX_NUM_WORD because in the dataset there could be less)
                                                    output_dim=EMBEDDING_SIZE,   #output dim, is the m, dim of the vector (0,1)^m represetation of the word, it is the m
                                                    input_length=max_question_length, #dimension of a input sequence
                                                    mask_zero=True)   #ignora gli zeri del padding
encoder_embedding_out = encoder_embedding_layer(encoder_input)
encoder = tf.keras.layers.LSTM(units=128, return_state=True)
#return_sequences: Boolean. Whether to return the last output. in the output sequence, or the full sequence. Default: False.
#return_state: Boolean. Whether to return the last state in addition to the output. Default: False.

encoded_question, h, c = encoder(encoder_embedding_out)

###########################################################################################################
#BUILD THE CNN

#input image size
img_h = 64
img_w = 64

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, Sequential

vision_model = Sequential()
vision_model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(img_h, img_w, 3)))
vision_model.add(Conv2D(32, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(Conv2D(256, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())
vision_model.add(Dense(128, activation='relu'))

image_input = Input(shape=(img_h, img_w, 3))
encoded_image = vision_model(image_input)

#I concatenate the two parts
merged = tf.keras.layers.concatenate([encoded_question, encoded_image])
output = Dense(vocabulary_answer_size, activation='softmax')(merged)
vqa_model = Model(inputs=[image_input, encoder_input], outputs=output)

##############################################################################################################
#IMAGE PREPROCESS
from PIL import Image

#function to load and process an image given the path
def load_and_proccess_image(image_path):
  # Load image, then scale
  im = Image.open(image_path)
  im = im.resize((img_h, img_w), resample=Image.ANTIALIAS)
  im = np.array(im)
  im = im[:,:,0:3]
  return np.float32(im / 255.0) 

#function to read the images given the paths
def read_images(paths, num_data):
  # paths is a dict mapping image ID to image path
  # Returns a dict mapping image ID to the processed image
  ims = np.zeros((num_data, img_h, img_w, 3), dtype="float32")
  i = 0
  for image_path in paths:
    ims[i,:,:,:] = load_and_proccess_image(image_path)
    i += 1
  return ims

#list containing the paths of all the images
image_list = df['image_id'].tolist()
images_dir = os.path.join(dataset_dir, 'Images')
image_list = [os.path.join(images_dir, im + '.png') for im in image_list]


########################################################################################
#Optimization Parameters
#loss
ls = tf.keras.losses.CategoricalCrossentropy()
#learning rate
lr = 1e-3
#optimizer (error descent method)
optim = tf.keras.optimizers.Adam(learning_rate=lr)
#validation metric
val_metric = ['accuracy']


#I compile the model
vqa_model.compile(optimizer=optim, loss=ls, metrics=val_metric)
print(vqa_model.summary())

######################################################################################
#CALLBACKS
#I add the callbacks

#I build the directories
from datetime import datetime

fold_name = 'VQA_experiments' + n_exp
exps_dir = os.path.join(cwd, fold_name)
if not os.path.exists(exps_dir):
    os.makedirs(exps_dir)
now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

exp_dir = os.path.join(exps_dir, model_name + '_' + str(now))
if not os.path.exists(exp_dir):
    os.makedirs(exp_dir)

callbacks = []

# #I add the model checkpoints
# ckpt_dir = os.path.join(exp_dir, 'ckpts')
# if not os.path.exists(ckpt_dir):
#     os.makedirs(ckpt_dir)
# ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
#                                                    save_weights_only=True)  # False to save the model directly
# callbacks.append(ckpt_callback)

#I visualize learning on TensorBoard
tb_dir = os.path.join(exp_dir, 'tb_logs')
if not os.path.exists(tb_dir):
    os.makedirs(tb_dir)
# By default shows losses and metrics for both training and validation
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_dir,
                                             profile_batch=0,
                                             histogram_freq=1)  # if 1 shows weights histograms
callbacks.append(tb_callback)

#I add the model checkpoints
ckpt_dir = os.path.join(exp_dir, 'ckpts')
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(ckpt_dir, 'cp_{epoch:02d}.ckpt'), 
                                                   save_weights_only=True,
                                                   monitor='val_accuracy',
                                                   save_best_only=False)  # False to save the model directly
callbacks.append(ckpt_callback)

#I implement early stopping
early_stop = True
pat = 10    #patience of the early stopping
if early_stop:
    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience = pat)
callbacks.append(tb_callback)

###########################################################################################
#I TRAIN THE MODEL

#I build class weights
from sklearn.utils import class_weight
class_weight = class_weight.compute_class_weight('balanced',
                                                 np.unique(answer_encoder_inputs[:,0]),
                                                 answer_encoder_inputs[:,0])

#Number of instances I decide to use for the training
num_data = 24000

#I convert answers in 1-hot encoded format
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(answer_encoder_inputs[0:num_data])

#I create images dataset
train_X_images = read_images(image_list[0:num_data], num_data)

#I set number of epocs, the batch size ad the validation split
num_epochs = 15
bs = 1
val_split = 0.2

#I assing to each answer the weigth
sample_weights = np.array([class_weight[i-1] for i in answer_encoder_inputs[:,0]])

#I train the model
vqa_model.fit(x=[train_X_images, question_encoder_inputs],
              y=y_train,
              epochs=num_epochs,
              batch_size = bs,
              sample_weight=sample_weights,
              validation_split=val_split,
              shuffle=True,
              callbacks = callbacks)

##########################################################################################
#predictions
# preds = vqa_model.predict(x=[train_X_images, question_encoder_inputs])

# pred_id = np.array((tf.argmax(preds, -1)))
# answer_itow = {v:k for k, v in answer_wtoi.items()}
# pred_words = [answer_itow[id] for id in pred_id]