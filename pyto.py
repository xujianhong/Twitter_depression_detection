from transformers import RobertaTokenizerFast
from transformers import TFRobertaModel
#keras
import tensorflow as tf
from tensorflow import keras

# !pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

df = pd.read_csv("/root/autodl-tmp/remotefolder/cleaned_tweet_10k.csv")
df['class'].value_counts()

texts = np.array(df['tweet'].astype(str))
labels = np.array(df['class'])

train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"train_texts length :{len(train_texts)}")
print(f"test_texts length :{len(test_texts)}")
print(f"train_labels length :{len(train_labels)}")
print(f"test_labels length :{len(test_labels)}")

ohe = OneHotEncoder(sparse_output=False)
train_labels = ohe.fit_transform(np.array(train_labels).reshape(-1,1))
test_labels = ohe.fit_transform(np.array(test_labels).reshape(-1,1))

print(f"Training data: {train_texts.shape[0]}\n Validation Data: {test_texts.shape[0]}")

print(f"Training labels: {train_labels.shape}\n Validation labels: {test_labels.shape}")

tokenizer_roberta = RobertaTokenizerFast.from_pretrained('roberta-base')

token_lens = []

for txt in train_texts:
    tokens = tokenizer_roberta.encode(txt, max_length=512,truncation=True)
    token_lens.append(len(tokens))

max_length = np.max(token_lens)
max_length

MAX_LEN = 128
def tokenize_roberta(data, max_len=MAX_LEN):
    input_ids = []
    attention_masks = []
    for i in range(len(data)):
        encoded = tokenizer_roberta.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
    return  np.array(input_ids),np.array(attention_masks)

train_input_ids, train_attention_masks = tokenize_roberta(train_texts,MAX_LEN)
val_input_ids, val_attention_masks = tokenize_roberta(test_texts, MAX_LEN)

def create_model(bert_model, max_len=MAX_LEN):

    
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5, decay=1e-7)
    loss = tf.keras.losses.CategoricalCrossentropy()
    accuracy = tf.keras.metrics.CategoricalAccuracy()

    input_ids = tf.keras.Input(shape=(max_len,),dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,),dtype='int32')
    output = bert_model([input_ids,attention_masks])
    output = output[1]
    output = tf.keras.layers.Dense(4, activation=tf.nn.softmax)(output)
    model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)
    model.compile(opt, loss=loss, metrics=accuracy)
    return model

roberta_model = TFRobertaModel.from_pretrained('roberta-base')

model = create_model(roberta_model, MAX_LEN)
model.summary()

history = model.fit([train_input_ids,train_attention_masks],train_labels,
                    validation_data=([val_input_ids,val_attention_masks],test_labels),
                    epochs=4,batch_size=30)

result_roberta = model.predict([val_input_ids,val_attention_masks])

def conf_matrix(y, y_pred, title):
    fig, ax =plt.subplots(figsize=(5,5))
    labels=["ADHD", "BIPO", "CTRl", "DEP"]
    ax=sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Blues", fmt='g', cbar=False, annot_kws={"size":25})
    plt.title(title, fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=15) 
    ax.yaxis.set_ticklabels(labels, fontsize=15)
    ax.set_ylabel('Test', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.show()

y_pred_roberta = np.zeros_like(result_roberta)
y_pred_roberta[np.arange(len(y_pred_roberta)),result_roberta.argmax(1)] =1

conf_matrix(test_labels.argmax(1), y_pred_roberta.argmax(1)," Roberta Analysis\n Confusion Matrix")
print('\tClassification Report for RoBERTa:\n\n',classification_report(test_labels,y_pred_roberta, target_names=["ADHD", "BIPOLAR", "CONTROL", "DEPRESSION"]))

