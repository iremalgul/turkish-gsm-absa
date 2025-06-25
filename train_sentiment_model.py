# GEREKLI KUTUPHANELERIN EKLENMESI

import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import torch

# CSV DOSYASININ OKUNMASI

dframe = pd.read_csv("/content/sentiment_dataset_final.csv", encoding='utf_8', on_bad_lines = 'skip', sep = ';')
dframe

# TITLE VE LINK SUTUNLARININ TABLODAN CIKARILMASI

#dframe = dframe.iloc[300:]
#dframe = dframe.drop(columns="Title",axis = 1)
#dframe = dframe.drop(columns="Link",axis=1)
#dframe = dframe.sample(frac = 1)
dframe

# TABLODAKI DUYGULARIN SAYISI

print("0: Olumsuz", "1: Olumlu", "2: Nötr\n")
dframe["Target"].value_counts()

# METINDEN NOKTALAMA ISARETLERINI, RAKAMLARI CIKARMA

def clean_text(text):
    text = " ".join(str(text).split())
    text = text.lower()
    text = text.replace("\\n", " ")
    text = re.sub("[0-9]+", "", text)
    text = re.sub("%|(|)|-", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return "".join(text)

dframe["Clean"] = dframe.apply(lambda row: clean_text(row["Explanation"]), axis=1)
dframe

# SATIRLARI NUMPY DIZISI YAPMA

X = dframe["Clean"]
y = dframe["Target"]

# TEST VE EGITIM VERILERININ AYRILMASI

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
X_train_data, X_val, y_train_data, y_val = train_test_split(X_train, y_train, test_size=0.5)

# BERT TOKENIZER EKLEME

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

# BERT TOKENIZER UYGULAMA

max_len= 1024

X_train_encoded = tokenizer.batch_encode_plus(X_train_data.tolist(),
                                              padding=True,
                                              truncation=True,
                                              max_length = max_len,
                                              return_tensors='tf')

X_val_encoded = tokenizer.batch_encode_plus(X_val.tolist(),
                                              padding=True,
                                              truncation=True,
                                              max_length = max_len,
                                              return_tensors='tf')

X_test_encoded = tokenizer.batch_encode_plus(X_test.tolist(),
                                              padding=True,
                                              truncation=True,
                                              max_length = max_len,
                                              return_tensors='tf')

# SINIFLANDIRMA MODELINI YUKLEME

model = TFBertForSequenceClassification.from_pretrained('dbmdz/bert-base-turkish-cased', num_labels = 3)

# MODELI COMPILE ETME

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# MODELI EGITME

history = model.fit(
    [X_train_encoded['input_ids'], X_train_encoded['token_type_ids'], X_train_encoded['attention_mask']],
    y_train_data,
    validation_data=(
      [X_val_encoded['input_ids'], X_val_encoded['token_type_ids'], X_val_encoded['attention_mask']],y_val),
    batch_size=16,
    epochs=5
)

# EVALUATE MODEL

test_loss, test_accuracy = model.evaluate(
    [X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']],
    y_test
)
print(f'Test loss: {test_loss}, Test accuracy: {test_accuracy}')

# MODELI KAYDETME

path = 'sentiment_model'
# Save tokenizer
tokenizer.save_pretrained(path +'/Tokenizer')

# Save model
model.save_pretrained(path +'/Model')

# MODELI YUKLEME

# Load tokenizer
bert_tokenizer = BertTokenizer.from_pretrained(path +'/Tokenizer')

# Load model
bert_model = TFBertForSequenceClassification.from_pretrained(path +'/Model')

### DENEME

pred = bert_model.predict(
	[X_test_encoded['input_ids'], X_test_encoded['token_type_ids'], X_test_encoded['attention_mask']])

# pred is of type TFSequenceClassifierOutput
logits = pred.logits

# Use argmax along the appropriate axis to get the predicted labels
pred_labels = tf.argmax(logits, axis=1)

# Convert the predicted labels to a NumPy array
pred_labels = pred_labels.numpy()

label = {
	0: 'Olumsuz',
	1: 'Olumlu',
  2: 'Nötr'
}

# Map the predicted labels to their corresponding strings using the label dictionary
pred_labels = [label[i] for i in pred_labels]
Actual = [label[i] for i in y_test]

print('Predicted Label :', pred_labels[:10])
print('Actual Label :', Actual[:10])

print("Classification Report: \n", classification_report(Actual, pred_labels))

# KULLANICI GIRISI ILE TAHMIN FONKSIYONU

def Get_sentiment(Review, Tokenizer=bert_tokenizer, Model=bert_model):
	# Convert Review to a list if it's not already a list
	if not isinstance(Review, list):
		Review = [Review]

	Input_ids, Token_type_ids, Attention_mask = Tokenizer.batch_encode_plus(Review,
																			padding=True,
																			truncation=True,
																			max_length=1024,
																			return_tensors='tf').values()
	prediction = Model.predict([Input_ids, Token_type_ids, Attention_mask])

	# Use argmax along the appropriate axis to get the predicted labels
	pred_labels = tf.argmax(prediction.logits, axis = 1)

	# Convert the TensorFlow tensor to a NumPy array and then to a list to get the predicted sentiment labels
	pred_labels = [label[i] for i in pred_labels.numpy().tolist()]
	return pred_labels

# KULLANICI GIRISI ILE TAHMIN FONKSIYONU

Review = "Merhaba. Vodafone nin sundugu bu hizmetlerden cok memnum kaldim. Daha once diger operatorleri denemistim ve sorunlar yasamistim. Vodafone ye gectikten sonra her sey duzeldi. Akrabalarıma da vodafone ye gecmeli konusunda ısrarcı olacagim. Fiyatlari cazip ve guzel. Tesekkurler."
Get_sentiment(Review)

Review = "Fiyatlar cok ucuz. Cok guzel cekiyor. Tavsiye ediyorum."
Get_sentiment(Review)

Review = "1 haftadır arama yapamıyorum. Faturalarımı ödememe ragmen sebekeye baglanılamıyor hatasi aliyorum. Bu sorunu lutfen en erken vakitte cozun. Ticari is yapiyorum. Cok zarardayim."
Get_sentiment(Review)

Review = "Vodafone yi cok seviyorum. Cok memnun kaldim. Cok tesekkur ederim. İyi ki varsiniz."
Get_sentiment(Review)

Review = "Cok iyi sevdim baya bunu. tesekkurler vodafone"
Get_sentiment(Review)

Review = "kotu berbat."
Get_sentiment(Review)

Review = "Vodafone nin sundugu hizmetlerden cok memnum kaldim. Diger operatorlerden daha iyi daha guzel. Vodafone ye gecmek hayatimda aldigim en guzel karardi. Akrabalarima da vodafone ye gecmeleri konusunda ısrarcı olacagim. Fiyatlari cazip ve guzel. Tesekkur ederim."
Get_sentiment(Review)

Review = "Bir daha asla vodafone kullanmam."
Get_sentiment(Review)

Review = "ne iyi ne de kotu. Kararsizim"
Get_sentiment(Review)