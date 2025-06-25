from google.colab import drive
drive.mount('/content/drive')
    
import spacy
import random
import pandas as pd
import numpy as np
import json
from spacy.training import Example
from sklearn.metrics import precision_recall_fscore_support
     

# Load the JSONL data
train_data = []
with open('ner_data.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        text = data.get('text', '').lower()  # 'text' anahtarı yoksa boş string kullan
        # 'entities' anahtarı yoksa boş bir liste kullan
        entities = data.get('entities', [])

       # Validation and removing duplicates
        seen_spans = set()
        unique_entities = []
        for start, end, label in entities:
            if (start, end) not in seen_spans:
                seen_spans.add((start, end))
                unique_entities.append((start, end, label))

        train_data.append((text, {"entities": unique_entities}))

     

# Create a new empty spacy model.
nlp = spacy.blank("tr")


# Adding ner to nlp model.
ner = nlp.add_pipe("ner")

# Adding annotations
for _, annotations in train_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Adding optimizer for training the model
optimizer = nlp.begin_training()


# Training
for epoch in range(100):
    random.shuffle(train_data)
    losses = {}
    # Mini-batch size
    batch_size = 8
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i + batch_size]
        examples = []
        for text, annotations in batch:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        # Updating the model
        nlp.update(examples, drop=0.5, losses=losses, sgd=optimizer)

    print(f"Epoch {epoch+1} - Losses: {losses}")


save_path = '/content/drive/MyDrive/ner_model'

# Save the trained model
nlp.to_disk(save_path)

load_path = '/content/drive/MyDrive/ner_model'

nlp = spacy.load(load_path)

test_data_with_annotations = [
{"id":14994,"text":" vodafonetr  vodafone  vodafonemedya müşteri memnuniyeti kaliteli hizmet denince ilk akıllara gelen  turkcell tercih edin rahat edin sizi mağazamıza çay içmeye bekliyoruz","entities":[[1,11,"Organization"],[13,21,"Organization"],[23,36,"Service"],[66,72,"Service"],[101,109,"Organization"]],"Comments":[]},
{"id":14995,"text":" vodafonedestek bu sorunun çözülemeyeceğini benden daha iyi biliyorsunuz . şebeke elektrik kesik değilken bile zayıf kötü anlatabildim mi . bölge de çalışma var bahanesiyle oyalayacaksınız yormayın beni.  vodafonedestek şebeke konusunda  turkcell sizi sollar.","entities":[[1,15,"Service"],[75,81,"Service"],[205,219,"Service"],[220,226,"Service"],[238,246,"Organization"]],"Comments":[]},
{"id":14996,"text":" turkcell sevgili  ttdestek daha hızlı bir internet erişimi istiyorum :bouquet:","entities":[[1,9,"Organization"],[19,27,"Service"],[43,51,"Service"]],"Comments":[]},
{"id":14997,"text":" turktelekom  turkcellhizmet digitürk satmaya çalışıyorlar.","entities":[[1,12,"Organization"],[14,28,"Service"],[29,37,"Application"]],"Comments":[]},
{"id":14998,"text":"allah aşkına bu mahallenin sahibi yokmu 40 gün oldu i̇nternet yok çoluk çocuk okul açılınca ne yapacak. zeytinli mahallesi  turktelekom  turkcellhizmet  turkcell haberimiz olmadan depremmi oldu selmi oldu","entities":[[52,61,"Organization"],[124,135,"Organization"],[137,151,"Service"],[153,161,"Organization"]],"Comments":[]},
{"id":14999,"text":" turktelekom  turkcell  vodafonetr felaket boyunca hangi hizmeti verdiler ki parasını istiyorlar? hepinize yazıklar olsun!","entities":[[1,12,"Organization"],[14,22,"Organization"],[24,34,"Organization"],[57,64,"Service"]],"Comments":[]},
{"id":15000,"text":" turktelekom  turkcell mesele youtuber olunca hemen el atıyorlar değil mi? çifte standartına ittiminin sistemi","entities":[[1,12,"Organization"],[14,22,"Organization"]],"Comments":[]},
{"id":15001,"text":"hattım turkcell telefonum tek sim kartlı. vodafone ilgili hiçbir uygulamam yok. daha önce de bu uyarıyı gördüm cihazı fabrika ayarlarına sıfırladım bu 2. oldu","entities":[[0,6,"Service"],[7,15,"Organization"],[42,50,"Organization"]],"Comments":[]},
{"id":15002,"text":" ttdestek turkcell faturalı hattımı size faturasız hat olarak taşımak istiyorum. yardımcı olacak yok mu aranızda??","entities":[[1,9,"Service"],[10,18,"Organization"],[19,27,"Packet"],[28,35,"Service"],[41,50,"Packet"],[51,54,"Service"]],"Comments":[]},
{"id":15003,"text":"vodafone sınırsız internet sınırsız konuşma ve mesajlaşma paketi 200 tl ? turkcell bu şekilde bir kampanya yapmıyor sınırsız tarife ve paket fiyatlarınız çok yüksek ?","entities":[[0,8,"Organization"],[18,26,"Service"],[27,64,"Packet"],[116,131,"Packet"],[135,140,"Packet"]],"Comments":[]},
{"id":15004,"text":" ttdestek 5 gün internetsiz kal 5 gün içinde 2 kere 48 saat içinde internet sorununuz çözülecek diyorsunuz şimdi arayıp pişkin pişkin 17 kasıma kadar diyorsun türk telekom pişmanlıktır turkcell bırakıp size geçerek en büyük yanlışı yaptım","entities":[[1,9,"Service"],[67,75,"Service"],[159,171,"Organization"],[185,193,"Organization"]],"Comments":[]},
{"id":15005,"text":"3 saatlik 1 gb diye hediye mi olur yemin ederim taahüütüm bitsin turkcellin önünden bile geçmem ne hediye veriliyor kol gibi fatura geliyor sizin gibi operatörü kınıyorum vodafone haftalık 4 gb hediye veriyor siz ise 3 saatlik 1 gb farka bak ya","entities":[[65,75,"Organization"],[125,131,"Service"],[171,179,"Organization"]],"Comments":[]},

]
     

# Function to evaluate the model
def evaluate_model(nlp, test_data):
    correct_predictions = 0
    total_entities = 0
    for example in test_data:
        doc = nlp(example['text'].lower())
        gold_entities = set((start, end, label) for start, end, label in example['entities'])
        pred_entities = set((ent.start_char, ent.end_char, ent.label_) for ent in doc.ents)

        correct_predictions += len(gold_entities & pred_entities)  # Correct predictions
        total_entities += len(gold_entities)  # Total entities in data

    accuracy = correct_predictions / total_entities if total_entities > 0 else 0
    return accuracy

# Evaluate the model on the test data
accuracy = evaluate_model(nlp, test_data_with_annotations)
print(f"Model Accuracy: {accuracy:.4f}")

for example in test_data_with_annotations:
    doc = nlp(example['text'])
    print("Text:", example['text'])
    print("Predicted Entities:", [(ent.text, ent.label_) for ent in doc.ents])
    print("\n")