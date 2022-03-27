import re
import string
from torch import clamp
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = 'cahya/bert-base-indonesian-522M'


class TokenSimilarity():

    def __init__(self, from_pretrained: str):
        self.tokenizer = AutoTokenizer.from_pretrained(from_pretrained)
        self.model = AutoModel.from_pretrained(from_pretrained)

    def __process(self, first_token: str, second_token: str):
        inputs = self.tokenizer([first_token, second_token],
                                max_length=self.max_length,
                                truncation=self.truncation,
                                padding=self.padding,
                                return_tensors='pt')

        attention = inputs.attention_mask
        outputs = self.model(**inputs)
        embeddings = outputs[0]
        mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
        masked_embeddings = embeddings * mask

        summed = masked_embeddings.sum(1)
        counts = clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts

        return mean_pooled.detach().numpy()

    def predict(self, first_token: str, second_token: str, max_length: int = 40,
                truncation: bool = True, padding: str = "max_length"):
        self.max_length = max_length
        self.truncation = truncation
        self.padding = padding

        mean_pooled_arr = self.__process(first_token, second_token)
        similarity = cosine_similarity([mean_pooled_arr[0]], [mean_pooled_arr[1]])

        return similarity

import torch
import dill
from nltk.tokenize import WordPunctTokenizer
import nltk
import math
import string
import pandas as pd
import re
import transformers
from transformers import BertModel, BertTokenizer
from torch import nn
import warnings
warnings.filterwarnings("ignore", 'This pattern has match groups')

class BertClassifier(nn.Module):

    def __init__(self, n_classes, dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = self.bert = BertModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        output = self.drop(pooled_output)
        logits = self.out(output)
        classifier = torch.nn.functional.softmax(logits, dim=1)
        _, pred = torch.max(classifier, dim=1)
        return logits, pred


class IdiomIdentification():

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.class_names = ['kalimat_biasa', 'kalimat_idiom']
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.classification_model = BertClassifier(len(self.class_names))
        self.classification_model.load_state_dict(torch.load('model/classification.bin'))
        self.classification_model = self.classification_model.to(self.device)
        self.hmm_tagger_model = dill.load(open('model/tagger_model.dill', 'rb'))
        self.similarity_model = TokenSimilarity(MODEL_NAME)
        self.truth_discovery_model = dill.load(open('model/truth_discovery.dill', 'rb'))
        self.idiom_example_df = pd.read_csv('data/IDENTIFIKASI_KLASIFIKASI/idiom-example.csv')

    def text_preprocessing(self, kalimat, remove_punctuation=False, tokenization=False, lowercase=False):
        if (remove_punctuation):
            punc = '''!()-[]{};:'"\<>/?@#$%^&*_~'''
            kalimat = kalimat.translate(str.maketrans('', '', punc))
            kalimat = re.sub(r'/s+', ' ', kalimat).strip()

        if (tokenization):
            word_punct_tokenizer = WordPunctTokenizer()
            kalimat = word_punct_tokenizer.tokenize(kalimat)

        if (lowercase):
            kalimat = str.lower(kalimat)

        return kalimat

    def idiom_sentence_classification(self, kalimat):
        encoded_text = self.tokenizer.encode_plus(
            kalimat,
            max_length=40,
            add_special_tokens=True,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoded_text['input_ids'].to(self.device)
        attention_mask = encoded_text['attention_mask'].to(self.device)
        token_type_ids = encoded_text['token_type_ids'].to(self.device)

        output, pred = self.classification_model(input_ids, attention_mask, token_type_ids)

        kategori = self.class_names[pred]

        return kategori

    def hasNumbers(self, inputString):
        result = False
        for char in list(inputString):
            if (char.isdigit()):
                result = True
        return result

    def check_tag(self, word, tag):
        punc = list(string.punctuation)
        punc.append('.')
        punc.append(',')
        punc.append('"')
        punc.append("'")

        dates = ['Januari', 'Februari', 'Maret', \
                 'April', 'Mei', 'Juni', 'Juli', 'Agustus', \
                 'September', 'Oktober', 'November', 'Desember', \
                 'Jan', 'Feb', 'Mar', 'Apr', 'Mei', \
                 'Jun', 'Jul', 'Agt', 'Sep', 'Okt', 'Nov', 'Des', \
                 'januari', 'februari', 'maret', 'april', \
                 'mei', 'juni', 'juli', 'agustus', \
                 'september', 'oktober', 'november', 'desember', \
                 'Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu'
                 ]

        if (word in dates):
            tag = 'DATE'

        if (word in punc):
            tag = 'Z'

        if (tag == 'CD' and word.isdigit()):
            tag = 'CD'

        if (tag in ['SYM', 'Z', 'CD', 'MD'] and word.upper() != word and self.hasNumbers(word) == False \
                and word[-3:] not in ['nya', 'kah', 'lah']):
            tag = 'NNP'

        if (tag == 'NN' and word[:1].upper() == word):
            tag = 'NNP'

        if (tag == 'NNP' and word.lower() == word):
            tag = 'NN'

        if (tag == 'NNP' and len(word) == 1):
            tag = 'NN'

        if (tag == 'FW' and word.lower() == word):
            tag = 'NN'

        return word, tag

    def pos_tagging(self, kalimat):
        kalimat_token = self.text_preprocessing(kalimat, tokenization=True)
        tagging = self.hmm_tagger_model.tag(kalimat_token)
        final_tag = []
        for pt in tagging:
            w, t = self.check_tag(pt[0], pt[1])
            final_tag.append((w, t))

        return final_tag

    def chunking(self, kalimat_tagged):
        grammar = ["CHUNK: {<NN>{2,}}", "CHUNK: {<NN><JJ>}",
                   "CHUNK: {<NN><VB>}", "CHUNK: {<CD><NN>}",
                   "CHUNK: {<VB><VB>}", "CHUNK: {<VB><NN>}",
                   "CHUNK: {<VB><JJ>}", "CHUNK: {<VB><CD>}",
                   "CHUNK: {<JJ><NN>}", "CHUNK: {<JJ><JJ>}"]

        frasa_kandidat = []

        for i in grammar:
            cp = nltk.RegexpParser(i)
            result = cp.parse(kalimat_tagged)

            leaves = [chunk.leaves() for chunk in result if ((type(chunk) == nltk.tree.Tree) and chunk.label() == 'CHUNK')]
            bigram_groups = [list(nltk.bigrams([w for w, t in leaf])) for leaf in leaves]

            fr = [' '.join(w) for group in bigram_groups for w in group]
            frasa_kandidat = frasa_kandidat + fr

        return frasa_kandidat

    def count_score_similarity(self, frasa):
        token = frasa.split()
        similarity_score = self.similarity_model.predict(token[0], token[1])[0][0]
        random = self.idiom_example_df.sample(n=5)
        idiom_example = random['idiom_example'].values

        for idiom in idiom_example:
            similarity_score = similarity_score + self.similarity_model.predict(frasa, idiom)[0][0]

        return similarity_score

    def similarity(self, frasa):
        frasa_pred = []
        if len(frasa) == 0:
            frasa_pred = []
        else:
            for f in frasa:
                sim_score = self.count_score_similarity(f)
                if sim_score > 0.5:
                    frasa_pred.append(f)

        return frasa_pred

    def validasi(self, frasa):
        frasa_idiom = []
        for f in frasa:
            f = self.text_preprocessing(f, lowercase=True)
            kategori = self.truth_discovery_model.predict([f])[0]
            if kategori == 1:
                frasa_idiom.append(f)

        return frasa_idiom

    def _predict(self, kalimat):
        kalimat = self.text_preprocessing(kalimat, remove_punctuation=True)
        klasifikasi = self.idiom_sentence_classification(kalimat)

        if (klasifikasi == 'kalimat_biasa'):
            frasa = 'none'
            hasil = kalimat, klasifikasi, frasa
        else:
            postag = self.pos_tagging(kalimat)
            frasa_chunk = self.chunking(postag)
            frasa_pred = self.similarity(frasa_chunk)
            frasa_idiom = self.validasi(frasa_pred)
            if len(frasa_idiom) == 0:
                frasa = 'none'
            else:
                frasa = frasa_idiom[0]
            hasil = kalimat, klasifikasi, frasa

        return hasil

    def predict(self, X):
        predicted_result = [self._predict(x) for x in X]
        return predicted_result


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

def kamus_idiom(input):
  data = pd.read_csv("data/IDENTIFIKASI_KLASIFIKASI/kamus-idiom-dataset.csv")
  factory = StemmerFactory()
  stemmer = factory.create_stemmer()
  input = str.lower(input)
  if len(input) == 1:
    result = data[data['huruf']==input]
    result = result.drop(['huruf'], axis=1)
    result = result.values.tolist()
    return result

  if len(input) > 1 and len(input.split()) == 1:
    input = stemmer.stem(input)
    r = input
    result = data[data['kata'].str.contains(r, na=False)]
    result = result.drop(['huruf'], axis=1)
    result = result.values.tolist()
    return result

  if len(input) > 1 and len(input.split()) > 1:
    input = input.split()
    r = stemmer.stem(input[0]) + '(.*?)' + stemmer.stem(input[1])
    result = data[data['idiom'].str.contains(r, na=False)]
    result = result.drop(['huruf'], axis=1)
    result = result.values.tolist()
    return result