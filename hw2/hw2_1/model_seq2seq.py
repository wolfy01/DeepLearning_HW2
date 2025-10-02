import numpy as np
import pandas as pd
import json
import re
import random
import os
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from scipy.special import expit
from os import listdir
from os.path import isfile, join
from bleu_eval import BLEU
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class create_dictionary(object):

    def __init__(self, filepath, min_word_count=10):

        self.filepath = filepath
        self.min_word_count = min_word_count
        self._word_count = {}
        self.vocabulary_size = None
        self._good_words = None
        self._bad_words = None
        self.i2w = None
        self.w2i = None

        
        self._initialize()
        self._build_mapping()
        self._sanitychk()


    def _initialize(self):
        with open(self.filepath, 'r') as f:
            file = json.load(f)
        for data in file:
            for sentence in data['caption']:
                word_sentence = re.sub('[.!,;?]]', ' ', sentence).split()
                for word in word_sentence:
                    word = word.replace('.', '') if '.' in word else word
                    self._word_count[word] = self._word_count.get(word, 0) + 1
        bad_words = [key for key, value in self._word_count.items() if value <= self.min_word_count]
        vocabulary= [key for key, value in self._word_count.items() if value > self.min_word_count]
        self._bad_words = bad_words
        self._good_words = vocabulary
        
    
    def tokenizer_eng(self,text):
        return [tok.text.lower() for tok in self.spacy_eng.tokenizer(text)]

    def _build_mapping(self):
        useful_tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
        self.i2w = {i + len(useful_tokens): w for i, w in enumerate(self._good_words)}
        self.w2i = {w: i + len(useful_tokens) for i, w in enumerate(self._good_words)}
        for token, index in useful_tokens:
            self.i2w[index] = token
            self.w2i[token] = index
        self.vocabulary_size = len(self.i2w) + len(useful_tokens)

    def _sanitychk(self):
        attributes = ['vocabulary_size', '_good_words', '_bad_words', 'i2w', 'w2i']
        for att in attributes:
            if getattr(self, att) is None:
                raise NotImplementedError('Class {} has an attribute "{}" which cannot be None. Error location: {}'.format(__class__.__name__, att, __name__))

    def reannot(self, sentence):
        sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
        sentence = ['<SOS>'] + [w if (self._word_count.get(w, 0) > self.min_word_count)                                     else '<UNK>' for w in sentence] + ['<EOS>']
        return sentence

    def word2index(self, w):
        return self.w2i[w]
        
    def index2word(self, i):
        return self.i2w[i]
        
    def sentence2index(self, sentence):
        return [self.w2i[w] for w in sentence]
        
    def index2sentence(self, index_seq):
        return [self.i2w[int(i)] for i in index_seq]



# ENCODER

class encoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_percentage=0.33):
        super(encoderModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.compress = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.dropout = nn.Dropout(dropout_percentage)
        self.lstm = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()    
        input = input.view(-1, feat_n)
        input = self.compress(input)
        input = self.dropout(input)
        input = input.view(batch_size, seq_len, self.hidden_size) 

        output, hidden_state = self.lstm(input)

        return output, hidden_state


    
# ATTENTION

class attention(nn.Module):
    def __init__(self, hidden_size):
        super(attention, self).__init__()
        
        self.hidden_size = hidden_size
        self.match1 = nn.Linear(2*hidden_size, hidden_size)
        self.match2 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        batch_size, seq_len, feat_n = encoder_outputs.size()
        hidden_state = hidden_state.view(batch_size, 1, feat_n).repeat(1, seq_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2*self.hidden_size)

        x = self.match1(matching_inputs)
        x = self.match2(x)
        
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(batch_size, seq_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        
        return context
    
    

# DECODER

class decoderModel(nn.Module):
    def __init__(self, hidden_size, output_size, vocabulary_size, word_dim, helper=None, dropout_percentage=0.25):
        super(decoderModel, self).__init__()

        self.hidden_size = hidden_size 
        self.output_size = output_size 
        self.vocabulary_size = vocabulary_size
        self.word_dim = word_dim
        self.helper = helper

        self.embedding = nn.Embedding(vocabulary_size, word_dim)
        self.dropout = nn.Dropout(dropout_percentage)
        self.lstm = nn.GRU(hidden_size+word_dim, hidden_size, batch_first=True)
        self.attention = attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)
        
    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = self.initialize_hidden_state(encoder_last_hidden_state)
        
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long()  
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []
        
        targets = self.embedding(targets) 
        _, seq_len, _ = targets.size()
        
        for i in range(seq_len-1): 
            threshold = self._get_teacher_learning_ratio(training_steps=tr_steps)
            current_input_word = targets[:, i] if random.uniform(0.05, 0.995) > threshold else self.embedding(decoder_current_input_word).squeeze(1)
            
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, decoder_current_hidden_state = self.lstm(lstm_input, decoder_current_hidden_state)
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        
        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
        
    
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_current_hidden_state = self.initialize_hidden_state(encoder_last_hidden_state)
        
        decoder_current_input_word = Variable(torch.ones(batch_size, 1)).long() 
        decoder_current_input_word = decoder_current_input_word.cuda() if torch.cuda.is_available() else decoder_current_input_word
        seq_logProb = []
        seq_predictions = []
        assumption_seq_len = 28
        
        for i in range(assumption_seq_len-1):
            current_input_word = self.embedding(decoder_current_input_word).squeeze(1)
            context = self.attention(decoder_current_hidden_state, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, decoder_current_hidden_state = self.lstm(lstm_input, decoder_current_hidden_state)
            
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq_logProb.append(logprob.unsqueeze(1))
            decoder_current_input_word = logprob.unsqueeze(1).max(2)[1]

        seq_logProb = torch.cat(seq_logProb, dim=1)
        seq_predictions = seq_logProb.max(2)[1]
        return seq_logProb, seq_predictions
    
    def initialize_hidden_state(self, last_encoder_hidden_state):
        if last_encoder_hidden_state is None:
            return None
        else:
            return last_encoder_hidden_state
        
    def initialize_cell_state(self, last_encoder_cell_state):
        if last_encoder_cell_state is None:
            return None
        else:
            return last_encoder_cell_state

    def _get_teacher_learning_ratio(self, training_steps):
        return (expit(training_steps/20 +0.85))



#Loss Calculation

class LossFun(nn.Module):
    def __init__(self):
        super(LossFun, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss = 0
        self.avg_loss = None

    def forward(self, x, y, lengths):
        
        batch_size = len(x)
        predict_cat = None
        groundT_cat = None
        flag = True

        for batch in range(batch_size):
            predict      = x[batch]
            ground_truth = y[batch]
            seq_len = lengths[batch] -1

            predict = predict[:seq_len]
            ground_truth = ground_truth[:seq_len]
            if flag:
                predict_cat = predict
                groundT_cat = ground_truth
                flag = False
            else:
                predict_cat = torch.cat((predict_cat, predict), dim=0)
                groundT_cat = torch.cat((groundT_cat, ground_truth), dim=0)

        try:
            assert len(predict_cat) == len(groundT_cat)

        except AssertionError as error:
            print('prediction length is not same as ground truth length')
            print('prediction length: {}, ground truth length: {}'.format(len(predict_cat), len(groundT_cat)))

        self.loss = self.loss_fn(predict_cat, groundT_cat)
        self.avg_loss = self.loss/batch_size
        
        return self.loss
    


# Sequence to Sequence MODEL 

class MODELS(nn.Module):
    def __init__(self, encoder, decoder):
        super(MODELS, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, avi_feats, mode, target_sentences=None, tr_steps=None):
        encoder_outputs, encoder_last_hidden_state = self.encoder(avi_feats)
        if mode == 'train':
            seq_logProb, seq_predictions = self.decoder(encoder_last_hidden_state = encoder_last_hidden_state,
                                                        encoder_output = encoder_outputs,
                targets = target_sentences, mode = mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_logProb, seq_predictions = self.decoder.infer(encoder_last_hidden_state=encoder_last_hidden_state,
                                                              encoder_output=encoder_outputs
                                                              )
        else:
            raise KeyError('mode is not valid')
        return seq_logProb, seq_predictions


# Dataset preparation 

class Creating_Dataset(Dataset):
    def __init__(self, label_json, training_data_path, helper, load_into_ram=False):
       
        if not os.path.exists(label_json):
            raise FileNotFoundError('The file path {} does not exist. Error location: {}'.format(label_json, __name__))
        if not os.path.exists(training_data_path):
            raise FileNotFoundError('The file path {} does not exist. Error location: {}'.format(training_data_path, __name__))

        self.training_data_path = training_data_path

        self.data_pair = []
        self.load_into_ram = load_into_ram
        self.helper = helper

        with open(label_json, 'r') as f:
            label = json.load(f)
        for data in label:
            for sentence in data['caption']:
                sentence = self.helper.reannot(sentence)
                sentence = self.helper.sentence2index(sentence)
                self.data_pair.append((data['id'], sentence))

        if load_into_ram:
            self.avi = {}
            files = os.listdir(training_data_path)

            for file in files:
                key = file.split('.npy')[0]
                value = np.load(os.path.join(training_data_path, file))
                self.avi[key] = value


    def __len__(self):
        return len(self.data_pair)


    def __getitem__(self, idx):
        assert (idx < self.__len__())
        avi_file_name, sentence = self.data_pair[idx]
        avi_file_path = os.path.join(self.training_data_path, '{}.npy'.format(avi_file_name))
        data = torch.Tensor(self.avi[avi_file_name]) if self.load_into_ram else torch.Tensor(np.load(avi_file_path))
        data += torch.Tensor(data.size()).random_(0, 2000)/10000.
        return torch.Tensor(data), torch.Tensor(sentence)


# Model training 

class training(object):
    def __init__(self, model, train_dataloader=None, test_dataloader=None, helper=None):
        self.train_loader = train_dataloader
        self.test_loader = test_dataloader

        self.__CUDA__ = torch.cuda.is_available()

        if self.__CUDA__:
            self.model = model.cuda()
            print('GPU is available')
        else:
            self.model = model.cpu()

        self.parameters = model.parameters()
        self.loss_fn = LossFun()
        self.loss = None
        self.optimizer = optim.Adam(self.parameters, lr=0.001)
        self.helper = helper

    
    def train(self, epoch):
        self.model.train()
        test_avi, test_truth = None, None
        for batch_idx, batch in enumerate(self.train_loader):
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()

            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)

            self.optimizer.zero_grad()
            seq_logProb, seq_predictions = self.model(avi_feats, target_sentences=ground_truths, mode='train', tr_steps=epoch)
                        
            ground_truths = ground_truths[:, 1:]  
            loss = self.loss_fn(seq_logProb, ground_truths, lengths)
            loss.backward()
            self.optimizer.step()

            if (batch_idx+1):
                info = self.get_training_info(epoch=epoch, batch_id=batch_idx, batch_size=len(lengths), total_data_size=len(self.train_loader.dataset),
                    n_batch=len(self.train_loader), loss=loss.item())
                print(info, end='\r')
                sys.stdout.write("\033[K")

        info = self.get_training_info(epoch=epoch, batch_id=batch_idx, batch_size=len(lengths), total_data_size=len(self.train_loader.dataset),
            n_batch=len(self.train_loader), loss=loss.item())
        print(info)
        
        self.loss = loss.item()


    def eval(self):
        self.model.eval()
        test_predictions, test_truth = None, None
        for batch_idx, batch in enumerate(self.test_loader):    
            avi_feats, ground_truths, lengths = batch
            if self.__CUDA__:
                avi_feats, ground_truths = avi_feats.cuda(), ground_truths.cuda()
            avi_feats, ground_truths = Variable(avi_feats), Variable(ground_truths)
            
            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')
            ground_truths = ground_truths[:, 1:]
            test_predictions = seq_predictions[:3]
            test_truth = ground_truths[:3]
            break


    def test(self):
        self.model.eval()
        ss = []
        for batch_idx, batch in enumerate(self.test_loader):
            id, avi_feats = batch
            if self.__CUDA__:
                avi_feats = avi_feats.cuda()
            id, avi_feats = id, Variable(avi_feats).float()

            seq_logProb, seq_predictions = self.model(avi_feats, mode='inference')
            test_predictions = seq_predictions
            result = [[x if x != '<UNK>' else 'something' for x in self.helper.index2sentence(s)] for s in test_predictions]
            result = [' '.join(s).split('<EOS>')[0] for s in result]
            rr = zip(id, result)
            for r in rr:
                ss.append(r)
        return ss

    
    def get_training_info(self,**kwargs):
        ep = kwargs.pop("epoch", None)
        bID = kwargs.pop("batch_id", None)
        bs = kwargs.pop("batch_size", None)
        tds = kwargs.pop("total_data_size", None)
        nb = kwargs.pop("n_batch", None)
        loss = kwargs.pop("loss", None)
        info = "Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(ep, (bID+1)*bs, tds, 100.*bID/nb, loss)
        return info


# Main traning file 

def minibatch(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    avi_data, captions = zip(*data) 
    avi_data = torch.stack(avi_data, 0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return avi_data, targets, lengths


def append_log_file(fileLoc,filename,model_name,time):
    f = open(f"{fileLoc}/{filename}.txt", "a")
    f.write(f"\n {model_name}, {time}")
    f.close()


def main_execution():
    training_json = 'training_data/training_label.json'
    training_feats = 'training_data/feat'
    testing_json = 'testing_data/testing_label.json' 
    testing_feats = 'testing_data/feat'

    min_word_count = 3

    helper = create_dictionary(training_json, min_word_count=3)
    train_dataset = Creating_Dataset(label_json=training_json, training_data_path=training_feats, helper=helper, load_into_ram=True)
    test_dataset = Creating_Dataset(label_json=testing_json, training_data_path=testing_feats, helper=helper, load_into_ram=True)


    inputFeatDim = 4096
    output_dim = helper.vocabulary_size
    batch_sizes = [16]
    hidden_sizes = [256]
    dropout_percentages=[0.3]
    word_dims = [2048]

    epochs_n = 20
    ModelSaveLoc = 'TrainedModels'
    if not os.path.exists(ModelSaveLoc):
        os.mkdir(ModelSaveLoc)

    log_filename = str(time.time())+ "_ModelTime_log"
    f = open(f"{ModelSaveLoc}/{log_filename}.txt", "x")
    f.write("Model Name, time")
    f.close()

    for batch_size in batch_sizes:
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=minibatch)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=minibatch)


        for hidden_size in hidden_sizes:
            for dropout_percentage in dropout_percentages:
                for word_dim in word_dims:

                    encoder = encoderModel(input_size=inputFeatDim, hidden_size=hidden_size, dropout_percentage=dropout_percentage)
                    decoder = decoderModel(hidden_size=hidden_size, output_size=output_dim, vocabulary_size=output_dim, word_dim=word_dim, dropout_percentage=dropout_percentage)
                    model = MODELS(encoder=encoder, decoder=decoder)
                    train = training(model=model, train_dataloader=train_dataloader, test_dataloader=test_dataloader, helper=helper)

                    start = time.time()
                    for epoch in range(epochs_n):
                        train.train(epoch+1)
                        train.eval()

                    end = time.time()

                    model_name = "model_batchsize_"+str(batch_size)+"_hiddensize_"+ str(hidden_size) + "_DP_"+str(dropout_percentage) + "_worddim_"+str(word_dim)
                    torch.save(model, "{}/{}.h5".format(ModelSaveLoc, model_name))
                    ti = end-start
                    append_log_file(ModelSaveLoc,log_filename,model_name,ti)



col=['model_name','Average_Bleu_Score']


class test_data(Dataset):
    def __init__(self, test_data_path):
        self.avi = []
        files = os.listdir(test_data_path)
        for file in files:
            key = file.split('.npy')[0]
            value = np.load(os.path.join(test_data_path, file))
            self.avi.append([key, value])

    def __len__(self):
        return len(self.avi)
    
    def __getitem__(self, idx):
        assert (idx < self.__len__())

        return self.avi[idx]


def testmodel(arg):
    ModelSaveLoc = 'TrainedModels'
    result_pd = pd.DataFrame(columns=col)
    
    onlyfiles = [f for f in listdir(ModelSaveLoc) if isfile(join(ModelSaveLoc, f))]
    fileExt = r".h5"
    file_full_path = [join(ModelSaveLoc, _) for _ in listdir(ModelSaveLoc) if _.endswith(fileExt)]
    
    result = pd.DataFrame(columns=col)
    
    dataset = test_data('{}/feat'.format(arg[0]))
    testing_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=8)
    training_json = 'training_data/training_label.json' 
    helper = create_dictionary(training_json, min_word_count=3)

    l = []
    for model_loc in file_full_path:
        print(model_loc)
        if not torch.cuda.is_available():
            model = torch.load(model_loc, map_location=lambda storage, loc: storage, weights_only=False)
        else:
            model = torch.load(model_loc, weights_only=False)


        testing = training(model=model, test_dataloader=testing_loader, helper=helper)

        for epoch in range(1):
            ss = testing.test()

        with open(arg[1], 'w') as f:  
            for id, s in ss:
                f.write('{},{}\n'.format(id, s))

    
        test = json.load(open('testing_data/testing_label.json','r')) 
        output = arg[1]  
        result = {}
        with open(output,'r') as f:
            for line in f:
                line = line.rstrip()
                comma = line.index(',')
                test_id = line[:comma]
                caption = line[comma+1:]
                result[test_id] = caption

        bleu=[]
        for item in test:
            score_per_video = []
            captions = [x.rstrip('.') for x in item['caption']]
            score_per_video.append(BLEU(result[item['id']],captions,True))
            bleu.append(score_per_video[0])
        average = sum(bleu) / len(bleu)
        print("The average bleu score is " + str(average))
        row={
            'model_loc' : model_loc,
            'bleu_Score' : average
        }
        df = pd.DataFrame({'model_loc': [model_loc], 'bleu_Score': [average]})
        # df.to_csv("temp_result.csv",index=False)
        l.append(df)
        
    res = pd.concat(l, axis =0)
    res.to_csv("final_result.csv",index=False)



if __name__=="__main__":
    train = False
    if train == True:
        main_execution()
    else:
        arg = [sys.argv[1],sys.argv[2]]
        testmodel(arg)