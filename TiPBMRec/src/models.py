import os
import copy
import pickle
import gensim
import random
from tqdm import tqdm
from modules import TransformerEncoder, get_attention_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TiPBMRec(nn.Module):
    def __init__(self, args):
        super(TiPBMRec, self).__init__()
        self.n_layers = args.n_layers
        self.n_heads = args.n_heads
        self.hidden_size = args.hidden_size
        self.inner_size = args.inner_size
        self.hidden_dropout_prob = args.hidden_dropout_prob
        self.attn_dropout_prob = args.attn_dropout_prob
        self.hidden_act = args.hidden_act
        self.layer_norm_eps = args.layer_norm_eps
        self.item_size = args.item_size
        self.max_seq_length = args.max_seq_length
        self.args = args
        self.time_threshold_coeff = 1
        self.co_occur_threshold = 4
        self.temperature = 1
        self.lambda1 = 0.5
        self.lambda2 = 0.5
        self.initializer_range = args.initializer_range
        self.item_embedding = nn.Embedding(self.item_size, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_seq_length, self.hidden_size)

        self.trm_encoder = TransformerEncoder(
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            hidden_size=self.hidden_size,
            inner_size=self.inner_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            attn_dropout_prob=self.attn_dropout_prob,
            hidden_act=self.hidden_act,
            layer_norm_eps=self.layer_norm_eps
        )
        self.seq_adj_graph_embedding = nn.Embedding(self.item_size, self.hidden_size)
        self.co_occur_graph_embedding = nn.Embedding(self.item_size, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

        self.time_decay_layer = nn.Linear(1, 1)

        self.output_layer = nn.Linear(self.hidden_size * 3, self.item_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def transformer_encoder(self, item_seq, time_intervals=None):
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids)

        item_emb = self.item_embedding(item_seq)
        if time_intervals is not None:
            time_emb = self._get_time_embedding(time_intervals)
            item_emb = item_emb + time_emb

        input_emb = item_emb + position_embedding
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)

        extended_attention_mask = self.get_attention_mask(item_seq)

        trm_output = self.trm_encoder(input_emb, extended_attention_mask, output_all_encoded_layers=True)
        output = trm_output[-1]
        return output

    def _get_time_embedding(self, time_intervals):

        time_emb = torch.zeros_like(time_intervals).unsqueeze(-1).float()
        return time_emb

    def get_attention_mask(self, item_seq):
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def build_time_aware_graphs(self, user_sequences, time_intervals, user_ids):
        batch_size, seq_len = user_sequences.shape

        # 获取图嵌入
        seq_adj_emb = self.seq_adj_graph_embedding(user_sequences)
        co_occur_emb = self.co_occur_graph_embedding(user_sequences)

        return seq_adj_emb, co_occur_emb

    def compute_conformity_weights(self, user_seq, time_intervals, seq_adj_emb, co_occur_emb):
        batch_size, seq_len = user_seq.shape

        if time_intervals is not None:
            time_std = torch.std(time_intervals, dim=1, keepdim=True)  # σ
        else:
            time_std = torch.ones(batch_size, 1, device=user_seq.device)

        individual_sim = F.cosine_similarity(seq_adj_emb, seq_adj_emb.detach(), dim=-1)
        individual_conformity = individual_sim.mean(dim=-1, keepdim=True) * time_std
        if time_intervals is not None:
            user_avg_time = time_intervals.mean(dim=1, keepdim=True)
            time_decay_user = torch.exp(-time_intervals / user_avg_time)
            overall_decay = torch.exp(-time_intervals / time_intervals.mean())
        else:
            time_decay_user = torch.ones_like(user_seq).float()
            overall_decay = torch.ones_like(user_seq).float()

        seq_emb = self.item_embedding(user_seq)
        local_rep = (seq_emb * time_decay_user.unsqueeze(-1)).sum(dim=1) / time_decay_user.sum(dim=1, keepdim=True)
        overall_rep = (seq_emb * overall_decay.unsqueeze(-1)).sum(dim=1) / overall_decay.sum(dim=1, keepdim=True)

        overall_conformity = F.cosine_similarity(local_rep, overall_rep, dim=-1).unsqueeze(-1)

        combined_weights = 0.5 * (individual_conformity + overall_conformity)

        return individual_conformity, overall_conformity, combined_weights

    def contrastive_learning(self, seq_emb, seq_adj_emb, co_occur_emb, conformity_weights):
        batch_size, seq_len, hidden_size = seq_emb.shape

        seq_emb_flat = seq_emb.reshape(-1, hidden_size)
        seq_adj_emb_flat = seq_adj_emb.reshape(-1, hidden_size)

        user_sim = F.cosine_similarity(seq_emb_flat.unsqueeze(1), seq_adj_emb_flat.unsqueeze(0), dim=-1)
        user_sim = user_sim / self.temperature

        pos_mask = torch.eye(batch_size * seq_len, device=seq_emb.device).bool()

        user_logits = F.log_softmax(user_sim, dim=-1)
        user_pos_logits = user_logits[pos_mask]
        weights_flat = conformity_weights.reshape(-1, 1).expand(-1, seq_len).reshape(-1)
        user_loss = -torch.mean(weights_flat * user_pos_logits)

        item_sim = F.cosine_similarity(seq_adj_emb_flat.unsqueeze(1),
                                       co_occur_emb.reshape(-1, hidden_size).unsqueeze(0),
                                       dim=-1)
        item_sim = item_sim / self.temperature
        item_logits = F.log_softmax(item_sim, dim=-1)
        item_pos_logits = item_logits[pos_mask]
        phi_weights = 1 - weights_flat  # φ = 1 - ω
        item_loss = -torch.mean(phi_weights * item_pos_logits)

        return user_loss, item_loss

    def kl_divergence_loss(self, conformity_weights):
        weights_flat = conformity_weights.reshape(-1)
        p = F.softmax(weights_flat, dim=0)

        q = torch.ones_like(p) / p.size(0)
        kl_loss = F.kl_div(p.log(), q, reduction='batchmean')

        return kl_loss

    def forward(self, item_seq, time_intervals=None, user_ids=None):
        batch_size, seq_len = item_seq.shape

        sequence_output = self.transformer_encoder(item_seq, time_intervals)

        seq_adj_emb, co_occur_emb = self.build_time_aware_graphs(
            item_seq, time_intervals, user_ids
        )

        individual_weight, overall_weight, conformity_weights = self.compute_conformity_weights(
            item_seq, time_intervals, seq_adj_emb, co_occur_emb
        )

        user_cl_loss, item_cl_loss = self.contrastive_learning(
            sequence_output, seq_adj_emb, co_occur_emb, conformity_weights
        )

        kl_loss = self.kl_divergence_loss(conformity_weights)

        combined_emb = torch.cat([
            sequence_output[:, -1, :],  # 序列最后位置的表示
            seq_adj_emb[:, -1, :],  # 序列邻接图表示
            co_occur_emb[:, -1, :]  # 共现图表示
        ], dim=-1)

        prediction_scores = self.output_layer(combined_emb)

        output_dict = {
            'prediction_scores': prediction_scores,
            'sequence_output': sequence_output,
            'conformity_weights': conformity_weights,
            'individual_weight': individual_weight,
            'overall_weight': overall_weight,
            'user_cl_loss': user_cl_loss,
            'item_cl_loss': item_cl_loss,
            'kl_loss': kl_loss,
            'total_loss': None
        }

        return output_dict

    def calculate_loss(self, output_dict, labels):
        rec_loss = F.cross_entropy(
            output_dict['prediction_scores'],
            labels.view(-1)
        )
        total_loss = rec_loss + \
                     self.lambda1 * output_dict['kl_loss'] + \
                     self.lambda2 * (output_dict['user_cl_loss'] + output_dict['item_cl_loss'])

        output_dict['rec_loss'] = rec_loss
        output_dict['total_loss'] = total_loss

        return total_loss

    def predict(self, item_seq, time_intervals=None, user_ids=None):
        output_dict = self.forward(item_seq, time_intervals, user_ids)
        return output_dict['prediction_scores']

class OnlineItemSimilarity:

    def __init__(self, item_size):
        self.item_size = item_size
        self.item_embedding = None
        self.cuda_condition = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        self.total_item_list = torch.tensor([i for i in range(self.item_size)],
                                            dtype=torch.long).to(self.device)

    def update_embedding_matrix(self, item_embedding):
        self.item_embedding = copy.deepcopy(item_embedding)
        self.base_embedding_matrix = self.item_embedding(self.total_item_list)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item_idx in range(1, self.item_size):
            try:
                item_vector = self.item_embedding(torch.tensor(item_idx).to(self.device)).view(-1, 1)
                item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
                max_score = max(torch.max(item_similarity), max_score)
                min_score = min(torch.min(item_similarity), min_score)
            except:
                continue
        return max_score, min_score

    def most_similar(self, item_idx, top_k=1, with_score=False):
        item_idx = torch.tensor(item_idx, dtype=torch.long).to(self.device)
        item_vector = self.item_embedding(item_idx).view(-1, 1)
        item_similarity = torch.mm(self.base_embedding_matrix, item_vector).view(-1)
        item_similarity = (self.max_score - item_similarity) / (self.max_score - self.min_score)
        # remove item idx itself
        values, indices = item_similarity.topk(top_k + 1)
        if with_score:
            item_list = indices.tolist()
            score_list = values.tolist()
            if item_idx in item_list:
                idd = item_list.index(item_idx)
                item_list.remove(item_idx)
                score_list.pop(idd)
            return list(zip(item_list, score_list))
        item_list = indices.tolist()
        if item_idx in item_list:
            item_list.remove(item_idx)
        return item_list


class OfflineItemSimilarity:
    def __init__(self, data_file=None, similarity_path=None, model_name='ItemCF', dataset_name='Sports_and_Outdoors'):
        self.dataset_name = dataset_name
        self.similarity_path = similarity_path
        # train_data_list used for item2vec, train_data_dict used for itemCF and itemCF-IUF
        self.train_data_list, self.train_item_list, self.train_data_dict = self._load_train_data(data_file)
        self.model_name = model_name
        self.similarity_model = self.load_similarity_model(self.similarity_path)
        self.max_score, self.min_score = self.get_maximum_minimum_sim_scores()

    def get_maximum_minimum_sim_scores(self):
        max_score, min_score = -1, 100
        for item in self.similarity_model.keys():
            for neig in self.similarity_model[item]:
                sim_score = self.similarity_model[item][neig]
                max_score = max(max_score, sim_score)
                min_score = min(min_score, sim_score)
        return max_score, min_score

    def _convert_data_to_dict(self, data):
        """
        split the data set
        testdata is a test data set
        traindata is a train set
        """
        train_data_dict = {}
        for user, item, record in data:
            train_data_dict.setdefault(user, {})
            train_data_dict[user][item] = record
        return train_data_dict

    def _save_dict(self, dict_data, save_path='./similarity.pkl'):
        print("saving data to ", save_path)
        with open(save_path, 'wb') as write_file:
            pickle.dump(dict_data, write_file)

    def _load_train_data(self, data_file=None):
        """
        read the data from the data file which is a data set
        """
        train_data = []
        train_data_list = []
        train_data_set_list = []
        for line in open(data_file).readlines():
            userid, items = line.strip().split(' ', 1)
            # only use training data
            items = items.split(' ')[:-3]
            train_data_list.append(items)
            train_data_set_list += items
            for itemid in items:
                train_data.append((userid, itemid, int(1)))
        return train_data_list, set(train_data_set_list), self._convert_data_to_dict(train_data)

    def _generate_item_similarity(self, train=None, save_path='./'):
        """
        calculate co-rated users between items
        """
        print("getting item similarity...")
        train = train or self.train_data_dict
        C = dict()
        N = dict()

        if self.model_name in ['ItemCF', 'ItemCF_IUF']:
            print("Step 1: Compute Statistics")
            data_iter = tqdm(enumerate(train.items()), total=len(train.items()))
            for idx, (u, items) in data_iter:
                if self.model_name == 'ItemCF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1
                elif self.model_name == 'ItemCF_IUF':
                    for i in items.keys():
                        N.setdefault(i, 0)
                        N[i] += 1
                        for j in items.keys():
                            if i == j:
                                continue
                            C.setdefault(i, {})
                            C[i].setdefault(j, 0)
                            C[i][j] += 1 / math.log(1 + len(items) * 1.0)
            self.itemSimBest = dict()
            print("Step 2: Compute co-rate matrix")
            c_iter = tqdm(enumerate(C.items()), total=len(C.items()))
            for idx, (cur_item, related_items) in c_iter:
                self.itemSimBest.setdefault(cur_item, {})
                for related_item, score in related_items.items():
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score / math.sqrt(N[cur_item] * N[related_item])
            self._save_dict(self.itemSimBest, save_path=save_path)
        elif self.model_name == 'Item2Vec':
            # details here: https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py
            print("Step 1: train item2vec model")
            item2vec_model = gensim.models.Word2Vec(sentences=self.train_data_list,
                                                    vector_size=20, window=5, min_count=0,
                                                    epochs=100)
            self.itemSimBest = dict()
            total_item_nums = len(item2vec_model.wv.index_to_key)
            print("Step 2: convert to item similarity dict")
            total_items = tqdm(item2vec_model.wv.index_to_key, total=total_item_nums)
            for cur_item in total_items:
                related_items = item2vec_model.wv.most_similar(positive=[cur_item], topn=20)
                self.itemSimBest.setdefault(cur_item, {})
                for (related_item, score) in related_items:
                    self.itemSimBest[cur_item].setdefault(related_item, 0)
                    self.itemSimBest[cur_item][related_item] = score
            print("Item2Vec model saved to: ", save_path)
            self._save_dict(self.itemSimBest, save_path=save_path)

    def load_similarity_model(self, similarity_model_path):
        if not similarity_model_path:
            raise ValueError('invalid path')
        elif not os.path.exists(similarity_model_path):
            print("the similirity dict not exist, generating...")
            self._generate_item_similarity(save_path=self.similarity_path)
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            with open(similarity_model_path, 'rb') as read_file:
                similarity_dict = pickle.load(read_file)
            return similarity_dict
        elif self.model_name == 'Random':
            similarity_dict = self.train_item_list
            return similarity_dict

    def most_similar(self, item, top_k=1, with_score=False):
        if self.model_name in ['ItemCF', 'ItemCF_IUF', 'Item2Vec', 'LightGCN']:
            """TODO: handle case that item not in keys"""
            if str(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[str(item)].items(), key=lambda x: x[1],
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (self.max_score - float(x[1])) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            elif int(item) in self.similarity_model:
                top_k_items_with_score = sorted(self.similarity_model[int(item)].items(), key=lambda x: x[1],
                                                reverse=True)[0:top_k]
                if with_score:
                    return list(
                        map(lambda x: (int(x[0]), (self.max_score - float(x[1])) / (self.max_score - self.min_score)),
                            top_k_items_with_score))
                return list(map(lambda x: int(x[0]), top_k_items_with_score))
            else:
                item_list = list(self.similarity_model.keys())
                random_items = random.sample(item_list, k=top_k)
                if with_score:
                    return list(map(lambda x: (int(x), 0.0), random_items))
                return list(map(lambda x: int(x), random_items))
        elif self.model_name == 'Random':
            random_items = random.sample(self.similarity_model, k=top_k)
            if with_score:
                return list(map(lambda x: (int(x), 0.0), random_items))
            return list(map(lambda x: int(x), random_items))
