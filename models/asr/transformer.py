import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
from torch.autograd import Variable

import os
import numpy as np
import math

from utils import constant
from models.common_layers import MultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, PositionwiseFeedForwardWithConv, get_subsequent_mask, get_non_pad_mask, get_attn_key_pad_mask, get_attn_pad_mask, pad_list
from utils.metrics import calculate_metrics
from utils.lstm_utils import calculate_lm_score

class Transformer(nn.Module):
    """
    Transformer class
    args:
        encoder: Encoder object
        decoder: Decoder object
    """

    def __init__(self, encoder, decoder, feat_extractor='vgg_cnn'):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.id2label = decoder.id2label
        self.feat_extractor = feat_extractor

        # feature embedding
        if feat_extractor == 'emb_cnn':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            )
        elif feat_extractor == 'vgg_cnn':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, padded_input, input_lengths, padded_target, verbose=False):
        """
        args:
            padded_input: B x 1 (channel for spectrogram=1) x (freq) x T
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            pred: B x T x vocab
            gold: B x T
        """
        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn':
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH

        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)
        hyp_best_scores, hyp_best_ids = torch.topk(pred, 1, dim=2)

        hyp_seq = hyp_best_ids.squeeze(2)
        gold_seq = gold

        return pred, gold, hyp_seq, gold_seq

    def evaluate(self, padded_input, input_lengths, padded_target, beam_search=False, beam_width=0, beam_nbest=0, lm=None, lm_rescoring=False, lm_weight=0.1, c_weight=1, verbose=False):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            batch_ids_nbest_hyps: list of nbest id
            batch_strs_nbest_hyps: list of nbest str
            batch_strs_gold: list of gold str
        """
        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn':
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH

        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        hyp, gold, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)
        hyp_best_scores, hyp_best_ids = torch.topk(hyp, 1, dim=2)
        
        strs_gold = ["".join([self.id2label[int(x)] for x in gold_seq]) for gold_seq in gold]

        if beam_search:
            ids_hyps, strs_hyps = self.decoder.beam_search(encoder_padded_outputs, beam_width=beam_width, nbest=1, lm=lm, lm_rescoring=lm_rescoring, lm_weight=lm_weight, c_weight=c_weight)
            if len(strs_hyps) != sizes[0]:
                print(">>>>>>> switch to greedy")
                strs_hyps = self.decoder.greedy_search(encoder_padded_outputs)
        else:
            strs_hyps = self.decoder.greedy_search(encoder_padded_outputs)
        
        if verbose:
            print("GOLD", strs_gold)
            print("HYP", strs_hyps)

        return _, strs_hyps, strs_gold

class Encoder(nn.Module):
    """ 
    Encoder Transformer class
    """

    def __init__(self, num_layers, num_heads, dim_model, dim_key, dim_value, dim_input, dim_inner, dropout=0.1, src_max_length=1000):
        super(Encoder, self).__init__()

        self.dim_input = dim_input
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_inner = dim_inner

        self.src_max_length = src_max_length

        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        self.input_linear = nn.Linear(dim_input, dim_model)
        self.layer_norm_input = nn.LayerNorm(dim_model)
        self.positional_encoding = PositionalEncoding(
            dim_model, src_max_length)

        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, padded_input, input_lengths):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
        return:
            output: B x T x H
        """
        encoder_self_attn_list = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)  # B x T x D
        seq_len = padded_input.size(1)
        self_attn_mask = get_attn_pad_mask(padded_input, input_lengths, seq_len)  # B x T x T

        encoder_output = self.layer_norm_input(self.input_linear(
            padded_input)) + self.positional_encoding(padded_input)

        for layer in self.layers:
            encoder_output, self_attn = layer(
                encoder_output, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask)
            encoder_self_attn_list += [self_attn]

        return encoder_output, encoder_self_attn_list


class EncoderLayer(nn.Module):
    """
    Encoder Layer Transformer class
    """

    def __init__(self, num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardWithConv(
            dim_model, dim_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, self_attn_mask=None):
        enc_output, self_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, self_attn


class Decoder(nn.Module):
    """
    Decoder Layer Transformer class
    """

    def __init__(self, id2label, num_src_vocab, num_trg_vocab, num_layers, num_heads, dim_emb, dim_model, dim_inner, dim_key, dim_value, dropout=0.1, trg_max_length=1000, emb_trg_sharing=False):
        super(Decoder, self).__init__()
        # self.sos_id = constant.SOS_TOKEN
        # self.eos_id = constant.EOS_TOKEN

        self.id2label = id2label

        self.num_src_vocab = num_src_vocab
        self.num_trg_vocab = num_trg_vocab
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dim_emb = dim_emb
        self.dim_model = dim_model
        self.dim_inner = dim_inner
        self.dim_key = dim_key
        self.dim_value = dim_value

        self.dropout_rate = dropout
        self.emb_trg_sharing = emb_trg_sharing

        self.trg_max_length = trg_max_length

        self.trg_embedding = nn.Embedding(num_trg_vocab, dim_emb, padding_idx=constant.PAD_TOKEN)
        self.positional_encoding = PositionalEncoding(
            dim_model, max_length=trg_max_length)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderLayer(dim_model, dim_inner, num_heads,
                         dim_key, dim_value, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.output_linear = nn.Linear(dim_model, num_trg_vocab+1, bias=False)  #Ctc +1 added here
        nn.init.xavier_normal_(self.output_linear.weight)

        if emb_trg_sharing:
            self.output_linear.weight = self.trg_embedding.weight
            self.x_logit_scale = (dim_model ** -0.5)
        else:
            self.x_logit_scale = 1.0

    def preprocess(self, padded_input):
        seq_clean = [y[y != constant.PAD_TOKEN] for y in padded_input]
        seq_pad = pad_list(seq_clean, constant.PAD_TOKEN)  # shape: (B, max_T)
        return seq_pad, seq_pad  # both input and gold are the same for CTC

    def forward(self, padded_input, encoder_padded_outputs, encoder_input_lengths):
        decoder_self_attn_list, decoder_encoder_attn_list = [], []
        seq_in_pad, seq_out_pad = self.preprocess(padded_input)

        non_pad_mask = get_non_pad_mask(seq_in_pad, pad_idx=constant.PAD_TOKEN)
        self_attn_mask_subseq = get_subsequent_mask(seq_in_pad)
        self_attn_mask_keypad = get_attn_key_pad_mask(seq_k=seq_in_pad, seq_q=seq_in_pad, pad_idx=constant.PAD_TOKEN)
        self_attn_mask = (self_attn_mask_keypad + self_attn_mask_subseq).gt(0)

        output_length = seq_in_pad.size(1)
        dec_enc_attn_mask = get_attn_pad_mask(encoder_padded_outputs, encoder_input_lengths, output_length)

        decoder_output = self.dropout(
            self.trg_embedding(seq_in_pad) * self.x_logit_scale + self.positional_encoding(seq_in_pad)
        )

        for layer in self.layers:
            decoder_output, decoder_self_attn, decoder_enc_attn = layer(
                decoder_output, encoder_padded_outputs,
                non_pad_mask=non_pad_mask,
                self_attn_mask=self_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask
            )

            decoder_self_attn_list.append(decoder_self_attn)
            decoder_encoder_attn_list.append(decoder_enc_attn)

        seq_logit = self.output_linear(decoder_output)
        pred, gold = seq_logit, seq_out_pad

        return pred, gold, decoder_self_attn_list, decoder_encoder_attn_list


    def post_process_hyp(self, hyp):
        """
        args: 
            hyp: list of hypothesis
        output:
            list of hypothesis (string)>
        """
        return "".join([self.id2label[int(x)] for x in hyp['yseq'][1:]])

    def greedy_search(self, encoder_padded_outputs, beam_width=2, lm_rescoring=False, lm=None, lm_weight=0.1, c_weight=1):
        """
        Greedy search, decode 1-best utterance
        args:
            encoder_padded_outputs: B x T x H
        output:
            batch_ids_nbest_hyps: list of nbest in ids (size B)
            batch_strs_nbest_hyps: list of nbest in strings (size B)
        """
        batch_size = encoder_padded_outputs.size(0)

        # Project encoder outputs to vocab logits
        logits = self.output_linear(encoder_padded_outputs)  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)            # [B, T, V]

        # Greedy decoding: pick most likely token at each time step
        pred_ids = torch.argmax(log_probs, dim=-1)           # [B, T]

        decoded_batch = []
        for b in range(batch_size):
            prev_token = None
            hyp = []
            for t in range(pred_ids.size(1)):
                token = pred_ids[b, t].item()

                # Ignore CTC blanks and padding
                if token == constant.BLANK_TOKEN or token == constant.PAD_TOKEN:
                    continue

                # Collapse repeated tokens
                if token != prev_token:
                    hyp.append(token)
                prev_token = token

            # Convert token IDs to string
            hyp_str = "".join([self.id2label[token_id] for token_id in hyp])
            decoded_batch.append(hyp_str)

        return decoded_batch


    def beam_search(self, encoder_padded_outputs, beam_width=2, nbest=5, lm_rescoring=False, lm=None, lm_weight=0.1, c_weight=1, prob_weight=1.0):
        """
        Beam search, decode nbest utterances
        args:
            encoder_padded_outputs: B x T x H
            beam_size: int
            nbest: int
        output:
            batch_ids_nbest_hyps: list of nbest in ids (size B)
            batch_strs_nbest_hyps: list of nbest in strings (size B)
        """
        batch_size = encoder_padded_outputs.size(0)
        logits = self.output_linear(encoder_padded_outputs)  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)            # [B, T, V]

        blank_id = constant.BLANK_TOKEN
        pad_id = constant.PAD_TOKEN

        batch_ids_nbest_hyps = []
        batch_strs_nbest_hyps = []

        for b in range(batch_size):
            hyps = [([], 0.0)]  # List of (token sequence, score)

            for t in range(log_probs.size(1)):  # Iterate over time steps
                next_hyps = {}
                probs_t = log_probs[b, t]  # [V]

                for seq, score in hyps:
                    for c in range(probs_t.size(0)):
                        new_seq = seq.copy()
                        new_score = score + probs_t[c].item()

                        # CTC collapsing rules:
                        if c == blank_id or c == pad_id:
                            key = tuple(new_seq)
                        else:
                            if len(seq) == 0 or seq[-1] != c:
                                new_seq.append(c)
                            key = tuple(new_seq)

                        if key in next_hyps:
                            if next_hyps[key] < new_score:
                                next_hyps[key] = new_score
                        else:
                            next_hyps[key] = new_score

                # Prune to top beam_width
                sorted_hyps = sorted(next_hyps.items(), key=lambda x: x[1], reverse=True)
                hyps = [(list(k), v) for k, v in sorted_hyps[:beam_width]]

            # Sort final hypotheses
            final_hyps = sorted(hyps, key=lambda x: x[1], reverse=True)[:nbest]

            for token_seq, score in final_hyps:
                hyp_str = "".join(self.id2label[tok] for tok in token_seq if tok != pad_id and tok != blank_id)

                if lm_rescoring and lm is not None:
                    lm_score, num_words, oov_token = calculate_lm_score(hyp_str, lm, self.id2label)
                    score += lm_weight * lm_score + math.sqrt(num_words) * c_weight

                batch_ids_nbest_hyps.append(token_seq)
                batch_strs_nbest_hyps.append(hyp_str)

        return batch_ids_nbest_hyps, batch_strs_nbest_hyps

class DecoderLayer(nn.Module):
    """
    Decoder Transformer class
    """

    def __init__(self, dim_model, dim_inner, num_heads, dim_key, dim_value, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.encoder_attn = MultiHeadAttention(
            num_heads, dim_model, dim_key, dim_value, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForwardWithConv(
            dim_model, dim_inner, dropout=dropout)

    def forward(self, decoder_input, encoder_output, non_pad_mask=None, self_attn_mask=None, dec_enc_attn_mask=None):
        decoder_output, decoder_self_attn = self.self_attn(
            decoder_input, decoder_input, decoder_input, mask=self_attn_mask)
        decoder_output *= non_pad_mask

        decoder_output, decoder_encoder_attn = self.encoder_attn(
            decoder_output, encoder_output, encoder_output, mask=dec_enc_attn_mask)
        decoder_output *= non_pad_mask

        decoder_output = self.pos_ffn(decoder_output)
        decoder_output *= non_pad_mask

        return decoder_output, decoder_self_attn, decoder_encoder_attn        