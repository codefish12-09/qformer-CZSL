from itertools import product
import math
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
from my_model.base import *
import numpy as np
from lavis.models import load_model_and_preprocess
from transformers import BertTokenizer, BasicTokenizer
from clip_modules1.clip_model import load_clip, QuickGELU
class Adapter(nn.Module):
    # Referece: https://github.com/ShoufaChen/AdaptFormer
    def __init__(self,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="lora",
                 adapter_scalar="0.1",
                 adapter_layernorm_option="none"):
        super().__init__()
        self.n_embd = d_model
        self.down_size = bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        self.init_option = init_option

        self._reset_parameters()
    def _reset_parameters(self):
        if self.init_option == "bert":
            raise NotImplementedError
        elif self.init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training)
        up = self.up_proj(down)

        up = up * self.scale

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output

class Base(nn.Module):
    def __init__(self, config, attributes, classes,offset):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.clip = load_clip(name=config.clip_model, context_length=config.context_length)
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip2_feature_extractor", model_type="pretrain_vitL", is_eval=True, device=self.device)
        self.tokenizer=self.model.tokenizer
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.token_ids, self.soft_att_obj, ctx_vectors,self.token_masks = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True
        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        for name, param in self.named_parameters():
            if 'model.query_tokens' in name:  #1*32*768
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt = nn.Parameter(ctx_vectors).cuda()
        self.fusion = FusionTextImageBlock(config.width_img, config.width_txt, len(self.attributes), len(self.classes), config.SA_K, 
                                           context_length=self.config.context_length)
        self.weight = config.res_w
        self.additional_visual_params = self.add_visual_tunable_params()
        output_dim = self.clip.visual.output_dim

    def add_visual_tunable_params(self):
        adapter_num = 2 * self.clip.visual.transformer.layers
        params = nn.ModuleList([Adapter(d_model=self.clip.visual.transformer.width, 
                                    bottleneck=self.config.adapter_dim, 
                                    dropout=self.config.adapter_dropout
                                ) for _ in range(adapter_num)])
        return params
    def encode_image(self, x: torch.Tensor):
        return self.encode_image_with_adapter(x)


    def encode_image_with_adapter(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        # img_feature = self.clip.visual.transformer(x)
        for i_block in range(self.clip.visual.transformer.layers-1):
            # MHA
            adapt_x = self.additional_visual_params[i_block](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].attention(
                self.clip.visual.transformer.resblocks[i_block].ln_1(x)
            )
            x = x + adapt_x + residual

            # FFN
            i_adapter = i_block + self.clip.visual.transformer.layers
            adapt_x = self.additional_visual_params[i_adapter](x, add_residual=False)
            residual = x
            x = self.clip.visual.transformer.resblocks[i_block].mlp(
                self.clip.visual.transformer.resblocks[i_block].ln_2(x)
            )
            x = x + adapt_x + residual

        img_feature = x.permute(1, 0, 2)  # LND -> NLD

        img_feature = self.clip.visual.ln_post(img_feature)
        if self.clip.visual.proj is not None:
            img_feature1 = img_feature @ self.clip.visual.proj
        return img_feature1[:, 0, :], img_feature

    def construct_soft_prompt(self):
        token_text = self.tokenizer("a photo of x x",padding="max_length",truncation=True,
            max_length=self.config.context_length,
            return_tensors="pt",)
        token_ids=token_text.input_ids.cuda()
        token_mask=token_text.attention_mask.cuda()
        tokenized = torch.cat(
            [
                self.tokenizer(tok, padding="max_length",truncation=True,
            max_length=self.config.context_length,
            return_tensors="pt",).input_ids
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding =self.model.Qformer.bert.embeddings(tokenized.cuda())

        # with torch.no_grad():
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = torch.nonzero(tokenized[idx]==102)
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = self.tokenizer(ctx_init,
                            padding="max_length",truncation=True,
            max_length=self.config.context_length,
            return_tensors="pt",).input_ids.cuda()
        prompt_end=torch.nonzero(prompt[0]==102)
        with torch.no_grad():
            embedding = self.model.Qformer.bert.embeddings(prompt)
        ctx_vectors = embedding[0, 1 : prompt_end, :]
        return token_ids, soft_att_obj, ctx_vectors,token_mask


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.model.Qformer.bert.embeddings(
            class_token_ids.cuda()
        ).type(self.model.Qformer.bert.dtype)
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        eos_idx = int(torch.nonzero(self.token_ids[0]==102))
        token_tensor[:, eos_idx - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.model.Qformer.bert.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.model.Qformer.bert.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.model.Qformer.bert.dtype)
        return token_tensor
    
    def forward(self, batch_img, idx):
        b = batch_img.shape[0]
        l, _ = idx.shape
        # with self.model.maybe_autocast():
        #      image_embeds=self.model.visual_encoder(batch_img)
        #      image_embeds_frozen= self.model.ln_vision(image_embeds)
        # image_embeds_frozen = image_embeds_frozen.float()
        batch_img, batch_patch = self.encode_image(batch_img.type(self.clip.dtype))
        # img_ft=batch_patch#bs*257*1408
        # img_ft=self.fusion(None,img_ft.type(torch.float),idx, b)
        # img_ft=0.2*img_ft+0.8*image_embeds_frozen
        # img_ft=img_ft.permute(1,0,2)
        image_atts = torch.ones(
                batch_patch.size()[:-1], dtype=torch.long
            ).to(self.device) #bs*257
        query_tokens = self.model.query_tokens.expand(
                batch_patch.shape[0], -1, -1
            )  # bs*32*768

        query_output = self.model.Qformer.bert(
                query_embeds=query_tokens, 
                encoder_hidden_states=batch_patch,
                encoder_attention_mask=image_atts,
                return_dict=True,
                output_attentions=False
            )
        image_embeds = query_output.last_hidden_state  #bs*32*768

        image_features = F.normalize(image_embeds, dim=-1) #bs*32*768
        
        token_tensors = self.construct_token_tensors(idx)
        if self.token_masks.dim() == 3:
            extended_attention_mask = self.token_masks[:, None, :, :]
        elif self.token_masks.dim() == 2:
            extended_attention_mask = self.token_masks[:, None, None, :]
        extended_attention_mask = extended_attention_mask.to(
            dtype=self.model.Qformer.bert.dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        text_output=self.model.Qformer.bert.encoder(token_tensors,attention_mask=extended_attention_mask,return_dict=True)
        # txt_cls=text_output[0]
        text_ft=text_output.last_hidden_state #idx*10*768,BERT cls
        text_ft=text_ft.permute(1,0,2)
        text_features=text_ft[0,:,:] #idx*768
        text_ft=self.fusion(text_ft.type(torch.float))
        # img_ft,text_ft=self.fusion(img_ft.type(torch.float), text_ft.type(torch.float), idx, b)
        
        text_ft=text_ft.permute(1,0,2)
        text_ft=text_ft[:,0,:]
        text_features = self.weight * text_features + (1 - self.weight) * text_ft
        # image_features = self.weight * image_features + (1 - self.weight) * img_ft
        text_feat = F.normalize(
             text_features, dim=-1
        )  #5592*256
        
        sim_q2t=image_features@text_feat.t() 
        sim_i2t, _ = sim_q2t.max(1)
        logits = sim_i2t/self.model.temp
        return logits