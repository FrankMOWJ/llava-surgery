from transformers import PreTrainedTokenizer
from transformers import GPT2Tokenizer
import torch
from transformers import LlamaTokenizer
from PIL import Image

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200

def tokenizer_image_token(prompt, tokenizer, image_token_index=-200, return_tensors=None):
    # prompt : str
    """
    eg. "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. 
    USER: <image>\nReaching for needle with left hand ASSISTANT: 0.11944700,-0.03814500,0.40989800,-0.49762400,0.44487600,-0.74464800,-0.44848800,-0.86673500,-0.21817500,-0.74186100,0.22483800,0.63175800,-0.00137800,-0.02335500,0.01712200,-0.06154800,-0.03134100,0.12498500,0.17988100,-0.11427300,-0.03232500,0.40882600,0.12257000,0.35452100,0.92697500,-0.06269600,0.93493200,-0.34915000,-0.99056200,-0.01578700,0.13565000,0.01455200,-0.00292000,-0.02010500,0.04169800,0.32313500,-0.09774100,-1.19700000,0.03781100,0.02590100,-0.00729900,-0.04212500,-0.40001500,0.91553900,0.17839800,-0.90463800,-0.38704400,0.98305500,0.14702600,0.10946900,0.01403000,0.00003900,-0.00967700,0.03572500,0.30836000,-0.03105200,-0.83192700,0.06940900,0.01697100,-0.06562200,-0.57637500,0.37254400,-0.72732500,-0.28123300,-0.92609800,-0.25149300,-0.76726700,0.05959400,0.63855100,-0.00061300,-0.01127700,0.00885200,-0.07526200,0.02509000,-0.03849700,0.01614700</s>"
    """
    # 首先分理处<image>前后的句子
    prompt1 = prompt.split('<image>')
    prompt2 = prompt1[1].split("ASSISTANT: ")
    # print(prompt2[1].split("</s>")[0])
    # format of prompt_lst : [[before <image>], [between <image> and ASSISTANT: ]]
    # [After ASSISTANT:] need not to tokenize, just need to change to list
    kinematic_info = [float(info) for info in prompt2[1].split("</s>")[0].split(',')]
    # print(kinematic_info)
    assert len(kinematic_info) == 76, "there are 76 kinematic info per sampler"

    prompt_lst = [prompt1[0], prompt2[0]+"ASSISTANT: "]
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt_lst]
    # prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    # 每一个字符串会较少起始字符ID （1）

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id: # <bos> : beginning of sentence
        offset = 1
        # 首先插入起始token的ID
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        # x[offset:] 是想把起始token去掉， 这也是为什么[image_token_index] * (1 + 1)
        input_ids.extend(x[offset:])
    # insert_sepator()的作用时将<image>token 插入到prompt中，最后实现的效果是[prompt_chunk[0] + IMAGE_TOKEN_INDEX + prompr_chunk[1]]
    
    # add the [After ASSISTANT:] part back to input_ids
    # input_ids.extend(kinematic_info)

    if return_tensors is not None:
        if return_tensors == 'pt':
            # 要转成torch.float 才能保留浮点数
            return torch.tensor(input_ids, dtype=torch.long), torch.tensor(kinematic_info, dtype=torch.float)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

# 这里实现了将token 转成embedding （4096）
def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, image_sizes=None
):
    vision_tower = self.get_vision_tower()
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        return input_ids, position_ids, attention_mask, past_key_values, None, labels

    if type(images) is list or images.ndim == 5:
        if type(images) is list:
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
        image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
        if mm_patch_merge_type == 'flat':
            image_features = [x.flatten(0, 1) for x in image_features]
        elif mm_patch_merge_type.startswith('spatial'):
            new_image_features = []
            for image_idx, image_feature in enumerate(image_features):
                if image_feature.shape[0] > 1:
                    base_image_feature = image_feature[0]
                    image_feature = image_feature[1:]
                    height = width = self.get_vision_tower().num_patches_per_side
                    assert height * width == base_image_feature.shape[0]
                    if image_aspect_ratio == 'anyres':
                        num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                        image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                    else:
                        raise NotImplementedError
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                        image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                        image_feature = unpad_image(image_feature, image_sizes[image_idx])
                        image_feature = torch.cat((
                            image_feature,
                            self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                        ), dim=-1)
                        image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                    else:
                        image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                        image_feature = image_feature.flatten(0, 3)
                    image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                else:
                    image_feature = image_feature[0]
                    if 'unpad' in mm_patch_merge_type:
                        image_feature = torch.cat((
                            image_feature,
                            self.model.image_newline[None].to(image_feature.device)
                        ), dim=0)
                new_image_features.append(image_feature)
            image_features = new_image_features
        else:
            raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
    else:
        # image features shape [bs, 576, 4096]
        image_features = self.encode_images(images)

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.
    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    else:
        attention_mask = attention_mask.bool()
    if position_ids is None:
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
    if labels is None:
        labels = torch.full_like(input_ids, IGNORE_INDEX)

    # remove the padding using attention_mask -- FIXME
    # TODO： 这里要改！ 
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    # labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        # 有多少张image
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        if num_images == 0:
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            continue
        
        # [-1, index_of_image_token, seq_len] 放入 -1 是为了适配下面的image_token_indices[i]+1
        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            # cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        
        split_sizes = [x.shape[0] for x in cur_input_ids_noim] # 这里做了改动
        # 做了embbedding torch.cat(cur_input_ids_noim) 就是出去image_token之后的序列
        # seq_len = 213
        # cur_input_embeds shape : [212, 4096]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        cur_new_input_embeds = []
        cur_new_labels = []

        # 这一部分是将image_feature插入到embedding中
        """
        eg. cur_new_input_embeds shape = [[35, 4096], [576, 4096(image_feature)], [177, 4096]]
            cur_new_labels shape = [[35], [576(IGNORE_INDEX), [177]]]
        """
        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            # cur_new_labels.append(cur_labels_noim[i])
            if i < num_images:
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                # cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        # [35+576+177, 4096]
        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        # [35+576+177]
        # cur_new_labels = torch.cat(cur_new_labels)

        # 装回一个batch
        new_input_embeds.append(cur_new_input_embeds)
        # new_labels.append(cur_new_labels)

    # Truncate sequences to max length as image embeddings can make the sequence longer
    # max_length = 2048
    tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    if tokenizer_model_max_length is not None:
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        # new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    batch_size = len(new_input_embeds)

    new_input_embeds_padded = []
    # new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

    # 把长度不是max_len的句子pad成max_len(补0)
    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            new_input_embeds_padded.append(torch.cat((
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                cur_new_embed
            ), dim=0))
            if cur_len > 0:
                # new_labels_padded[i, -cur_len:] = cur_new_labels
                attention_mask[i, -cur_len:] = True
                position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
        else:
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                # new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

    if _labels is None:
        new_labels = None
    else:
        new_labels = labels

    if _attention_mask is None:
        attention_mask = None
    else:
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

    if _position_ids is None:
        position_ids = None

    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    
if __name__ == "__main__":
    # prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\nReaching for needle with left hand ASSISTANT: 0.11944700,-0.03814500,0.40989800,-0.49762400,0.44487600,-0.74464800,-0.44848800,-0.86673500,-0.21817500,-0.74186100,0.22483800,0.63175800,-0.00137800,-0.02335500,0.01712200,-0.06154800,-0.03134100,0.12498500,0.17988100,-0.11427300,-0.03232500,0.40882600,0.12257000,0.35452100,0.92697500,-0.06269600,0.93493200,-0.34915000,-0.99056200,-0.01578700,0.13565000,0.01455200,-0.00292000,-0.02010500,0.04169800,0.32313500,-0.09774100,-1.19700000,0.03781100,0.02590100,-0.00729900,-0.04212500,-0.40001500,0.91553900,0.17839800,-0.90463800,-0.38704400,0.98305500,0.14702600,0.10946900,0.01403000,0.00003900,-0.00967700,0.03572500,0.30836000,-0.03105200,-0.83192700,0.06940900,0.01697100,-0.06562200,-0.57637500,0.37254400,-0.72732500,-0.28123300,-0.92609800,-0.25149300,-0.76726700,0.05959400,0.63855100,-0.00061300,-0.01127700,0.00885200,-0.07526200,0.02509000,-0.03849700,0.01614700</s>"
    # tokenizer = LlamaTokenizer.from_pretrained("/ssd/bailong/LLaVA/pretrain_model/models--lmsys--vicuna-7b-v1.5")
    # input_ids, labels = tokenizer_image_token(prompt=prompt, 
    #                     tokenizer=tokenizer,
    #                     return_tensors="pt")

    # print(input_ids.float().dtype)
    # print(labels.dtype)

    instances = [[torch.rand(3, 336, 336) for i in range(3)], [torch.rand(3, 336, 336) for i in range(3)]]
    instances = [torch.stack(instance) for instance in instances]
    instances = torch.stack(instances)
    print(instances.shape)
    # instances = [torch.rand(3, 336, 336) for i in range(6)]
    # print(instances[0].shape)
    print(torch.stack(instances).view(instances.shape[0], -1, 3, 336,336).shape)