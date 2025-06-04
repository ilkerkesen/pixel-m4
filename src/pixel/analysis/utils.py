from PIL import Image
import torch
import math
from pixel import ViTModel, get_transforms, get_attention_mask


def resize_model_embeddings(model: ViTModel, max_seq_length: int) -> None:
    """
    Checks whether position embeddings need to be resized. If the specified max_seq_length is longer than
    the model's number of patches per sequence, the position embeddings will be interpolated.
    If max_seq_length is shorter, the position embeddings will be truncated

    Args:
        model (`ViTModel`):
            The model for which position embeddings may be resized.
        max_seq_length (`int`):
            The maximum sequence length that determines the number of patches (excluding CLS patch) in the
            model.
    """
    patch_size = model.config.patch_size
    if isinstance(model.config.image_size, tuple) or isinstance(model.config.image_size, list):
        old_height, old_width = model.config.image_size
    else:
        old_height, old_width = (model.config.image_size, model.config.image_size)

    # ppr means patches per row (image is patchified into grid of [ppr * ppr])
    old_ppr = math.sqrt(old_height * old_width) // patch_size
    new_ppr = math.sqrt(max_seq_length)

    if old_ppr < new_ppr:
        # Interpolate position embeddings
        # logger.info(f"Interpolating position embeddings to {max_seq_length}")
        model.config.interpolate_pos_encoding = True
    elif old_ppr > new_ppr:
        # logger.info(f"Truncating position embeddings to {max_seq_length}")
        # Truncate position embeddings
        old_pos_embeds = model.embeddings.position_embeddings[:, : max_seq_length + 1, :]
        model.embeddings.position_embeddings.data = old_pos_embeds.clone()
        # Update image_size
        new_height = int(new_ppr * patch_size) if old_height == old_width else int(patch_size)
        new_width = int(new_ppr * patch_size) if old_height == old_width else int(patch_size * new_ppr ** 2)
        model.config.image_size = [new_height, new_width]
        model.image_size = [new_height, new_width]
        model.embeddings.patch_embeddings.image_size = [new_height, new_width]



def batcher(params, batch):
    model = params["model"]
    model_name = params["model_name"]

    if "bert" in model_name:
        tokenizer = params["tokenizer"]
        if "xlm" in model.config._name_or_path:
            batch = [["<s>"] + tokenizer.tokenize(sent) + ["</s>"] for sent in batch]
        else:
            batch = [["[CLS]"] + tokenizer.tokenize(sent) + ["[SEP]"] for sent in batch]
        batch = [b[:512] for b in batch]
        seq_length = max([len(sent) for sent in batch])
        mask = [[1] * len(sent) + [0] * (seq_length - len(sent)) for sent in batch]
        segment_ids = [[0] * seq_length for _ in batch]
        batch = [tokenizer.convert_tokens_to_ids(sent) + [0] * (seq_length - len(sent)) for sent in batch]
        with torch.no_grad():
            batch = torch.tensor(batch).cuda()
            mask = torch.tensor(mask).cuda()  # bs * seq_length
            segment_ids = torch.tensor(segment_ids).cuda()
            outputs, pooled_output, hidden_states, _ = model(batch, token_type_ids=segment_ids, attention_mask=mask,
                                                             return_dict=False)

        extended_mask = mask.unsqueeze(-1)

    else:
        processor = params["tokenizer"]
        if "vit" in model.config._name_or_path:
            transforms = get_transforms(
                do_resize=False,
                do_squarify=True
            )
        else:
            transforms = get_transforms(
                do_resize=True,
                size=(processor.pixels_per_patch, processor.pixels_per_patch * processor.max_seq_length),
            )

        encodings = [processor(a, preprocessor="whitespace_only") for a in batch]
        pixel_values = [transforms(Image.fromarray(e.pixel_values)) for e in encodings]
        attention_mask = [
            get_attention_mask(e.num_text_patches, seq_length=processor.max_seq_length) for e in encodings
        ]

        with torch.no_grad():
            batch = torch.stack(pixel_values).cuda()
            mask = torch.stack(attention_mask).cuda()  # bs * seq_length
            outputs, pooled_output, hidden_states, _ = model(batch, attention_mask=mask, return_dict=False)

        extended_mask = torch.cat((torch.ones(mask.shape[0], 1).cuda(), mask), -1).unsqueeze(-1)

    embeddings = {}
    for layer in range(0, 13):
        output = hidden_states[int(layer)]
        output = extended_mask * output
        output = torch.sum(output, -2) / torch.sum(mask, -1).unsqueeze(-1)
        embeddings[layer] = output.data.cpu().numpy()

    return embeddings


