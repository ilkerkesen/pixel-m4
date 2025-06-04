import argparse
import os
import os.path as osp
import numpy as np
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer
from pixel import ViTModel, PIXELConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pixel.analysis.utils import resize_model_embeddings, batcher
from pixel.analysis.retrieval import RetrievalExperiment


LANGUAGES = 'eng_Latn hin_Deva ukr_Cyrl zho_Hans'.split()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Required parameters
    parser.add_argument("--model_id", default="Team-PIXEL/pixel-base-bigrams", type=str,
                        help="the name of transformer model to evaluate on")
    parser.add_argument("--layer", default="all", type=str,
                        help="Which layer to probe on. Mention 'all' to probe on all layers")
    parser.add_argument("--seed", default=1111, type=int,
                        help="which seed to use")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="which max length to use")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size")
    parser.add_argument("--output_dir", default="outputs/analysis/retrieval", type=str,
                        help="Output path.")
    parser.add_argument("--auth", default=None, type=str,
                        help="hf authentication token")
    args = parser.parse_args()

    if "bert" in args.model_id:
        config = AutoConfig.from_pretrained(args.model_id)
        config.output_hidden_states = True
        config.output_attentions = True
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModel.from_pretrained(args.model_id, config=config).cuda()
        model.eval()

    else:
        access_token = args.auth
        renderer_cls = PangoCairoBigramsRenderer
        tokenizer = renderer_cls.from_pretrained(
            args.model_id,
            rgb=False,
            max_seq_length=args.max_seq_length,
            fallback_fonts_dir="fallback_fonts",
            use_auth_token=access_token
        )

        config = PIXELConfig.from_pretrained(args.model_id, use_auth_token=access_token)
        config.output_hidden_states = True
        config.output_attentions = True
        model = ViTModel.from_pretrained(args.model_id, config=config, use_auth_token=access_token).cuda()
        resize_model_embeddings(model, args.max_seq_length)
        model.eval()

    model_name = args.model_id.split('/')[-1]
    output_dir = osp.abspath(osp.expanduser(args.output_dir))
    if not osp.isdir(output_dir):
        os.makedirs(output_dir)

    params = {
        'model': model,
        'model_name': model_name,
        'tokenizer': tokenizer,
        'seed': args.seed,
        'languages': LANGUAGES,
        'batch_size': args.batch_size,
    }

    exp = RetrievalExperiment(params=params, batcher=batcher, layer="all")
    scores, labels, indices = exp.run()
    for layer_idx, emb in scores.items():
        fname = (
            f"{model_name}"
            f"-layer={layer_idx}.npz"
        )
        path = osp.join(output_dir, fname)

        # save both embeddings and labels
        np.savez_compressed(
            path,
            embeddings=emb,
            labels=labels,
            indices=indices,
        )

        print(f"Saved {path}")

