import argparse
import os
import os.path as osp
import numpy as np
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer
from pixel import ViTModel, PIXELConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pixel.analysis.utils import resize_model_embeddings, batcher
from pixel.analysis.tsne import TSNEExperiment


LANGUAGES = 'arb_Arab arb_Latn arz_Arab azb_Arab azj_Latn ben_Beng deu_Latn eng_Latn fin_Latn fra_Latn hin_Deva jpn_Jpan kas_Arab kas_Deva kir_Cyrl kor_Hang rus_Cyrl tam_Taml taq_Latn taq_Tfng tel_Telu tur_Latn uig_Arab ukr_Cyrl urd_Arab uzn_Latn zho_Hans zho_Hant'.split()

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
    parser.add_argument("--perplexity", default=50, type=int, help="Perplexity value for t-SNE.")
    parser.add_argument("--n_iter", default=1000, type=int, help="Number of iterations for t-SNE.")
    parser.add_argument("--lr", default=500, type=int, help="Learning rate for t-SNE.")
    parser.add_argument("--output_dir", default="outputs/tsne", type=str, help="Output path.")
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

    params = {
        'model': model,
        'model_name': args.model_id.split('/')[-1],
        'tokenizer': tokenizer,
        'seed': args.seed,
        'languages': LANGUAGES,
        'batch_size': args.batch_size,
    }
    model_name = args.model_id.split('/')[-1]
    n_components = 2
    
    tsne_exp = TSNEExperiment(params=params, batcher=batcher, layer="all")
    tsne_results, labels = tsne_exp.run(
        params=params,
        batcher=batcher,
        n_components=n_components, perplexity=args.perplexity, learning_rate=args.lr, n_iter=args.n_iter,    
    )

    output_dir = osp.abspath(osp.expanduser(args.output_dir))
    if not osp.isdir(output_dir):
        os.makedirs(output_dir)

    for layer_idx, emb in tsne_results.items():
        fname = (
            f"{model_name}"
            f"-layer={layer_idx}"
            f"-n_components={n_components}"
            f"-perp={args.perplexity}"
            f"-lr={args.lr}"
            f"-n_iter={args.n_iter}.npz"
        )
        path = osp.join(output_dir, fname)

        # save both embeddings and labels
        np.savez_compressed(path,
            embeddings=emb,
            labels=labels,
        )

        print(f"Saved: {path}")

