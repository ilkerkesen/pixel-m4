import pickle
import argparse
import os
import os.path as osp
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer
from pixel import ViTModel, PIXELConfig
from transformers import AutoTokenizer, AutoModel, AutoConfig
from pixel.analysis.utils import resize_model_embeddings, batcher
from pixel.analysis.probe import Probe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Required parameters
    parser.add_argument("--model_id", default="Team-PIXEL/pixel-base-bigrams", type=str,
                        help="the name of transformer model to evaluate on")
    parser.add_argument("--data_dir", default="data/", type=str,
                        help="Path to probing data directory")
    parser.add_argument("--output_dir", default="outputs/analysis/linspector", type=str,
                        help="Output path.")
    parser.add_argument("--layer", default="all", type=str,
                        help="Which layer to probe on. Mention 'all' to probe on all layers")
    parser.add_argument("--seed", default=1111, type=int,
                        help="which seed to use")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="which max length to use")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size")
    parser.add_argument("--auth", default=None, type=str,
                        help="hf authentication token")
    parser.add_argument("--lang", default=None, type=str)
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
    params = {'task_dir': args.data_dir, 'usepytorch': True, 'kfold': 10, 'batch_size': 16,
              'tokenizer': tokenizer, 'model': model,
              'model_name': model_name, 'seed': args.seed, 'lang': args.lang}

    data_dir = osp.abspath(osp.expanduser(args.data_dir))
    all_tasks = os.listdir(data_dir)
    this_tasks = []
    for task in all_tasks:
        full_path = osp.join(data_dir, task, args.lang)
        if osp.isdir(full_path) and not task.startswith("Character"):
            this_tasks.append(task)

    output_dir = osp.abspath(osp.expanduser(args.output_dir))
    if not osp.isdir(output_dir):
        os.makedirs(output_dir)

    print('tasks = {}'.format(', '.join(this_tasks)))
    for task in this_tasks:
        probe = Probe(params=params, batcher=batcher, task=task, layer=args.layer)
        results = probe.run(params=params, batcher=batcher, task=task)
        
        filename = f"{model_name}-{args.lang}-{task}-layer_{args.layer}.pickle" 
        filepath = osp.join(output_dir, filename)
        with open(filepath, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

