import os 
import os.path as osp
import json
import multiprocessing as mp
import unicodedata
from tqdm import tqdm
import time
import argparse

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata

import cairo
import gi
gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo

# NOTE
# include argparse: out path; renderer config 
# name results file according to renderer config  

# out_path = "/scratch/project/open-28-68/data/unicode-hash-map"

# NOTE adapt to testpixel env
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer

class MyRenderer(PangoCairoBigramsRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _render_text_to_surface(
        self,
        text: list,
        rtl: bool = False,
        max_length: Union[None, int] = None,
        **kwargs,
    ):
        surface, context, sep_patches = self.get_empty_surface()

        offset = 0
        text_width = self._render_single_sentence(
            text, context=context, input_offset=offset, max_length=max_length, rtl=rtl
            )
        return text_width
    
    def _render_single_sentence(
        self, 
        words: List[str], 
        context,
        input_offset: int,
        rtl: bool = False, 
        max_length: Optional[int] = None, 
        **kwargs,
    ):
        return self._offset_bigram(words, input_offset, context)
        
    def _offset_bigram(
        self, word: str, offset: int, context: cairo.Context, is_last: bool = False
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        layout1 = PangoCairo.create_layout(context)
        layout1.set_font_description(self.font)

        layout1.set_text(word[0], -1)

        if layout1.get_unknown_glyphs_count() > 0:
            pass
            # print(
            #     f"Found {layout1.get_unknown_glyphs_count()} unknown glyphs in word: {word}. Consider "
            #     f"double-checking that the correct fonts are loaded."
            # )

        # Get logical extents
        width1, height1 = layout1.get_pixel_size()
        return width1
    
    def __call__(
        self,
        text: Union[str, Tuple[str, str], List[str]],
    ):
        fn = self._render_text_to_surface
        return fn(text)


def X(renderer, char):
    return {char: renderer(char)}


def worker(output_path, renderer, char, lock, progress):
    result = X(renderer, char)
    with lock:
        with open(output_path, 'a') as f:
            json.dump(result, f)
            f.write('\n')  
    with progress.get_lock():
        progress.value += 1


def unicode_processor(output_path, renderer, start=0x0000, end=0x007F, debug=False):
    lock = mp.Lock()
    progress = mp.Value('i', 0)
    processes = []

    start_time = time.time()
    # Discard Non-class and surrogates 
    valid_chars = [i for i in range(start, end+1) if not unicodedata.category(chr(i)) in ['Cn', 'Cs']]
    end_time = time.time()

    print(f'Time taken to filter characters: {end_time - start_time} seconds')

    with tqdm(total=len(valid_chars)) as pbar:
        for i in valid_chars:
            char = chr(i)
            
            if debug:
                result = X(renderer, char)
                with open(output_path, 'a') as f:
                    json.dump(result, f)
                    f.write('\n')
                pbar.update()
            else:
                p = mp.Process(target=worker, args=(output_path, renderer, char, lock, progress))
                p.start()
                processes.append(p)

                if len(processes) >= mp.cpu_count() or pbar.n >= len(valid_chars):
                    for p in processes:
                        p.join()
                    processes = []
                pbar.update(progress.value - pbar.n)

        # Make sure the last batch of processes finish
        if not debug:
            for p in processes:
                p.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute unicode bigram widths.")
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to save unicode bigram widths."
    )
    parser.add_argument(
        "--fallback-fonts-dir",
        type=str,
        help="Path to the fallback fonts dir.",
    )
    args = parser.parse_args()
    output_path = osp.abspath(osp.expanduser(args.output_path))
    fallback_fonts_dir = osp.abspath(osp.expanduser(args.fallback_fonts_dir))

    assert output_path.endswith(".jsonl")
    if not osp.isdir(osp.dirname(output_path)):
        os.makedirs(osp.dirname(output_path))

    renderer = MyRenderer.from_pretrained(
        "Team-PIXEL/pixel-base",
        fallback_fonts_dir=fallback_fonts_dir,
        rgb=False   
    )

    unicode_processor(
        output_path=output_path,
        renderer=renderer,
        start=0x0,
        end=0x10FFFF,
        debug=True,
    )
    print("done.")
    # unicode_processor(start=0x0000, end=0xE007F, debug=True) 
