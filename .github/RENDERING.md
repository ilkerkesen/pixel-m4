## Rendering guide
In the following, we will only discuss the `PangoCairoTextRenderer` as the `PyGameTextRenderer` is deprecated. Note that, our version of `PangoCairoTextRenderer` particularly implements the bigrams rendering strategy proposed in [Text Rendering Strategies for Pixel Language Models](https://aclanthology.org/2023.emnlp-main.628/).


### Loading a renderer
A [`PangoCairoTextRenderer`](https://github.com/ilkerkesen/pixel-m4/blob/main/src/pixel/data/rendering/pangocairo_renderer_bigrams_iso_char.py)object can be created  directly via its `__init__` method or using methods such as `from_pretrained`, `from_dict`, and `from_json_file` of the 
[`TextRenderingMixin`](https://github.com/ilkerkesen/pixel-m4/blob/main/src/pixel/data/rendering/rendering_utils.py#L434) class it 
inherits from. The easiest way is to use the `from_pretrained` method which supports loading from the Hugging Face hub or from disk. 
You need two components to successfully load a renderer using this method: 

**1)** a `text_renderer_config.json`. It can look as follows:

```json
{
  "background_color": "white",
  "dpi": 120,
  "font_color": "black",
  "font_file": "GoNotoCurrent.ttf",
  "font_size": 8,
  "max_seq_length": 529,
  "pad_size": 3,
  "pixels_per_patch": 16,
  "text_renderer_type": "PangoCairoTextRenderer"
}
```
The docstring of the [`PangoCairoTextRenderer`](https://github.com/ilkerkesen/pixel-m4/blob/main/src/pixel/data/rendering/pangocairo_renderer_bigrams_iso_char.py#L29)
class provides descriptions for these fields. Note that **`font_file` is the only mandatory** field.
All remaining ones default to the PIXEL-BIGRAMS and PIXEL-M4 settings.

**2)** a font file, which is typically of the format `.ttf` (TrueType font) or `.otf` (OpenType font).

The font file must match the `font_file` field in the `text_renderer_config.json`. It should be a font with large coverage such as the 
[`GoNotoCurrent.ttf`](https://github.com/satbyy/go-noto-universal) font used in PIXEL-BIGRAMS and PIXEL-M4.

Putting the two together, we can load a renderer with minimal functionality as follows:
```python
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer

# Load from Hugging Face hub
text_renderer = PangoCairoBigramsRenderer.from_pretrained("Team-PIXEL/pixel-m4")

# Load from folder on disk containing a config file and font file
text_renderer = PangoCairoBigramsRenderer.from_pretrained("/path/to/my/renderer/with/config/and/font/file")

```

#### Fallback fonts
A text renderer can optionally be loaded with a list of fallback fonts. **This is highly recommended**,
in particular when working with multilingual corpora or emoji. How does this work? You can create a directory that contains your fallback fonts.
We provide a script that downloads all available NotoSans fonts and the `AppleColorEmoji.ttf`
[here](https://github.com/ilkerkesen/pixel-m4/blob/main/scripts/data/download_fallback_fonts.py).
You can then load the renderer with fallback fonts included as follows:
```python
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer

text_renderer = PangoCairoBigramsRenderer.from_pretrained(
    "Team-PIXEL/pixel-m4", 
    fallback_fonts_dir="/path/to/fallback/fonts"
)
```

#### RGB Rendering
If you want to render images in RGB, e.g. because you want color emoji to display correctly (for rendering emoji you need fallback fonts as described above), you need to load the renderer with the `rgb` flag enabled:

```python
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer

text_renderer = PangoCairoBigramsRenderer.from_pretrained(
    "Team-PIXEL/pixel-m4", 
    fallback_fonts_dir="/path/to/fallback/fonts",
    rgb=True
)
```


### Inputs and outputs
#### Renderer inputs
The [`PangoCairoTextRenderer`](https://github.com/ilkerkesen/pixel-m4/blob/main/src/pixel/data/rendering/pangocairo_renderer_bigrams_iso_char.py) takes three types of inputs: a string, a 2-string tuple (i.e., pair of strings), or a list of strings. 
Different rendering functions (we will take a look at these later) are called in the backend based on the type of input, 
but the renderer is always called in the same way:

```python
from pixel.data.rendering.pangocairo_renderer_bigrams_iso_char import PangoCairoTextRenderer as PangoCairoBigramsRenderer

# Load renderer (w/ basic functionality)
text_renderer = PangoCairoBigramsRenderer.from_pretrained("Team-PIXEL/pixel-m4")

# Render a single string
example = "My cat loves oatmeal."
encoding = text_renderer(example)

# Render a pair of strings
example = ("What does my cat love?", "Oatmeal")
encoding = text_renderer(example)

# Render a list of strings (word-level rendering)
example = ["My", "cat", "loves", "oatmeal", "."]
encoding = text_renderer(example)
```

#### Renderer outputs
The renderer outputs an [`Encoding`](https://github.com/ilkerkesen/pixel-m4/blob/main/src/pixel/data/rendering/rendering_utils.py#L38) object, which contains the following fields:

```python
pixel_values: np.ndarray
sep_patches: List[int]
num_text_patches: int
word_starts: Optional[List[int]] = None
offset_mapping: Optional[List[Tuple[int, int]]] = None
overflowing_patches: Optional[List] = None
sequence_ids: Optional[List[Optional[int]]] = None
```

The classes' docstring provides descriptions for each of these fields. The most important fields are `pixel_values`, an array containing the pixel values of the rendered image, and `num_text_patches` which contains information on how many image patches contain actual text, i.e. are neither the black end-of-sequence patch nor any blank patches that may follow.

The fields `offset_mapping` and `overflowing_patches` are currently only supported when rendering a pair of strings because we primarily use them for question answering. Likewise `word_starts` are only supported when rendering a list of strings because we only need this information when rendering word-by-word, e.g. in POS tagging or NER.

The field `sep_patches` is currently not used by any downstream application but may be useful in the future. `sequence_ids` is currently not provided by the renderer either, but may be added in the future.
