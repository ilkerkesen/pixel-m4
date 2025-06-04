import glob
import logging
import math
import os
import string
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata

import cairo
import gi
import manimpango
import numpy as np
from fontTools import ttLib

gi.require_version("Pango", "1.0")
gi.require_version("PangoCairo", "1.0")
from gi.repository import Pango, PangoCairo

from ...utils.defaults import *
from .rendering_utils import Encoding, TextRenderingMixin

logger = logging.getLogger(__name__)

SUPPORTED_INPUT_TYPES = [str, Tuple[str, str], List[str]]

# NOTE : for QA and Arabic, try passing both as list and text now that list is not reordered
#   to see if the pos. emb. ordering also makes a difference for sentence level tasks

class PangoCairoTextRenderer(TextRenderingMixin):
    """
    Constructs a text renderer using Pango and Cairo as rendering backend.
    This feature extractor inherits from [`TextRenderingMixin`] which contains most of the main methods. Users
    should refer to this superclass for more information regarding those methods.

    Args:
        font_file (`str`):
            The font file (typically a file with a .ttf or .otf extension) that is loaded to render text
        font_size (`int`, defaults to 8):
            The font size that is used when rendering text
        font_color (`str`, defaults to "black"):
            The font color that is used when rendering text
        background_color (`str`, defaults to "white"):
            The background color of the image
        rgb (`bool`, defaults to False):
            Whether or not to render images in RGB. RGB rendering can be useful when working with emoji but it makes
            rendering a bit slower, so it is recommended to turn on RGB rendering only when there is need for it
        dpi (`int`, defaults to 120):
            The dpi (dots per inch) count that determines the resolution of the rendered images
        pad_size (`int`, defaults to 3):
            The amount of padding that is applied. Note: Currently, dynamic padding is not supported so this argument
            does not do anything
        pixels_per_patch (`int`, defaults to 16):
            The number of pixels, both horizontally and vertically, of each patch in the rendered image
        max_seq_length (`int`, defaults to 529):
            The maximum number of patches which, when multiplied with pixels_per_patch, determines the width of each
            rendered image
        fallback_fonts_dir (`str`, *optional*, defaults to None):
            Path to a directory containing font files (.ttf or .otf) which will be registered as fallback fonts. This
            can be useful when working with datasets with a large Unicode range

    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        font_file: str,
        font_size: int = DEFAULT_FONT_SIZE,
        font_color: str = "black",
        background_color: str = "white",
        rgb: bool = False,
        dpi: int = 120,
        pad_size: int = DEFAULT_PAD_SIZE,
        pixels_per_patch: int = DEFAULT_PPB,
        max_seq_length: int = MAX_SEQ_LENGTH,
        fallback_fonts_dir: Optional[str] = None,
        **kwargs,
    ):

        super().__init__(**kwargs)

        self.font_file = font_file
        self.font_size = font_size
        self.font_color = font_color
        self.background_color = background_color
        self.rgb = rgb

        self.pixels_per_patch = pixels_per_patch
        self.max_seq_length = max_seq_length
        self.pad_size = pad_size
        self.pad_left, self.pad_right, self.pad_top, self.pad_bottom = (pad_size, pad_size, pad_size, pad_size)

        self.dpi = dpi

        self.font = None
        self.fonts_list = None
        self.fallback_fonts_dir = fallback_fonts_dir
        self.load_font()

        self.PANGO_SCALE = 1024

    @property
    def max_pixels_len(self):
        return self.max_seq_length * self.pixels_per_patch

    def px2patch_ceil(self, px: int):
        return math.ceil(px / self.pixels_per_patch)

    def px2patch_floor(self, px: int):
        return math.floor(px / self.pixels_per_patch)

    def patch2px(self, patch: int):
        return patch * self.pixels_per_patch

    @staticmethod
    def is_rtl(text: str) -> bool:
        """
        Returns whether a piece of text is written in a right-to-left (RTL) script based on a majority vote of the
        first, middle, and last characters in the text after removing whitespace, punctuation, and numbers

        Returns:
            Whether the piece of text is RTL, type `bool`
        """
        text = text.translate(str.maketrans("", "", string.whitespace))
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = text.translate(str.maketrans("", "", string.digits))

        if len(text) == 0:
            return False

        vote = 0
        for char in [text[0], text[-1], text[len(text) // 2]]:
            if Pango.unichar_direction(char) == Pango.Direction.RTL:
                vote += 1

        is_rtl = vote >= 2
        # if not is_rtl:
        #    print(sys._getframe().f_back.f_code.co_name)
        #    print(f"{text[0] = }, {text[-1] = }, {text[len(text)//2] = }")
        return is_rtl

    def __getstate__(self) -> Dict[str, Any]:
        """
        Returns the state dict of the renderer without the loaded font to make it pickleable

        Returns:
            The state dict of type `Dict[str, Any]`
        """

        return {
            "font_file": self.font_file,
            "font_size": self.font_size,
            "font_color": self.font_color,
            "background_color": self.background_color,
            "rgb": self.rgb,
            "dpi": self.dpi,
            "pad_size": self.pad_size,
            "pixels_per_patch": self.pixels_per_patch,
            "max_seq_length": self.max_seq_length,
            "fonts_list": self.fonts_list,
        }

    def __setstate__(self, state_dict: Dict[str, Any]) -> None:
        """
        Sets the state dict of the renderer, e.g. from a pickle

        Args:
            state_dict (`Dict[str, Any]`):
                The state dictionary of a `PangoCairoTextRenderer`, containing all necessary and optional fields to
                initialize a `PangoCairoTextRenderer`
        """

        self.__init__(**state_dict)

    def _get_offset_to_next_patch(self, x: int) -> int:
        """
        Get the horizontal position (offset) where the next patch begins, based on how many pixels a patch contains
        and the maximum width

        Args:
            x (`int`):
                The horizontal position from where the next patch is to be found

        Returns:
            The starting position of the next patch (offset) of  type `int`
        """

        return min(
            math.ceil(x / self.pixels_per_patch) * self.pixels_per_patch,
            self.max_pixels_len - self.pixels_per_patch,
        )
        
        
    def _get_unbounded_offset_to_next_patch(self, x: int) -> int:
        """
        Get the horizontal position (offset) where the next patch begins, based on how many pixels a patch contains
        and the maximum width

        Args:
            x (`int`):
                The horizontal position from where the next patch is to be found

        Returns:
            The starting position of the next patch (offset) of  type `int`
        """

        return math.ceil(x / self.pixels_per_patch) * self.pixels_per_patch

    def _get_offset_to_previous_patch(self, x: int) -> int:
        """
        Get the horizontal position (offset) where the previous patch begins, based on how many pixels a patch contains
        and the maximum width

        Args:
            x (`int`):
                The horizontal position from where the next patch is to be found

        Returns:
            The starting position of the next patch (offset) of  type `int`
        """

        return math.floor(x / self.pixels_per_patch) * self.pixels_per_patch

    def get_empty_surface(self) -> Tuple[cairo.ImageSurface, cairo.Context, List[int]]:
        """
        Create and return a tuple containing
        (1) an empty surface that we will later render the text to,
        (2) a context object used to draw on the surface, and
        (3) an empty list in which we keep track of where to insert black separator patches

        Returns:
            A tuple of type (`~cairo.ImageSurface`, `~cairo.Context`, `List[int]`) containing the blank surface,
            the context object, an the empty list for keeping track of black separator patches, respectively
        """

        cairo_format = cairo.FORMAT_RGB24 if self.rgb else cairo.FORMAT_A8
        surface = cairo.ImageSurface(cairo_format, self.max_pixels_len, self.pixels_per_patch)
        context = cairo.Context(surface)
        if self.rgb:
            context.set_source_rgb(1.0, 1.0, 1.0)
            context.rectangle(0, 0, self.max_pixels_len, self.pixels_per_patch)
            context.fill()
            context.set_source_rgb(0.0, 0.0, 0.0)
        sep_patches = []
        return surface, context, sep_patches

    def get_cluster_idx_and_logical_widths(self, layout_iter: Pango.LayoutIter):
        """
        Returns the logical extents (first pixel in text direction) at the grapheme cluster level for a given index

        Args:
            layout_iter (`Pango.LayoutIter`):
                An object used to iterate over a pango layout (here, cluster-by-cluster).
        """
        logical_extents = layout_iter.get_cluster_extents()[1]
        x_offset = logical_extents.x / self.PANGO_SCALE
        idx = layout_iter.get_index()
        return idx, x_offset

    def get_char_idx_and_logical_widths(self, layout_iter: Pango.LayoutIter):
        """
        Returns the logical extents (first pixel in text direction) at the character level for a given index

        Args:
            layout_iter (`Pango.LayoutIter`):
                An object used to iterate over a pango layout (here, character-by-character).
        """
        logical_extents = layout_iter.get_char_extents()
        x_offset = logical_extents.x / self.PANGO_SCALE
        idx = layout_iter.get_index()
        return idx, x_offset

    def get_text_offset_mapping(
        self, layout: Pango.Layout, offset: int, text_width: int, text_shift: int = 0, rtl: bool = False
    ) -> List[Tuple[int, int]]:
        """
        Returns an offset mapping, i.e. a list that keeps track of where in the rendered image each character of
        the input text is located. It has the form [(start_character_index, end_character_index)] with an entry for
        every image patch.

        Args:
            layout (`Pango.Layout`):
                The layout containing the rendered text.
            offset (`int`):
                The offset in pixels of the first character of the text from the beginning of the first patch.
            text_width (`int`):
                The logical width of the rendered text in pixels.
            text_shift (`int`, *optional*, defaults to 0):
                The number of pixels that a text is shifted to the right on the layout, i.e. the starting position
                as pixel offset of the first image patch corresponding to this text. This value is typically set when
                obtaining the offset_mapping for the second text in a rendered text pair.
            rtl (`bool`, *optional*, defaults to False):
                Indicates whether the text is rendered right-to-left (RTL), in which case the offset mapping needs to
                account for the fact that the actual beginning of the text is on the right.
        """
        # Find starting positions for each character in the text
        layout_iter = layout.get_iter()
        # Get offset for first character
        idx, x_offset = self.get_char_idx_and_logical_widths(layout_iter)
        character_positions = [x_offset + offset]
        # Loop through remaining characters
        while layout_iter.next_char():
            idx, x_offset = self.get_char_idx_and_logical_widths(layout_iter)
            character_positions.append(x_offset + offset)

        # Find starting positions for each cluster in the text. A cluster may consist of multiple characters rendered
        # as one glyph
        layout_iter = layout.get_iter()
        # Get offset for first cluster
        idx, x_offset = self.get_cluster_idx_and_logical_widths(layout_iter)
        cluster_positions = [x_offset + offset]
        # Loop through remaining clusters
        while layout_iter.next_cluster():
            idx, x_offset = self.get_cluster_idx_and_logical_widths(layout_iter)
            cluster_positions.append(x_offset + offset)

        # In case clusters exist, the length of the cluster list will be shorter than the length of the character list.
        # However, the offset mapping maps between clusters in the rendered image and characters in the written text,
        # so we need to assign a starting position to each character in the cluster position list. We do this by
        # assigning the starting position of a cluster to each character in that cluster.
        if len(character_positions) != len(cluster_positions):
            buffer = []
            cluster_idx = 0
            for idx in range(len(character_positions)):
                if cluster_idx == len(cluster_positions) or character_positions[idx] != cluster_positions[cluster_idx]:
                    buffer.append(cluster_positions[cluster_idx - 1])
                else:
                    buffer.append(character_positions[idx])
                    cluster_idx += 1

            buffered_cluster_positions = buffer
        else:
            buffered_cluster_positions = character_positions

        # Retrieve the rendered text from the layout. This is necessary for RTL scripts
        text = layout.get_text()

        # This means we add a full blank patch
        if self._get_offset_to_next_patch(text_width) - text_width < offset - self._get_offset_to_previous_patch(
            offset
        ):
            is_blank_patch_inserted = True
        else:
            is_blank_patch_inserted = False

        buffered_cluster_positions.append(self._get_offset_to_next_patch(text_width + offset))

        offset_mapping = []
        patch_start = 0
        cleared = 0
        for k, v in enumerate(buffered_cluster_positions):
            if v - text_shift >= self.pixels_per_patch * (len(offset_mapping) + 1):
                if v - text_shift == self.pixels_per_patch * (len(offset_mapping) + 1):
                    patch_end = k
                else:
                    patch_end = k - 1
                offset_mapping.append(
                    (
                        (len(text) - patch_start) if rtl else patch_start,
                        (len(text) - patch_end) if rtl else patch_end,
                    )
                )

                patch_start = patch_end
                cleared += 1

        # The `cleared` variable counts how many times we have added a character span to the offset mapping, i.e.,
        # cleared the cluster buffer. If at the end of processing the buffered_cluster_positions we still have clusters
        # in the buffer, we add the remainder to the offset mapping
        if cleared < self.px2patch_ceil(text_width + offset - text_shift):
            if rtl:
                offset_mapping.append((len(text) - patch_start, 0))
            else:
                offset_mapping.append((patch_start, len(buffered_cluster_positions)))

        # We add padding between the end of the rendered sequence and the final black separator patch. If this padding
        # happens to be a full patch, this means that we need to merge the penultimate and last patches in the offset
        # mapping and add a buffer to the offset mapping
        if is_blank_patch_inserted:
            offset_mapping[-2] = (
                offset_mapping[-2][0],
                offset_mapping[-1][1],
            )
            offset_mapping[-1] = (-1, -1)

        # print(f"{len(offset_mapping) = }")

        return offset_mapping

    def pad_or_truncate_offset_mapping(self, offset_mapping: List[Tuple[int, int]]):
        if len(offset_mapping) >= self.max_seq_length:
            offset_mapping = offset_mapping[: self.max_seq_length - 1] + [(0, 0)]
        if len(offset_mapping) < self.max_seq_length:
            offset_mapping += (self.max_seq_length - len(offset_mapping)) * [(0, 0)]
        return offset_mapping

    def _render_single_word(
        self, word: str, offset: int, context: cairo.Context, is_last: bool = False
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Renders a single token to a surface with a horizontal offset, i.e. the rendered
        word begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        word vertically on the surface

        Args:
            word (`str`):
                The word to be rendered
            offset (`int`):
                The horizontal starting position of the rendered token on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            is_last (`bool`, *optional*, defaults to False):
                Boolean variable that indicates whether we are rendering the last token of the sequence, in which
                case additional padding is added to the final offset so that the black separator patch is guaranteed
                to be spaced at least this padded amount from the last token

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the offset to where the next patch
            begins, type `int`
        """

        layout = PangoCairo.create_layout(context)
        layout.set_font_description(self.font)

        layout.set_text(word, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout.get_unknown_glyphs_count()} unknown glyphs in word: {word} ({[ord(w) for w in word]}). Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width, height = layout.get_pixel_size()

        position = (offset, self.pixels_per_patch / 2.0 - height / 2.0 - 2)
        context.move_to(*position)

        PangoCairo.show_layout(context, layout)

        if is_last:
            offset += 2
        offset = self._get_offset_to_next_patch(offset + width)

        return context, layout, offset
    
    def _render_bigram_old(
        self, word: str, offset: int, context: cairo.Context, is_last: bool = False
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Renders a single token to a surface with a horizontal offset, i.e. the rendered
        word begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        word vertically on the surface

        Args:
            word (`str`):
                The word to be rendered
            offset (`int`):
                The horizontal starting position of the rendered token on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            is_last (`bool`, *optional*, defaults to False):
                Boolean variable that indicates whether we are rendering the last token of the sequence, in which
                case additional padding is added to the final offset so that the black separator patch is guaranteed
                to be spaced at least this padded amount from the last token

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the offset to where the next patch
            begins, type `int`
        """
        # NOTE : do the same as for _offset_bigram

        layout = PangoCairo.create_layout(context)
        layout.set_font_description(self.font)

        layout.set_text(word, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout.get_unknown_glyphs_count()} unknown glyphs in word: {word} ({[ord(w) for w in word]}). Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width, height = layout.get_pixel_size()

        position = (offset, self.pixels_per_patch / 2.0 - height / 2.0 - 2)
        context.move_to(*position)

        PangoCairo.show_layout(context, layout)

        if is_last:
            offset += 2
        offset = self._get_offset_to_next_patch(offset + width)

        return context, layout, offset
    
    def _offset_single_word(
        self, word: str, offset: int, context: cairo.Context, is_last: bool = False
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Renders a single token to a surface with a horizontal offset, i.e. the rendered
        word begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        word vertically on the surface

        Args:
            word (`str`):
                The word to be rendered
            offset (`int`):
                The horizontal starting position of the rendered token on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            is_last (`bool`, *optional*, defaults to False):
                Boolean variable that indicates whether we are rendering the last token of the sequence, in which
                case additional padding is added to the final offset so that the black separator patch is guaranteed
                to be spaced at least this padded amount from the last token

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the offset to where the next patch
            begins, type `int`
        """

        layout = PangoCairo.create_layout(context)
        layout.set_font_description(self.font)

        layout.set_text(word, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout.get_unknown_glyphs_count()} unknown glyphs in word: {word} ({[ord(w) for w in word]}). Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width, height = layout.get_pixel_size()

        # position = (offset, self.pixels_per_patch / 2.0 - height / 2.0 - 2)
        # context.move_to(*position)

        # PangoCairo.show_layout(context, layout)

        if is_last:
            offset += 2
        offset = self._get_unbounded_offset_to_next_patch(offset + width)

        return offset    
    
    def _render_bigram(
        self, word: str, offset: int, context: cairo.Context, is_last: bool = False
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Renders a single token to a surface with a horizontal offset, i.e. the rendered
        word begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        word vertically on the surface

        Args:
            word (`str`):
                The word to be rendered
            offset (`int`):
                The horizontal starting position of the rendered token on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            is_last (`bool`, *optional*, defaults to False):
                Boolean variable that indicates whether we are rendering the last token of the sequence, in which
                case additional padding is added to the final offset so that the black separator patch is guaranteed
                to be spaced at least this padded amount from the last token

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the offset to where the next patch
            begins, type `int`
        """

        # Measure for first character in bigrams 
        layout1 = PangoCairo.create_layout(context)
        layout1.set_font_description(self.font)

        layout1.set_text(word[0], -1)

        if layout1.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout1.get_unknown_glyphs_count()} unknown glyphs in word: {word} ({[ord(w) for w in word]}). Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width1, height1 = layout1.get_pixel_size()
        
        position1 = (offset, self.pixels_per_patch / 2.0 - height1 / 2.0 - 2)
        context.move_to(*position1)

        PangoCairo.show_layout(context, layout1)
        
        # Last "bigram" of a word can be an unigram; handle these separately 
        if not len(word) > 1: 
            if is_last:
                offset += 2
            offset = self._get_unbounded_offset_to_next_patch(offset + width1) 
            return context, layout1, offset    
        
        # Repeat for second part of bigram
        layout2 = PangoCairo.create_layout(context)
        layout2.set_font_description(self.font)

        layout2.set_text(word[1], -1)

        if layout2.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout2.get_unknown_glyphs_count()} unknown glyphs in word: {word} ({[ord(w) for w in word]}). Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width2, height2 = layout2.get_pixel_size()
        
        position2 = (offset+width1, self.pixels_per_patch / 2.0 - height2 / 2.0 - 2)
        context.move_to(*position2)

        PangoCairo.show_layout(context, layout2)

        if is_last:
            offset += 2
        offset = self._get_unbounded_offset_to_next_patch(offset + width1 + width2) 

        return context, layout2, offset    
    
    def _offset_bigram(
        self, word: str, offset: int, context: cairo.Context, is_last: bool = False
    ) -> Tuple[cairo.Context, Pango.Layout, int]:
        """
        Renders a single token to a surface with a horizontal offset, i.e. the rendered
        word begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        word vertically on the surface

        Args:
            word (`str`):
                The word to be rendered
            offset (`int`):
                The horizontal starting position of the rendered token on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            is_last (`bool`, *optional*, defaults to False):
                Boolean variable that indicates whether we are rendering the last token of the sequence, in which
                case additional padding is added to the final offset so that the black separator patch is guaranteed
                to be spaced at least this padded amount from the last token

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the offset to where the next patch
            begins, type `int`
        """

        # Measure for first character in bigrams 
        layout1 = PangoCairo.create_layout(context)
        layout1.set_font_description(self.font)

        layout1.set_text(word[0], -1)

        if layout1.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout1.get_unknown_glyphs_count()} unknown glyphs in word: {word} ({[ord(w) for w in word]}). Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width1, height1 = layout1.get_pixel_size()
        
        # Last "bigram" of a word can be an unigram; handle these separately 
        if not len(word) > 1: 
            if is_last:
                offset += 2
            offset = self._get_unbounded_offset_to_next_patch(offset + width1) 
            return offset   
        
        # Repeat for second part of bigram
        layout2 = PangoCairo.create_layout(context)
        layout2.set_font_description(self.font)

        layout2.set_text(word[1], -1)

        if layout2.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout2.get_unknown_glyphs_count()} unknown glyphs in word: {word} ({[ord(w) for w in word]}). Consider "
                f"double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width2, height2 = layout2.get_pixel_size()

        if is_last:
            offset += 2
        offset = self._get_unbounded_offset_to_next_patch(offset + width1 + width2) 

        return offset   

    def _render_single_sentence_old(
        self, sentence: str, offset: int, context, max_length: Optional[int] = None, rtl: bool = False
    ) -> Tuple[cairo.Context, Tuple[Pango.Layout, Pango.Layout], int]:
        """
        Renders a single sentence to a surface with a horizontal offset, i.e. the rendered
        sentence begins <offset> pixels to the right from the beginning of the surface, and centers the rendered
        text vertically on the surface

        Args:
            sentence (`str`):
                The sentence to be rendered
            offset (`int`):
                The horizontal starting position of the rendered sentence on the surface (in pixels)
            context (`~cairo.Context`):
                The context object used to render text to the surface
            max_length (`int`, *optional*, defaults to None):
                Maximum number of patches that the rendered sentence may fill on the surface. If set, anything longer
                than this number of patches will be truncated.

        Returns:
            A tuple containing the context of type `~cairo.Context` that we used to draw on the surface,
            the layout of type `~Pango.Layout` containing the rendered sentence, and the width of the rendered
            sentence in pixels, type `int`
        """
        pango_context = PangoCairo.create_context(context)
        pango_context.set_font_description(self.font)
        layout = Pango.Layout(pango_context)

        if rtl:
            layout.set_auto_dir(False)
            pango_context.set_base_dir(Pango.Direction.RTL)
            layout.set_alignment(Pango.Alignment.RIGHT)
        layout.set_text(sentence, -1)

        if layout.get_unknown_glyphs_count() > 0:
            logger.warning(
                f"Found {layout.get_unknown_glyphs_count()} unknown glyphs in sentence: {sentence}. Consider"
                f" double-checking that the correct fonts are loaded."
            )

        # Get logical extents
        width, height = layout.get_pixel_size()
        full_width = width
        full_layout = layout
        truncated_layout = layout

        if max_length is not None:
            if self.px2patch_ceil(offset + width) > max_length:
                truncated_layout = Pango.Layout(pango_context)

                # print(
                #     f"Truncating {sentence} ({self.px2patch_ceil(offset + width)} patches) to fit {max_length = }."
                # )

                # Run binary search to find truncation point
                lo = 0
                hi = len(sentence)
                while lo <= hi:
                    mid = (lo + hi) // 2
                    new_sentence = sentence[:mid]
                    truncated_layout.set_text(new_sentence, -1)
                    width, height = truncated_layout.get_pixel_size()
                    if self.px2patch_ceil(offset + width) < max_length:
                        lo = mid + 1
                    elif self.px2patch_ceil(offset + width) > max_length:
                        hi = mid - 1
                    else:
                        break
                # print(f"New sentence = {new_sentence}, width = {self.px2patch_ceil(offset + width)} patches")

        position = (offset, self.pixels_per_patch / 2.0 - height / 2.0 - 2)
        context.move_to(*position)

        PangoCairo.show_layout(context, truncated_layout)

        return context, (full_layout, truncated_layout), full_width

    # def _render_words_to_surface(
    def _render_single_sentence(
        self, 
        words: List[str], 
        context,
        input_offset: int,
        rtl: bool = False, 
        max_length: Optional[int] = None, 
        **kwargs,
        ) -> Encoding:
        """
        Renders a list of words to a surface and keeps track of
        (a) how many patches in the rendered surface contain text, i.e. are neither blank nor black separator patches
        and (b) the patch index, starting with 0, where each word begins on the rendered surface

        Args:
            words (`List[str]`):
                The list of words to be rendered

        Returns:
            An Encoding of type `Encoding` containing the rendered words and metadata
        """
        def string_to_ngrams(s:str, n:int=2, rtl:bool=False):
            """
            Takes a string and returns a list of character n-grams by splitting `s` on every `n` character.
            Args:
                s (str): The input string to be converted to bigrams.
                n (int): The frequency with which the input string is split. Defaults to `n`=2
            Returns:
                list: A list of character n-grams.
            """
            if rtl:
                # return [s[i:i + n] for i in range(0, len(s), n)]
                return [s[i:i + n][::-1] for i in range(0, len(s), n)][::-1]
            else:
                return [s[i:i + n] for i in range(0, len(s), n)]
        
        # Account for edge case in UD Treebanks v2.10
        if len(words) == 1 and words[0].startswith("http://www.google.com/search"):
            print(f"Detected edge case with {words=}")
            words = ['http://www.google.com/search']
            
        # And create a map between word and patch index incl. meta data 
        word_patch_map = {
            "sum_len_words": sum([len(f"{word} ") for word in words])-len(" "), # add space for every word but last
            "total_number_of_words": len(words),
            "word_order_on_canvas": {},
            "text_is_truncated": False,
            "word_list_truncation_index": len(words)-1, # index of last word after truncation
            "is_rtl": rtl,
            }
        # Counter for index of word in list of words 
        word_counter = 0 
        # Set max length if none is given 
        max_length = self.max_seq_length if max_length is None else max_length
        # Start with an offset equal to the input when measuring the width of every word
        offset = input_offset # NOTE : changed from 0  
        # print(offset)
        skip_last = False
        # Keep track of the patch index at which each new token begins
        word_start_indices = [math.ceil(offset / self.pixels_per_patch)]
        # Keep track of accumulated len of rendered glyphs
        sentence_len_ltr = 0
        # For RTL languages, the left most patch will sum to the entire sequence since it is the last word 
        # sentence_len_rtl = np.sum([len(word.encode('utf-8')) for word in words])
        sentence_len_rtl = word_patch_map['sum_len_words']

        def add_space(s:str, rtl:bool) -> str:
            "Adds space to the left or right of word based on text direction"
            if rtl:
                return f" {s}"
            else:
                return f"{s} "
            
        # First token is measured without whitespace for rtl but with for ltr
        if rtl:
            word_len = len(f"{words[0]}")
        else: 
            word_len = len(f"{words[0]} ")
        for w in string_to_ngrams(words[0], rtl=rtl):
            offset = self._offset_bigram(w, offset, context)
            
        sentence_len_ltr += word_len
        word_patch_map["word_order_on_canvas"][word_counter] = {
            "start_patch_index": word_start_indices[-1], 
            "word": words[0], 
            "word_len": word_len,
            "sentence_len_ltr" : sentence_len_ltr,
            "sentence_len_rtl" : sentence_len_rtl,
            }
        word_counter += 1 
        sentence_len_rtl -= word_len
        word_start_indices.append(math.ceil(offset / self.pixels_per_patch))
        
        # Get width of each token from the start of the start of a patch and 
        # at least a whitespace width apart
        for word in words[1:-1]:
            word_len = len(f"{word} ")
            sentence_len_ltr += word_len 
            word_patch_map["word_order_on_canvas"][word_counter] = {
                "start_patch_index": word_start_indices[-1], 
                "word": word, 
                "word_len": word_len,
                "sentence_len_ltr" : sentence_len_ltr,
                "sentence_len_rtl" : sentence_len_rtl,
                }
            word_counter += 1 
            sentence_len_rtl -= word_len # should include the word, so discount after 
            for w in string_to_ngrams(word, rtl=rtl):
                offset = self._offset_bigram(w, offset, context) 
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

        # Last token is measured without whitespace for ltr but not for rtl
        if rtl: 
            word_len = len(f"{words[-1]} ")
        else:
            word_len = len(f"{words[-1]}")
        for w in string_to_ngrams(words[-1], rtl=rtl):
            offset = self._offset_bigram(w, offset, context)
            
        sentence_len_ltr += word_len
        word_patch_map["word_order_on_canvas"][word_counter] = {
            "start_patch_index": word_start_indices[-1], 
            "word": words[-1], 
            "word_len": word_len,
            "sentence_len_ltr" : sentence_len_ltr,
            "sentence_len_rtl" : sentence_len_rtl,
            }
        word_counter += 1 
        sentence_len_rtl -= word_len

        word_start_indices.append(math.ceil(offset / self.pixels_per_patch))
        
        # Save index of first patch after the last word (where `sep_patch` is) 
        word_patch_map['first_patch_after_last_word'] = word_start_indices[-1]
        
        # Copies in case of truncation
        truncated_word_patch_map = word_patch_map.copy()
        index = word_patch_map['total_number_of_words'] - 1 # Remember: zero-index 
        patches = word_patch_map['first_patch_after_last_word']
        word_order_on_canvas = word_patch_map['word_order_on_canvas']
        
        # Truncate if words don't fit within `max_length`
        if max_length is not None and word_patch_map['first_patch_after_last_word'] > max_length:
            truncated_word_patch_map['text_is_truncated'] = True
            # Run search for cutoff word 
            while patches > max_length:
                index -= 1 
                # check if edge case: first word is longer than `max_length`
                if index < 0:
                    patches = max_length
                else:
                    patches = word_order_on_canvas[index]['start_patch_index']
                
            # Go one more word back to make sure that the entire list of words fit within max_length
            # as very long words might go beyong max_length if rendered as the last 
            index -= 1 
            if index < 0:
                logger.warning(
                    f"First word in {words=} doesn't fit in {max_length=}. Returning canvas with only first word"
                )
                index = 0
        
        # Reset offset equal to `input_offset` for rendering 
        offset = input_offset 
        for word in words[:index]:
            for w in string_to_ngrams(word, rtl=rtl):
                context, layout, offset = self._render_bigram(w, offset, context) 
            
            if offset == self.max_pixels_len - self.pixels_per_patch:
                skip_last = True
                break
            
        # Last token is rendered without whitespace
        if not skip_last: # and index == word_patch_map['total_number_of_words']:
            for w in string_to_ngrams(words[index], rtl=rtl):
                context, layout, offset = self._render_bigram(w, offset, context)
        
        # Update `truncated_word_patch_map` to the truncated layout 
        truncated_word_patch_map["word_order_on_canvas"] = {k:v for k,v in word_patch_map["word_order_on_canvas"].items() 
                                                            if k in list(range(index + 1))}
        truncated_word_patch_map['sum_len_words'] = np.sum([len(f"{word} ") for word in words[:index+1]]) - len(' ')
        truncated_word_patch_map['total_number_of_words'] = len(words[:index+1])
        truncated_word_patch_map["word_list_truncation_index"] = index
        truncated_word_patch_map['first_patch_after_last_word'] = math.ceil(offset / self.pixels_per_patch)
            
        return context, layout, offset, (word_patch_map, truncated_word_patch_map)

    def _render_text_to_surface_exp(
        self, 
        words: List[str], 
        offset: int = 0,
        # context,
        rtl: bool = False, 
        max_length: Optional[int] = None, 
        **kwargs,
        ) -> Encoding:
        """
        Renders a list of words to a surface and keeps track of
        (a) how many patches in the rendered surface contain text, i.e. are neither blank nor black separator patches
        and (b) the patch index, starting with 0, where each word begins on the rendered surface

        Args:
            words (`List[str]`):
                The list of words to be rendered

        Returns:
            An Encoding of type `Encoding` containing the rendered words and metadata
        """
        if not rtl:
            # Check directions of words 
            rtl = self.is_rtl(' '.join(words))
        if rtl: 
             # reverse the order of words to ensure correct word order on canvas
            words = words[::-1]
            
        # Keep track of the patch index at which each new token begins
        word_start_indices = [0]
        # And create a map between word and patch index incl. meta data 
        word_patch_map = {
            "sum_len_words": np.sum([len(word) for word in words]),
            "total_number_of_words": len(words),
            "word_order_on_canvas": {},
            "text_is_truncated": False,
            "word_list_truncation_index": len(words)+1
            }
        # counter for word index in list of words 
        word_counter = 0 
        
        # Start with blank surface
        surface, context, sep_patches = self.get_empty_surface()

        # Pad left with 2px
        # offset = 2
        offset = 0 # NOTE changed 
        skip_last = False
        
        def add_space(s:str, rtl:bool) -> str:
            " Adds space either based on text direction"
            if rtl:
                return f" {s}"
            else:
                return f"{s} "
            
        # Render each token to the start of the next patch but at least a whitespace width apart
        for word in words[:-1]:
            word_patch_map["word_order_on_canvas"][word_counter] = {
                "start_patch_index": word_start_indices[-1], "word": word, "word_len": len(word)}
            word_counter += 1 
            context, layout, offset = self._render_single_word(add_space(word, rtl), offset, context) 
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))

            if offset == self.max_pixels_len - self.pixels_per_patch:
                skip_last = True
                break

        # Last token is rendered without whitespace
        if not skip_last:
            word_patch_map["word_order_on_canvas"][word_counter] = {
                "start_patch_index": word_start_indices[-1], "word": words[-1], "word_len": len(word)}
            word_counter += 1 
            context, layout, offset = self._render_single_word(words[-1], offset, context, is_last=True)
            word_start_indices.append(math.ceil(offset / self.pixels_per_patch))
        
        # Save index of first patch after the last word (where `sep_patch` is) 
        word_patch_map['first_patch_after_last_word'] = word_start_indices[-1]
        
        # Truncate if words don't fit within `max_length`
        if max_length is not None and word_patch_map['first_patch_after_last_word'] > max_length:
            # Run search for cutoff word 
            index = word_patch_map['total_number_of_words']
            patches = word_patch_map['first_patch_after_last_word']
            word_order_on_canvas = word_patch_map['word_order_on_canvas']
            while patches > max_length:
                index -= 1 
                patches = word_order_on_canvas[index]['start_patch_index']
                
            # Go one more word back to make sure that the entire list of words fit within max_length
            # as very long words might go beyong max_length if rendered as the last 
            index -= 1 
            if index < 0:
                logger.warning(
                    f"First word doesn't fit in {max_length=}. Returning canvas with only first word"
                )
                index = 0
        
            # New blank surface 
            surface, context, sep_patches = self.get_empty_surface()
            offset = 0 
            truncated_indices = word_patch_map['total_number_of_words']-index
            truncated_indices_list = list(range(index + 1))
            for word in words[:index+1]:
                context, truncated_layout, offset = self._render_single_word(add_space(word, rtl), offset, context) 
            
            # Adjust `word_patch_map` to the truncated layout 
            word_patch_map["word_order_on_canvas"] = {k:v for k,v in word_patch_map["word_order_on_canvas"].items() 
                                                      if k in truncated_indices_list}
            word_patch_map['text_is_truncated'] = True
            word_patch_map["word_list_truncation_index"] = index
            word_start_indices = word_start_indices[:index+1]
            word_patch_map['first_patch_after_last_word'] = math.ceil(offset / self.pixels_per_patch)
            
        else: 
            truncated_layout = layout
            
        # return context, (layout, truncated_layout), offset, word_patch_map
            # NOTE : do I need to adjust sum_len_words and `total_number_of_words` as well for the truncated layout?

        # Draw black rectangle on surface as separator patch
        sep_patches.append(offset)
        # print(math.ceil(offset / self.pixels_per_patch))

        num_text_patches = math.ceil(offset / self.pixels_per_patch)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface, sep_patches=sep_patches),
            sep_patches=sep_patches,
            num_text_patches=num_text_patches,
            word_starts=word_start_indices,
            word_patch_map=word_patch_map,
        )
        return encoding
        
    def find_word_index_from_sentence_len(self, dictionary:dict, sentence_len:int, rtl:bool) -> int:
        """
        Relies on the structure of `word_patch_map` 
        Takes a sentence len and finds the corresponding word index.
        """
        if rtl:
            sentence_key = 'sentence_len_rtl'
        else:
            sentence_key = 'sentence_len_ltr'
        return min(dictionary['word_order_on_canvas'], \
                    key=lambda i: math.floor(abs(sentence_len - dictionary['word_order_on_canvas'][i][sentence_key]))) 

    def _render_text_pair_to_surface_ltr(
        self,
        text_pair: Tuple[str, str],
        return_overflowing_patches: bool = False,
        return_offset_mapping: bool = False,
        stride: int = 0,
        text_a_max_length: Optional[int] = None,
        rtl : bool = False,
        **kwargs,
    ) -> Encoding:
        """
        Renders a text pair left-to-right (LTR).

        Args:
            text_pair (`Tuple[str, str]`):
                The text pair to be rendered
            return_overflowing_patches (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing patch sequences.
            return_offset_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each patch.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing patches returned when
                `return_overflowing_patches=True` will contain some patches from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping patches.
            text_a_max_length (`int`, *optional*):
                Maximum length (in image patches) of the first text in the text pair.
            rtl (`bool`, *optional*, defaults to `False`):
                Whether text is written in right-to-left (RTL) script. Note: If set to False, the renderer will detect
                the text direction automatically, so the text can still be rendered RTL, depending on its content

        Returns:
            An Encoding of type `Encoding` containing the rendered text pair and metadata
        """
        
        def find_closest_start_of_word_patch(dictionary:dict, x:int) -> int:
            """
            Relies on the structure of `word_patch_map` 
            Takes a patch offset and finds the nearest start-of-word patch that we know can be rendered.
            Returns the word index 
            """
            return min(dictionary['word_order_on_canvas'], \
                       key=lambda i: math.floor(abs(x - dictionary['word_order_on_canvas'][i]['start_patch_index']))) - 1 
            
        def slice_word_order_on_canvas(dictionary:dict, start:int, end:int) -> dict:
            """
            Slices `dictionary` based on word indices
            Return including `start` and `end`.
            """
            data = dictionary.copy()
            word_order_slice = dict(filter(lambda x: start <= x[0] <= end, data['word_order_on_canvas'].items()))
            data['word_order_on_canvas'] = word_order_slice
            try:
                data['word_list_truncation_index'] = list(data['word_order_on_canvas'].keys())[-1]
            except IndexError:
                print(start, end, dictionary)
                raise IndexError()
            data['total_number_of_words'] = len(list(data['word_order_on_canvas'].keys()))
            return data
        
        def extract_words(dictionary: dict) -> list:
            """
            Returns list of words under the key `word_order_on_canvas` in `dictionary`
            """
            return [word_dict['word'] for word_dict in dictionary['word_order_on_canvas'].values()]

        text_a, text_b = text_pair

        surface, context, sep_patches = self.get_empty_surface()
        sequence_ids = [None]

        offset = 0
        # print(text_a_max_length)

        # Render first sentence and draw on surface
        # context, (full_layout, truncated_layout), text_a_width = self._render_single_sentence(
        context, layout, text_a_width, (word_patch_map_a, truncated_word_patch_map_a) = self._render_single_sentence(
            text_a, input_offset=offset, context=context, max_length=text_a_max_length, rtl=rtl,
        )
        if return_offset_mapping: # Should no longer be necesarry given `word_patch_map`
            pass

        offset = self._get_offset_to_next_patch(text_a_width) # discard additional 2px padding 
        # offset = self._get_offset_to_next_patch(offset + text_a_width + 2)
        
        # Mark patch starting at offset as black separator patch
        sep_patches.append(offset)

        # # Add a 0 to sequence_ids for each patch in text_a and None for the separator patch
        sequence_ids.extend([0] * self.px2patch_floor(offset) + [None])

        # Reserve space for the black separator patch + 2 (padding)
        offset = offset + self.pixels_per_patch 
        # offset = offset + self.pixels_per_patch + 2

        # Render second sentence and draw on surface
        context, layout, text_b_width, (word_patch_map_b, truncated_word_patch_map_b) = self._render_single_sentence(
            text_b, 
            input_offset=offset, 
            context=context,
            rtl=rtl,
            )
        
        if return_offset_mapping:
            pass
        else:
            offset_mapping = None

        eos_patch_offset = self._get_offset_to_next_patch(text_b_width) 
        # eos_patch_offset = self._get_offset_to_next_patch(offset + text_b_width)

        # Mark patch starting at offset as black separator patch
        sep_patches.append(eos_patch_offset)

        # Add a 1 to sequence_ids for each patch in text_b and None for the separator patch
        b_patches = self.px2patch_floor(eos_patch_offset - offset)
        sequence_ids.extend([1] * b_patches + [None])

        image = self.get_image_from_surface(surface, sep_patches=sep_patches)
        
        num_text_patches = self.px2patch_floor(eos_patch_offset)

        encoding = Encoding(
            pixel_values=image,
            sep_patches=[self.px2patch_ceil(s) for s in sep_patches],
            num_text_patches=num_text_patches,
            offset_mapping=offset_mapping,
            word_patch_map={
                "text_a": truncated_word_patch_map_a, 
                "text_b": truncated_word_patch_map_b,
                "full_text_b": word_patch_map_b,
                },
            overflowing_patches=None,
        )

        if return_overflowing_patches:

            pixel_overflow = (truncated_word_patch_map_a['first_patch_after_last_word'] + \
                word_patch_map_b['first_patch_after_last_word'])*self.pixels_per_patch \
                - (self.max_pixels_len - self.pixels_per_patch) 
            patch_overflow = self.px2patch_ceil(pixel_overflow)
            
            if pixel_overflow > 0:

                # Determine how many additional sequences we need to generate               
                new_text_b_patches_per_sequence = self.max_seq_length - \
                    (truncated_word_patch_map_a['first_patch_after_last_word'] + 4*1) # 4*1 for added polstering 

                # Store start and end for word index in additional sequences 
                list_offset_start_end_word_index = []
                word_index_end_last_seq = truncated_word_patch_map_b['word_list_truncation_index']
                word_index_start_last_seq = -1 # Only used for words that on its own are longer than the entire seq max length
                
                while word_index_end_last_seq < word_patch_map_b['word_list_truncation_index']: # current last word index < global last word index  
                    # Find closest word start index to a given patch index 
                    word_index_start_new_seq = find_closest_start_of_word_patch(
                        word_patch_map_b,
                        word_patch_map_b['word_order_on_canvas'][word_index_end_last_seq]['start_patch_index'] - stride, 
                    )
                    
                    # Some "words" are very long. Ensure progress by pushing start index forward
                    # There could be "words" that are longer than the allowed length for a new sequence
                    # in that case, just skip the word 
                    if word_index_start_new_seq <= word_index_start_last_seq:
                        word_index_start_new_seq = word_index_start_last_seq + 1
                        # Check if it's the last word
                        if word_index_start_new_seq == word_patch_map_b['word_list_truncation_index'] - 1:
                            this_seq_length = word_patch_map_b['word_order_on_canvas'][word_index_end_new_seq + 1]['start_patch_index'] \
                                - word_patch_map_b['word_order_on_canvas'][word_index_start_new_seq]['start_patch_index'] \
                                + truncated_word_patch_map_a['first_patch_after_last_word'] + 2 
                            if this_seq_length + word_patch_map_b['first_patch_after_last_word'] \
                                 - word_patch_map_b['word_order_on_canvas'][word_index_start_new_seq]['start_patch_index'] \
                                     < self.max_seq_length - 1:
                                # Last word fits in last sequence; nothing is lost 
                                word_index_end_new_seq = word_patch_map_b['word_list_truncation_index']
                                list_offset_start_end_word_index.append((word_index_start_new_seq, word_index_end_new_seq))
                                break
                            else:
                                # Last word does not fit on canvas. Nothing we can do.
                                print("[!] Last word does not fit on canvas. That word is lost.")
                                break 
                             
                    word_index_end_new_seq = find_closest_start_of_word_patch(
                        word_patch_map_b,
                        word_patch_map_b['word_order_on_canvas'][word_index_start_new_seq]['start_patch_index'] + new_text_b_patches_per_sequence,
                        )
                            
                    # Add + 1 to `word_index_end_new_seq` to also include the width of the last word 
                    this_seq_length = word_patch_map_b['word_order_on_canvas'][word_index_end_new_seq + 1]['start_patch_index'] \
                        - word_patch_map_b['word_order_on_canvas'][word_index_start_new_seq]['start_patch_index'] \
                        + truncated_word_patch_map_a['first_patch_after_last_word'] + 2 
                    # print(f"{this_seq_length=}")
                    
                    # Check that new sequence will fit 
                    # Recall: some "words" are very long, so nothing is granted
                    while this_seq_length > self.max_seq_length - 1:
                        # NOTE : include check here that if temp_stride is 0 >>> probably not necessary anyway
                        # we just need to skip 
                        # print("! `this_seq_length` > `self.max_seq_length`, going back 1 index")
                        # print(f"BEFORE {word_index_end_new_seq=}")
                        word_index_end_new_seq = min(word_index_end_new_seq - 1, word_index_end_last_seq + 1)  
                        # print(f"AFTER {word_index_end_new_seq=}")
                        # word_index_end_new_seq = min(word_index_end_new_seq - 1, word_index_end_last_seq + 1)  
                        this_seq_length = word_patch_map_b['word_order_on_canvas'][word_index_end_new_seq + 1]['start_patch_index'] \
                            - word_patch_map_b['word_order_on_canvas'][word_index_start_new_seq]['start_patch_index'] \
                            + truncated_word_patch_map_a['first_patch_after_last_word'] + 2                           
                        
                    # For the last word (recall: `find_closest_start_of_word_patch()` returns index of last word we know we can fit in)
                    if word_index_end_new_seq == word_patch_map_b['word_list_truncation_index'] - 1:
                        # Check if we can fit in the last word in this sequence 
                        if this_seq_length + word_patch_map_b['first_patch_after_last_word'] \
                            - word_patch_map_b['word_order_on_canvas'][word_index_end_new_seq]['start_patch_index'] > self.max_seq_length - 1:
                                # If not, go back one index and create another round of overflow 
                                word_index_end_new_seq -= 1
                        else: 
                            word_index_end_new_seq = word_patch_map_b['word_list_truncation_index']
                            list_offset_start_end_word_index.append((word_index_start_new_seq, word_index_end_new_seq))
                            break
                        
                    word_index_end_last_seq = word_index_end_new_seq
                    word_index_start_last_seq = word_index_start_new_seq
                    # Edge case: if a word is just too long to fit on the canvas,
                    # the model can get stuck and we just have to skip the word
                    if word_index_start_new_seq <= word_index_end_new_seq:
                        list_offset_start_end_word_index.append((word_index_start_new_seq, word_index_end_new_seq))
                
                if not np.all([a<=b for (a,b) in list_offset_start_end_word_index]):
                    print(list_offset_start_end_word_index)
                    raise ValueError("Some start indeces are larger than end!! ")
                overflow_word_patch_maps = [slice_word_order_on_canvas(dictionary=word_patch_map_b, 
                                                                       start=start,
                                                                       end=end) 
                                            for (start, end) in list_offset_start_end_word_index
                                            ]

                overflow_encodings = []
                
                for i in overflow_word_patch_maps:

                    # Start a new surface for the overflow sequence
                    o_surface, o_context, o_sep_patches = self.get_empty_surface()

                    text_remainder = extract_words(i)

                    continuation_starting_point = (
                        self._get_offset_to_next_patch(text_a_width + self.pixels_per_patch)
                        # self._get_offset_to_next_patch(2 + text_a_width + 2) + self.pixels_per_patch + 2
                    )

                    # Render only the continuation (i.e., the part that is new in this overflow sequence and the stride)
                    # onto the surface for now. The text_a content gets copied over later
                    o_context, o_layout, o_text_b_width, (o_word_patch_map_b, o_truncated_word_patch_map_b) = self._render_single_sentence(
                        text_remainder, 
                        input_offset=continuation_starting_point, 
                        context=o_context,
                        rtl=rtl,
                        )

                    # Remember where to put SEP patch
                    o_eos_offset = self._get_offset_to_next_patch(o_text_b_width)
                    o_sep_patches.append(o_eos_offset)

                    num_text_patches = self.px2patch_floor(o_eos_offset)

                    # Take original image or previous overflow sequence image to copy data from
                    previous_image = image
                    image = self.get_image_from_surface(o_surface, sep_patches=[sep_patches[0]] + o_sep_patches)
                    

                    # Copy [text_a, sep patch, padding] content from previous image
                    image[:, : (truncated_word_patch_map_a['first_patch_after_last_word'] + 1)*self.pixels_per_patch] = previous_image[
                        :, : (truncated_word_patch_map_a['first_patch_after_last_word'] + 1)*self.pixels_per_patch
                    ]

                    overflow_encodings.append(
                        Encoding(
                            pixel_values=image,
                            sep_patches=sep_patches + o_sep_patches,
                            num_text_patches=num_text_patches,
                            word_patch_map={
                                "text_a": truncated_word_patch_map_a, 
                                "text_b": o_truncated_word_patch_map_b,
                                "full_text_b": word_patch_map_b,
                                'sliced_word_patch_map_b': i,
                                },
                        )
                    )
                encoding.overflowing_patches = overflow_encodings

        return encoding

    def _render_text_pair_to_surface_rtl(
        self,
        text_pair: Tuple[str, str],
        return_overflowing_patches: bool = False,
        return_offset_mapping: bool = False,
        stride: int = 0,
        text_a_max_length: Optional[int] = None,
        **kwargs,
    ) -> Encoding:
        """
        Renders a text pair right-to-left (RTL).

        Args:
            text_pair (`Tuple[str, str]`):
                The text pair to be rendered
            return_overflowing_patches (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing patch sequences.
            return_offset_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each patch.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing patches returned when
                `return_overflowing_patches=True` will contain some patches from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping patches.
            text_a_max_length (`int`, *optional*):
                Maximum length (in image patches) of the first text in the text pair.
            rtl (`bool`, *optional*, defaults to `False`):
                Whether text is written in right-to-left (RTL) script. Note: If set to False, the renderer will detect
                the text direction automatically, so the text can still be rendered RTL, depending on its content

        Returns:
            An Encoding of type `Encoding` containing the rendered text pair and metadata
        """

        text_a, text_b = text_pair

        surface, context, sep_patches = self.get_empty_surface()
        sequence_ids = [None]

        offset = 0

        # Render first sentence and draw on surface
        context, (full_layout, truncated_layout), text_a_width = self._render_single_sentence(
            text_a, offset=offset, context=context, max_length=text_a_max_length, rtl=True
        )

        if return_offset_mapping:
            # cluster_position_dict = self.get_cluster_position_dict(layout, offset, text_a_width + 2)
            text_a_offset_mapping = self.get_text_offset_mapping(
                layout=truncated_layout, offset=offset, text_width=(text_a_width + 2), text_shift=0, rtl=True
            )
        # Offset is left padding + rendered width of text_a + 2 (padding)
        offset = self._get_offset_to_next_patch(offset + text_a_width + 2)

        # Mark patch starting at offset as black separator patch
        sep_patches.append(offset)

        # # Add a 0 to sequence_ids for each patch in text_a and None for the separator patch
        sequence_ids.extend([0] * self.px2patch_floor(offset) + [None])

        # Reserve space for the black separator patch + 2 (padding)
        offset = offset + self.pixels_per_patch + 2

        # Render second sentence and draw on surface
        context, (full_layout, truncated_layout), text_b_width = self._render_single_sentence(
            text_b,
            offset=offset,
            context=context,
            max_length=self.px2patch_floor(self.max_pixels_len - self.pixels_per_patch),
            rtl=True,
        )

        if return_offset_mapping:
            # Get offset mapping both for the full and the truncated layout. The truncated one is used in any case and
            # full one is used in case there is overflow
            full_text_b_offset_mapping = self.get_text_offset_mapping(
                layout=full_layout, offset=offset, text_width=(text_b_width + 2), text_shift=offset - 2, rtl=True
            )
            truncated_text_b_offset_mapping = self.get_text_offset_mapping(
                layout=truncated_layout, offset=offset, text_width=(text_b_width + 2), text_shift=offset - 2, rtl=True
            )

            offset_mapping = text_a_offset_mapping + [(0, 0)] + truncated_text_b_offset_mapping
            offset_mapping = self.pad_or_truncate_offset_mapping(offset_mapping)
        else:
            offset_mapping = None

        # Offset is left padding + rendered width of text_b + 2 (padding)
        eos_patch_offset = self._get_offset_to_next_patch(offset + text_b_width + 2)

        # Mark patch starting at offset as black separator patch
        sep_patches.append(eos_patch_offset)

        # # Add a 1 to sequence_ids for each patch in text_b and None for the separator patch
        b_patches = self.px2patch_floor(eos_patch_offset - offset)
        sequence_ids.extend([1] * b_patches + [None])

        image = self.get_image_from_surface(surface, sep_patches=sep_patches)

        num_text_patches = self.px2patch_floor(eos_patch_offset)

        encoding = Encoding(
            pixel_values=image,
            sep_patches=sep_patches,
            num_text_patches=num_text_patches,
            offset_mapping=offset_mapping,
            overflowing_patches=None,
        )

        # Calculate how many patches / pixels of the overflow sequence are already filled by
        # text_a, the sep patch, padding, and the stride
        num_patches_filled = self.px2patch_floor(self._get_offset_to_next_patch(2 + text_a_width + 2)) + 1 + stride
        num_pixels_filled = self.patch2px(num_patches_filled)

        if return_overflowing_patches:

            if not return_offset_mapping:
                raise ValueError(
                    "The argument return_overflowing_patches=True requires that return_offset_mapping"
                    " is also set to True"
                )
            offset_mapping = full_text_b_offset_mapping

            pixel_overflow = (offset + text_b_width) - (self.max_pixels_len - self.pixels_per_patch)
            patch_overflow = self.px2patch_ceil(pixel_overflow)

            if pixel_overflow > 0:
                # Determine how many additional sequences we need to generate
                max_num_additional_sequences = math.ceil(
                    pixel_overflow
                    / (
                        self.max_pixels_len
                        - self.pixels_per_patch
                        - (
                            self._get_offset_to_next_patch(2 + text_a_width + 2)
                            + self.pixels_per_patch
                            + stride * self.pixels_per_patch
                        )
                    )
                )

                overflow_encodings = []
                for i in range(max_num_additional_sequences):

                    # By shifting the continuation in each overflowing sequence to the left by some small amount
                    # it can happen that there is actually less overflow than initially calculated, potentially even
                    # requiring fewer additional sequences.
                    if pixel_overflow <= 0:
                        break

                    # Start a new surface for the overflow sequence
                    o_surface, o_context, o_sep_patches = self.get_empty_surface()

                    text_remainder = text_b[offset_mapping[patch_overflow + stride][1] :]

                    continuation_starting_point = (
                        self._get_offset_to_next_patch(2 + text_a_width + 2) + self.pixels_per_patch + 2
                    )

                    # Render only the continuation (i.e., the part that is new in this overflow sequence and the stride)
                    # onto the surface for now. The text_a content gets copied over later
                    o_context, (o_full_layout, o_truncated_layout), o_text_width = self._render_single_sentence(
                        text_remainder,
                        continuation_starting_point,
                        o_context,
                        max_length=self.px2patch_floor(self.max_pixels_len - self.pixels_per_patch),
                        rtl=True,
                    )

                    # Remember where to put SEP patch
                    o_eos_offset = self._get_offset_to_next_patch(continuation_starting_point + o_text_width + 2)
                    o_sep_patches.append(o_eos_offset)

                    # Determine the real (i.e., excluding additional overflow) rendered width of the continuation
                    # to find its starting and end points in the offset_mapping
                    rendered_width_real = min(
                        2 + o_text_width,
                        self.max_pixels_len - self.pixels_per_patch - continuation_starting_point,
                    )

                    continuation_end_letter = patch_overflow + stride
                    continuation_start_letter = max(
                        0, continuation_end_letter - self.px2patch_floor(rendered_width_real)
                    )
                    o_offset_mapping = offset_mapping[continuation_start_letter : continuation_end_letter + 1]

                    # Re-calculate overflow
                    patch_overflow = continuation_start_letter
                    pixel_overflow = self.patch2px(patch_overflow)

                    num_text_patches = self.px2patch_floor(o_eos_offset)

                    # Take original image or previous overflow sequence image to copy data from
                    previous_image = image
                    image = self.get_image_from_surface(o_surface, sep_patches=[sep_patches[0]] + o_sep_patches)

                    # Copy [text_a, sep patch, padding] content from previous image
                    image[:, : num_pixels_filled - self.patch2px(stride)] = previous_image[
                        :, : num_pixels_filled - self.patch2px(stride)
                    ]

                    o_offset_mapping = text_a_offset_mapping + [(0, 0)] + o_offset_mapping
                    o_offset_mapping = self.pad_or_truncate_offset_mapping(o_offset_mapping)

                    overflow_encodings.append(
                        Encoding(
                            pixel_values=image,
                            sep_patches=sep_patches + o_sep_patches,
                            num_text_patches=num_text_patches,
                            offset_mapping=o_offset_mapping,
                        )
                    )
                encoding.overflowing_patches = overflow_encodings

        return encoding

    def _render_text_pair_to_surface(
        self,
        text_pair: Tuple[str, str],
        return_overflowing_patches: bool = False,
        return_offset_mapping: bool = False,
        stride: int = 0,
        text_a_max_length: Optional[int] = None,
        rtl: bool = False,
        **kwargs,
    ) -> Encoding:
        """
        Renders a pair of sentences or paragraphs to a surface and keeps track of
        how many patches in the rendered surface contain text, i.e. are neither blank nor black separator patches

        Args:
            text_pair (`Tuple[str, str]`):
                The text pair to be rendered
            return_overflowing_patches (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing patch sequences.
            return_offset_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each patch.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_length`, the overflowing patches returned when
                `return_overflowing_patches=True` will contain some patches from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping patches.
            text_a_max_length (`int`, *optional*):
                Maximum length (in image patches) of the first text in the text pair.
            rtl (`bool`, *optional*, defaults to `False`):
                Whether text is written in right-to-left (RTL) script. Note: If set to False, the renderer will detect
                the text direction automatically, so the text can still be rendered RTL, depending on its content

        Returns:
            An Encoding of type `Encoding` containing the rendered text pair and metadata
        """

        text_a, text_b = text_pair

        # Clean texts
        # text_a = text_a.replace("\n", " ")
        # text_b = text_b.replace("\n", " ")

        # Check whether text is written in a right-to-left script
        if rtl or (self.is_rtl(' '.join(text_a)) and self.is_rtl(' '.join(text_b))):
            # rendering_fn = self._render_text_pair_to_surface_rtl
            rtl = True
            # rtl = False # NOTE : changed for experiment on positional embeddings 
        # else: 
        rendering_fn = self._render_text_pair_to_surface_ltr

        if rtl: 
            # pass
            # reverse the order of words to ensure correct word order on canvas
            # NOTE : changed for experiment on positional embeddings 
            text_a = text_a[::-1]
            text_b = text_b[::-1]

        return rendering_fn(
            text_pair=(text_a, text_b),
            return_overflowing_patches=return_overflowing_patches,
            return_offset_mapping=return_offset_mapping,
            stride=stride,
            text_a_max_length=text_a_max_length,
            rtl=rtl,
            **kwargs,
        )

    def _render_text_to_surface(
        self,
        text: list,
        rtl: bool = False,
        max_length: Union[None, int] = None,
        **kwargs,
    ) -> Encoding:
        """
        Renders a single piece of text, e.g. a sentence or paragraph, to a surface and keeps track of
        metadata, e.g. how many patches in the rendered surface contain text, i.e. are neither blank nor black separator
        patches
        Args:
            text (`str`):
                The piece of text to be rendered


        Returns:
            An Encoding of type `Encoding` containing the rendered text and metadata
        """
        
        if not rtl:
            # Check directions of words 
            rtl = self.is_rtl(' '.join(text))
        if rtl: 
             # reverse the order of words to ensure correct word order on canvas
            text = text[::-1]

        # Clean text
        # text = text.replace("\n", " ")

        surface, context, sep_patches = self.get_empty_surface()

        offset = 0
        # offset = 2

        # Render text
        # max_length = self.max_seq_length if max_length is None else max_length
        # print(f"{max_length}")
        context, layout, text_width, (word_patch_map, truncated_word_patch_map) = self._render_single_sentence(text, 
                                                                                                               context=context, 
                                                                                                               input_offset=offset, 
                                                                                                               max_length=max_length,
                                                                                                               rtl=rtl)
        # surface, context, sep_patches = self.get_empty_surface()
        # context, layout, text_width, (word_patch_map, truncate_word_patch_map) = self._render_single_sentence2(text[:word_patch_map["word_list_truncation_index"]], 
        #                                                                                 offset, max_length=max_length)
        
        
        # Offset is left padding + rendered width of first sentence + 2 (padding)
        # eos_patch_offset = self._get_offset_to_next_patch(text_width+2)
        eos_patch_offset = self._get_offset_to_next_patch(text_width)
        # print(eos_patch_offset)
        sep_patches.append(eos_patch_offset)

        num_text_patches = self.px2patch_floor(eos_patch_offset)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface, sep_patches=sep_patches),
            sep_patches=[self.px2patch_ceil(s) for s in sep_patches],
            num_text_patches=num_text_patches,
            word_patch_map=truncated_word_patch_map,
            word_starts=[v['start_patch_index'] for k,v in truncated_word_patch_map['word_order_on_canvas'].items()] + [truncated_word_patch_map['first_patch_after_last_word']]
        )

        return encoding
    
    def _render_as_list_text_to_surface(
        self,
        text: list,
        rtl: bool = False,
        max_length: Union[None, int] = None,
        **kwargs,
    ) -> Encoding:
        """
        Renders a single piece of text, e.g. a sentence or paragraph, to a surface and keeps track of
        metadata, e.g. how many patches in the rendered surface contain text, i.e. are neither blank nor black separator
        patches
        Args:
            text (`str`):
                The piece of text to be rendered


        Returns:
            An Encoding of type `Encoding` containing the rendered text and metadata
        """
        
        if not rtl:
            # Check directions of words 
            rtl = self.is_rtl(' '.join(text))
        # if rtl: 
        #      # reverse the order of words to ensure correct word order on canvas
        #     text = text[::-1]

        # Clean text
        # text = text.replace("\n", " ")
        # print(text)
        # print(text[::-1])
        

        surface, context, sep_patches = self.get_empty_surface()

        offset = 0
        # offset = 2

        # Render text
        # max_length = self.max_seq_length if max_length is None else max_length
        # print(f"{max_length}")
        context, layout, text_width, (word_patch_map, truncated_word_patch_map) = self._render_single_sentence(text, 
                                                                                                               context=context, 
                                                                                                               input_offset=offset, 
                                                                                                               max_length=max_length,
                                                                                                               rtl=rtl)
        # surface, context, sep_patches = self.get_empty_surface()
        # context, layout, text_width, (word_patch_map, truncate_word_patch_map) = self._render_single_sentence2(text[:word_patch_map["word_list_truncation_index"]], 
        #                                                                                 offset, max_length=max_length)
        
        
        # Offset is left padding + rendered width of first sentence + 2 (padding)
        eos_patch_offset = self._get_offset_to_next_patch(text_width)
        sep_patches.append(eos_patch_offset)

        num_text_patches = self.px2patch_floor(eos_patch_offset)

        encoding = Encoding(
            pixel_values=self.get_image_from_surface(surface, sep_patches=sep_patches),
            sep_patches=[self.px2patch_ceil(s) for s in sep_patches],
            num_text_patches=num_text_patches,
            word_patch_map=truncated_word_patch_map,
            word_starts=[v['start_patch_index'] for k,v in truncated_word_patch_map['word_order_on_canvas'].items()] + [truncated_word_patch_map['first_patch_after_last_word']]
        )

        return encoding

    def get_image_from_surface(self, surface: cairo.ImageSurface, sep_patches: List[int]) -> np.ndarray:
        """
        Transforms a surface containing a rendered image into a numpy image and inserts black separator patches.

        Args:
            surface (`cairo.ImageSurface`):
                The cairo surface containing the rendered text
            sep_patches (`List[int]`):
                A list of offset values at which black separator patches will be inserted
        Returns:
            An image of type `np.ndarray` of size [self.pixels_per_patch, self.max_pixels_len]
        """

        # Get image data from surface
        data = surface.get_data()
        if self.rgb:
            data = np.frombuffer(data, dtype=np.uint8).reshape((self.pixels_per_patch, self.max_pixels_len, 4))
            data = data[:, :, :3]
            # Reverse channels BGR -> RGB
            image = data[:, :, ::-1]
            # Insert black separator patches
            for idx, sep_patch in enumerate(sep_patches):
                image[:, sep_patch : sep_patch + self.pixels_per_patch, :] = 0
        else:
            data = np.frombuffer(data, dtype=np.uint8).reshape((self.pixels_per_patch, self.max_pixels_len))
            image = np.invert(data)
            # Insert black separator patches
            for idx, sep_patch in enumerate(sep_patches):
                image[:, sep_patch : sep_patch + self.pixels_per_patch] = 0

        return image

    def preprocess_text(self, text: str) -> list:
        # TODO make this instance-specific like Phillip had ...
        # NOTE since we only care about QA right now, just do the stripping and report if something is removed... 
        # ... we can deal with the rest later 
        # text = text.replace("\n", " ")
        # text = unicodedata.normalize("NFKD", text)
        leading_whitespace = len(text) - len(text.lstrip(' '))
        trailing_whitespace = len(text) - len(text.rstrip(' '))
        if leading_whitespace != 0:
            print('Leading whitespace removed')
        if trailing_whitespace != 0:
            print('Trailing whitespace removed')
        text = text.strip()
        # NOTE : also return how many whitespaces are stripped for QA 
        # and probably don't do unicodedata.normalize() ... 
        # perhaps return len_as_bytes for raw string and make sure that it aligns with the preprocessed one 
        splitted_text = text.split(' ')
        # return  text.split(' ')
        text_stripped = []
        for t in splitted_text:
            if t:
                tt = t.split(',')
                if t[-1]==',':
                    tt[-2] += ','
                    tt = tt[:-1]
                text_stripped.append(tt)
            else:
                text_stripped.append(t)
            #     print(t, splitted_text, text)
            #     raise ValueError()
        out = [item for sublist in text_stripped for item in sublist]
        return out

    def preprocess_text_ws_only(self, text: str) -> list:
        # leading_whitespace = len(text) - len(text.lstrip(' '))
        # trailing_whitespace = len(text) - len(text.rstrip(' '))
        # if leading_whitespace != 0:
        #     print('Leading whitespace removed')
        # if trailing_whitespace != 0:
        #     print('Trailing whitespace removed')
        text = text.strip()
        return text.split()
    
    def __call__(
        self,
        text: Union[str, Tuple[str, str], List[str]],
        return_overflowing_patches: bool = False,
        return_offset_mapping: bool = False,
        stride: int = 0,
        rtl: bool = False,
        **kwargs,
    ) -> Encoding:
        """
        Render a piece of text to a surface, convert the surface into an image and return the image
        along with metadata (the number of patches containing text and, when rendering a list of words, the patch
        indices at which each word starts)

        Args:
            text (`str` or `Tuple[str, str]` or `List[str]`):
                The text to be rendered
            return_overflowing_patches (`bool`, *optional*, defaults to `False`):
                Whether or not to return overflowing patch sequences.
            return_offset_mapping (`bool`, *optional*, defaults to `False`):
                Whether or not to return `(char_start, char_end)` for each patch.
            stride (`int`, *optional*, defaults to 0):
                If set to a number along with `max_seq_length`, the overflowing patches returned when
                `return_overflowing_patches=True` will contain some patches from the end of the truncated sequence
                returned to provide some overlap between truncated and overflowing sequences. The value of this
                argument defines the number of overlapping patches.

        Returns:
            An encoding of type `Encoding` containing the rendered image and metadata
        """
        
        if isinstance(text, list):
            # rendering_fn = self._render_single_sentence
            rendering_fn = self._render_as_list_text_to_surface
        elif isinstance(text, tuple):
            text = (self.preprocess_text_ws_only(text[0]), self.preprocess_text_ws_only(text[1]))
            rendering_fn = self._render_text_pair_to_surface
        elif isinstance(text, str):
            text = self.preprocess_text_ws_only(text)
            # rendering_fn = self._render_single_sentence
            rendering_fn = self._render_text_to_surface
        else:
            raise TypeError(
                f"{self.__class__.__name__} does not support inputs of type {type(text)}. Supported types are "
                f"{SUPPORTED_INPUT_TYPES}"
            )

        encoding = rendering_fn(
            text,
            return_overflowing_patches=return_overflowing_patches,
            return_offset_mapping=return_offset_mapping,
            stride=stride,
            rtl=rtl,
            **kwargs,
        )

        return encoding

    def load_font(self) -> None:
        """
        Loads the font from specified font file with specified font size and color.
        """

        logger.info(f"Loading font from {self.font_file}")

        manimpango.register_font(self.font_file)
        if self.fallback_fonts_dir is not None:
            for fallback_font in glob.glob(os.path.join(self.fallback_fonts_dir, "*tf")):
                logger.info(f"Loading fallback font {fallback_font}")
                manimpango.register_font(fallback_font)
        self.fonts_list = manimpango.list_fonts()

        font_family_name = ttLib.TTFont(self.font_file)["name"].getDebugName(1)

        scaled_font_size = (self.dpi / 72) * self.font_size
        font_str = f"{font_family_name} {scaled_font_size}px"
        self.font = Pango.font_description_from_string(font_str)