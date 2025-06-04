import copy
import fnmatch
import json
import logging
import os
import shutil
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, BinaryIO
from urllib.parse import urlparse
from zipfile import is_zipfile, ZipFile

import numpy as np
import requests
from filelock import FileLock
from huggingface_hub.hf_api import HfFolder
from requests import HTTPError
from tqdm import tqdm
from transformers.dynamic_module_utils import custom_object_save
from transformers.file_utils import (EntryNotFoundError, PushToHubMixin,
                                     RepositoryNotFoundError,
                                     RevisionNotFoundError,
                                     copy_func, is_offline_mode, TRANSFORMERS_CACHE, http_user_agent, HUGGINGFACE_CO_PREFIX)

logger = logging.getLogger(__name__)

TEXT_RENDERER_NAME = "text_renderer_config.json"

PreTrainedTextRenderer = Union["PyGameTextRenderer", "PangoCairoTextRenderer"]  # noqa: F821


@dataclass
class Encoding:
    """
    Dataclass storing renderer outputs

    Args:
        pixel_values (`numpy.ndarray`):
            A 3D numpy array containing the pixel values of a rendered image
        sep_patches (`List[int]`):
            A list containing the starting indices (patch-level) at which black separator patches were inserted in the
            image.
        num_text_patches (`int`):
            The number of patches in the image containing text (excluding the final black sep patch). This value is
            e.g. used to construct an attention mask.
        word_starts (`List[int]`, *optional*, defaults to None):
            A list containing the starting index (patch-level) of every word in the rendered sentence. This value is
            set when rendering texts word-by-word (i.e., when calling a renderer with a list of strings/words).
        offset_mapping (`List[Tuple[int, int]]`, *optional*, defaults to None):
            A list containing `(char_start, char_end)` for each image patch to map between text and rendered image.
        overflowing_patches (`List[Encoding]`, *optional*, defaults to None):
            A list of overflowing patch sequences (of type `Encoding`). Used in sliding window approaches, e.g. for
            question answering.
        sequence_ids (`[List[Optional[int]]`, *optional*, defaults to None):
            A list that can be used to distinguish between sentences in sentence pairs: 0 for sentence_a, 1 for
            sentence_b, and None for special patches.
    """

    pixel_values: np.ndarray
    sep_patches: List[int]
    num_text_patches: int
    word_starts: Optional[List[int]] = None
    word_patch_map: Optional[Dict[int, str]] = None
    offset_mapping: Optional[List[Tuple[int, int]]] = None
    overflowing_patches: Optional[List] = None
    sequence_ids: Optional[List[Optional[int]]] = None


def _raise_for_status(request):
    """
    Internal version of `request.raise_for_status()` that will refine a potential HTTPError.
    """
    if "X-Error-Code" in request.headers:
        error_code = request.headers["X-Error-Code"]
        if error_code == "RepoNotFound":
            raise RepositoryNotFoundError(f"404 Client Error: Repository Not Found for url: {request.url}")
        elif error_code == "EntryNotFound":
            raise EntryNotFoundError(f"404 Client Error: Entry Not Found for url: {request.url}")
        elif error_code == "RevisionNotFound":
            raise RevisionNotFoundError((f"404 Client Error: Revision Not Found for url: {request.url}"))

    request.raise_for_status()


def url_to_filename(url: str, etag: Optional[str] = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way. If `etag` is specified, append its hash to the url's,
    delimited by a period. If the url ends with .h5 (Keras HDF5 weights) adds '.h5' to the name so that TF 2.0 can
    identify it as a HDF5 file (see
    https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    url_bytes = url.encode("utf-8")
    filename = sha256(url_bytes).hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        filename += "." + sha256(etag_bytes).hexdigest()

    if url.endswith(".h5"):
        filename += ".h5"

    return filename


def is_remote_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def http_get(url: str, temp_file: BinaryIO, proxies=None, resume_size=0, headers: Optional[Dict[str, str]] = None):
    """
    Download remote file. Do not gobble up errors.
    """
    headers = copy.deepcopy(headers)
    if resume_size > 0:
        headers["Range"] = f"bytes={resume_size}-"
    r = requests.get(url, stream=True, proxies=proxies, headers=headers)
    _raise_for_status(r)
    content_length = r.headers.get("Content-Length")
    total = resume_size + int(content_length) if content_length is not None else None
    # `tqdm` behavior is determined by `utils.logging.is_progress_bar_enabled()`
    # and can be set using `utils.logging.enable/disable_progress_bar()`
    progress = tqdm(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        total=total,
        initial=resume_size,
        desc="Downloading",
    )
    for chunk in r.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(
    url: str,
    cache_dir=None,
    force_download=False,
    proxies=None,
    etag_timeout=10,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given a URL, look for the corresponding file in the local cache. If it's not there, download it. Then return the
    path to the cached file.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    headers = {"user-agent": http_user_agent(user_agent)}
    if isinstance(use_auth_token, str):
        headers["authorization"] = f"Bearer {use_auth_token}"
    elif use_auth_token:
        token = HfFolder.get_token()
        if token is None:
            raise EnvironmentError("You specified use_auth_token=True, but a huggingface token was not found.")
        headers["authorization"] = f"Bearer {token}"

    url_to_download = url
    etag = None
    if not local_files_only:
        try:
            r = requests.head(url, headers=headers, allow_redirects=False, proxies=proxies, timeout=etag_timeout)
            _raise_for_status(r)
            etag = r.headers.get("X-Linked-Etag") or r.headers.get("ETag")
            # We favor a custom header indicating the etag of the linked resource, and
            # we fallback to the regular etag header.
            # If we don't have any of those, raise an error.
            if etag is None:
                raise OSError(
                    "Distant resource does not have an ETag, we won't be able to reliably ensure reproducibility."
                )
            # In case of a redirect,
            # save an extra redirect on the request.get call,
            # and ensure we download the exact atomic version even if it changed
            # between the HEAD and the GET (unlikely, but hey).
            if 300 <= r.status_code <= 399:
                url_to_download = r.headers["Location"]
        except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
            # Actually raise for those subclasses of ConnectionError
            raise
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Otherwise, our Internet connection is down.
            # etag is None
            pass

    filename = url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)

    # etag is None == we don't have a connection or we passed local_files_only.
    # try to get the last downloaded one
    if etag is None:
        if os.path.exists(cache_path):
            return cache_path
        else:
            matching_files = [
                file
                for file in fnmatch.filter(os.listdir(cache_dir), filename.split(".")[0] + ".*")
                if not file.endswith(".json") and not file.endswith(".lock")
            ]
            if len(matching_files) > 0:
                return os.path.join(cache_dir, matching_files[-1])
            else:
                # If files cannot be found and local_files_only=True,
                # the models might've been found if local_files_only=False
                # Notify the user about that
                if local_files_only:
                    raise FileNotFoundError(
                        "Cannot find the requested files in the cached path and outgoing traffic has been"
                        " disabled. To enable model look-ups and downloads online, set 'local_files_only'"
                        " to False."
                    )
                else:
                    raise ValueError(
                        "Connection error, and we cannot find the requested files in the cached path."
                        " Please try again or make sure your Internet connection is on."
                    )

    # From now on, etag is not None.
    if os.path.exists(cache_path) and not force_download:
        return cache_path

    # Prevent parallel downloads of the same file with a lock.
    lock_path = cache_path + ".lock"
    with FileLock(lock_path):

        # If the download just completed while the lock was activated.
        if os.path.exists(cache_path) and not force_download:
            # Even if returning early like here, the lock will be released.
            return cache_path

        if resume_download:
            incomplete_path = cache_path + ".incomplete"

            @contextmanager
            def _resumable_file_manager() -> "io.BufferedWriter":
                with open(incomplete_path, "ab") as f:
                    yield f

            temp_file_manager = _resumable_file_manager
            if os.path.exists(incomplete_path):
                resume_size = os.stat(incomplete_path).st_size
            else:
                resume_size = 0
        else:
            temp_file_manager = partial(tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False)
            resume_size = 0

        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info(f"{url} not found in cache or force_download set to True, downloading to {temp_file.name}")

            http_get(url_to_download, temp_file, proxies=proxies, resume_size=resume_size, headers=headers)

        logger.info(f"storing {url} in cache at {cache_path}")
        os.replace(temp_file.name, cache_path)

        # NamedTemporaryFile creates a file with hardwired 0600 perms (ignoring umask), so fixing it.
        umask = os.umask(0o666)
        os.umask(umask)
        os.chmod(cache_path, 0o666 & ~umask)

        logger.info(f"creating metadata file for {cache_path}")
        meta = {"url": url, "etag": etag}
        meta_path = cache_path + ".json"
        with open(meta_path, "w") as meta_file:
            json.dump(meta, meta_file)

    return cache_path


def cached_path(
    url_or_filename,
    cache_dir=None,
    force_download=False,
    proxies=None,
    resume_download=False,
    user_agent: Union[Dict, str, None] = None,
    extract_compressed_file=False,
    force_extract=False,
    use_auth_token: Union[bool, str, None] = None,
    local_files_only=False,
) -> Optional[str]:
    """
    Given something that might be a URL (or might be a local path), determine which. If it's a URL, download the file
    and cache it, and return the path to the cached file. If it's already a local path, make sure the file exists and
    then return the path

    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-download the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletely received file is found.
        user_agent: Optional string or dict that will be appended to the user-agent on remote requests.
        use_auth_token: Optional string or boolean to use as Bearer token for remote files. If True,
            will get token from ~/.huggingface.
        extract_compressed_file: if True and the path point to a zip or tar file, extract the compressed
            file in a folder along the archive.
        force_extract: if True when extract_compressed_file is True and the archive was already extracted,
            re-extract the archive and override the folder where it was extracted.

    Return:
        Local path (string) of file or if networking is off, last version of file cached on disk.

    Raises:
        In case of non-recoverable file (non-existent or inaccessible url + no cache on disk).
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    if is_offline_mode() and not local_files_only:
        logger.info("Offline mode: forcing local_files_only=True")
        local_files_only = True

    if is_remote_url(url_or_filename):
        # URL, so get it from the cache (downloading if necessary)
        output_path = get_from_cache(
            url_or_filename,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            user_agent=user_agent,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
        )
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        output_path = url_or_filename
    elif urlparse(url_or_filename).scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError(f"file {url_or_filename} not found")
    else:
        # Something unknown
        raise ValueError(f"unable to parse {url_or_filename} as a URL or as a local path")

    if extract_compressed_file:
        if not is_zipfile(output_path) and not tarfile.is_tarfile(output_path):
            return output_path

        # Path where we extract compressed archives
        # We avoid '.' in dir name and add "-extracted" at the end: "./model.zip" => "./model-zip-extracted/"
        output_dir, output_file = os.path.split(output_path)
        output_extract_dir_name = output_file.replace(".", "-") + "-extracted"
        output_path_extracted = os.path.join(output_dir, output_extract_dir_name)

        if os.path.isdir(output_path_extracted) and os.listdir(output_path_extracted) and not force_extract:
            return output_path_extracted

        # Prevent parallel extractions
        lock_path = output_path + ".lock"
        with FileLock(lock_path):
            shutil.rmtree(output_path_extracted, ignore_errors=True)
            os.makedirs(output_path_extracted)
            if is_zipfile(output_path):
                with ZipFile(output_path, "r") as zip_file:
                    zip_file.extractall(output_path_extracted)
                    zip_file.close()
            elif tarfile.is_tarfile(output_path):
                tar_file = tarfile.open(output_path)
                tar_file.extractall(output_path_extracted)
                tar_file.close()
            else:
                raise EnvironmentError(f"Archive format of {output_path} could not be identified")

        return output_path_extracted

    return output_path


def hf_bucket_url(
    model_id: str, filename: str, subfolder: Optional[str] = None, revision: Optional[str] = None, mirror=None
) -> str:
    """
    Resolve a model identifier, a file name, and an optional revision id, to a huggingface.co-hosted url, redirecting
    to Cloudfront (a Content Delivery Network, or CDN) for large files.

    Cloudfront is replicated over the globe so downloads are way faster for the end user (and it also lowers our
    bandwidth costs).

    Cloudfront aggressively caches files by default (default TTL is 24 hours), however this is not an issue here
    because we migrated to a git-based versioning system on huggingface.co, so we now store the files on S3/Cloudfront
    in a content-addressable way (i.e., the file name is its hash). Using content-addressable filenames means cache
    can't ever be stale.

    In terms of client-side caching from this library, we base our caching on the objects' ETag. An object' ETag is:
    its sha1 if stored in git, or its sha256 if stored in git-lfs. Files cached locally from transformers before v3.5.0
    are not shared with those new files, because the cached file's name contains a hash of the url (which changed).
    """
    if subfolder is not None:
        filename = f"{subfolder}/{filename}"

    if mirror:
        if mirror in ["tuna", "bfsu"]:
            raise ValueError("The Tuna and BFSU mirrors are no longer available. Try removing the mirror argument.")
        legacy_format = "/" not in model_id
        if legacy_format:
            return f"{mirror}/{model_id}-{filename}"
        else:
            return f"{mirror}/{model_id}/{filename}"

    if revision is None:
        revision = "main"
    return HUGGINGFACE_CO_PREFIX.format(model_id=model_id, revision=revision, filename=filename)


class TextRenderingMixin(PushToHubMixin):
    """
    This is a text rendering mixin used to provide saving/loading functionality for text renderers.
    """

    _auto_class = None

    def __init__(self, **kwargs):
        """Set elements of `kwargs` as attributes."""
        # Additional attributes without default values
        for key, value in kwargs.items():
            try:
                setattr(self, key, value)
            except AttributeError as err:
                logger.error(f"Can't set {key} with value {value} for {self}")
                raise err

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> PreTrainedTextRenderer:
        r"""
        Instantiate a type of [`~text_rendering_utils.TextRenderingMixin`] from a text renderer, *e.g.* a
        derived class of [`PangoCairoTextRenderer`] or [`PyGameTextRenderer`].
        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - a string, the *model id* of a pretrained text renderer hosted inside a model repo on
                  huggingface.co. Valid model ids can be located at the root-level, like `bert-base-uncased`, or
                  namespaced under a user or organization name, like `dbmdz/bert-base-german-cased`.
                - a path to a *directory* containing a text renderer file saved using the
                  [`~renderer.TextRenderingMixin.save_pretrained`] method, e.g.,
                  `./my_model_directory/`.
                - a path or url to a saved text renderer JSON *file*, e.g.,
                  `./my_model_directory/text_renderer_config.json`.
            cache_dir (`str` or `os.PathLike`, *optional*):
                Path to a directory in which a downloaded pretrained model text renderer should be cached if the
                standard cache should not be used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force to (re-)download the text renderer files and override the cached versions
                if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to delete incompletely received file. Attempts to resume the download if such a file
                exists.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}.` The proxies are used on each request.
            use_auth_token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `transformers-cli login` (stored in `~/.huggingface`).
            revision(`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                If `False`, then this function returns just the final text renderer object. If `True`, then this
                functions returns a `Tuple(text_renderer, unused_kwargs)` where *unused_kwargs* is a dictionary
                consisting of the key/value pairs whose keys are not text renderer attributes: i.e., the part of
                `kwargs` which has not been used to update `text_renderer` and is otherwise ignored.
            kwargs (`Dict[str, Any]`, *optional*):
                The values in kwargs of any keys which are text renderer attributes will be used to override the
                loaded values. Behavior concerning key/value pairs whose keys are *not* text renderer attributes is
                controlled by the `return_unused_kwargs` keyword parameter.
        <Tip>
        Passing `use_auth_token=True` is required when you want to use a private model.
        </Tip>
        Returns:
            A text renderer of type [`~text_rendering_utils.TextRenderingMixin`].
        Examples:
        ```python
        # We can't instantiate directly the base class *TextRenderingMixin* so let's show the examples on a
        # derived class: *PyGameTextRenderer*
        text_renderer = PyGameTextRenderer.from_pretrained(
            "Team-PIXEL/pixel-base"
        )
        # or *PangoCairoTextRenderer*
                text_renderer = PyGameTextRenderer.from_pretrained(
            "Team-PIXEL/pixel-base"
        )
        ```"""
        text_renderer_dict, kwargs = cls.get_text_renderer_dict(pretrained_model_name_or_path, **kwargs)
        text_renderer_dict, kwargs = cls.resolve_and_update_font_file(
            pretrained_model_name_or_path, text_renderer_dict, **kwargs
        )

        return cls.from_dict(text_renderer_dict, **kwargs)

    def save_pretrained(self, save_directory: Union[str, os.PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a text_renderer object to the directory `save_directory`, so that it can be re-loaded using the
        [`~.PyGameTextRenderer.from_pretrained`] or [`~.PangoCairoTextRenderer.from_pretrained`] class method.
        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the text renderer JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your text renderer to the Hugging Face model hub after saving it.
                <Tip warning={true}>
                Using `push_to_hub=True` will synchronize the repository you are pushing to with `save_directory`,
                which requires `save_directory` to be a local clone of the repo you are pushing to if it's an existing
                folder. Pass along `temp_dir=True` to use a temporary directory instead.
                </Tip>
            kwargs:
                Additional key word arguments passed along to the [`~file_utils.PushToHubMixin.push_to_hub`] method.
        """

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo = self._create_or_get_repo(save_directory, **kwargs)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        os.makedirs(save_directory, exist_ok=True)
        # If we save using the predefined names, we can load using `from_pretrained`
        output_text_renderer_file = os.path.join(save_directory, TEXT_RENDERER_NAME)

        self.to_json_file(output_text_renderer_file)

        logger.info(f"Text renderer saved in {output_text_renderer_file}")

        # Also save the font file
        self.copy_font_file_to_save_dir(save_directory)

        if push_to_hub:
            url = self._push_to_hub(repo, commit_message=commit_message)
            logger.info(f"Text renderer pushed to the hub in this commit: {url}")

    @classmethod
    def get_text_renderer_dict(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        text renderer of type [`~text_rendering_utils.TextRenderingMixin`] using `from_dict`.
        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the text renderer object.
        """
        cache_dir = kwargs.get("cache_dir", None)
        force_download = kwargs.get("force_download", False)
        resume_download = kwargs.get("resume_download", False)
        proxies = kwargs.get("proxies", None)
        use_auth_token = kwargs.get("use_auth_token", None)
        local_files_only = kwargs.get("local_files_only", False)
        revision = kwargs.get("revision", None)

        from_pipeline = kwargs.get("_from_pipeline", None)
        from_auto_class = kwargs.get("_from_auto", False)

        user_agent = {"file_type": "text renderer", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            text_renderer_file = os.path.join(pretrained_model_name_or_path, TEXT_RENDERER_NAME)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            text_renderer_file = pretrained_model_name_or_path
        else:
            text_renderer_file = hf_bucket_url(
                pretrained_model_name_or_path, filename=TEXT_RENDERER_NAME, revision=revision, mirror=None
            )

        try:
            # Load from URL or cache if already cached
            resolved_text_renderer_file = cached_path(
                text_renderer_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier listed on "
                "'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token having "
                "permission to this repo with `use_auth_token` or log in with `huggingface-cli login` and pass "
                "`use_auth_token=True`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
                f"model name. Check the model page at 'https://huggingface.co/{pretrained_model_name_or_path}' for "
                "available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {TEXT_RENDERER_NAME}."
            )
        except HTTPError:
            raise EnvironmentError(
                "We couldn't connect to 'https://huggingface.co/' to load this model and it looks like "
                f"{pretrained_model_name_or_path} is not the path to a directory conaining a "
                f"{TEXT_RENDERER_NAME} file.\nCheckout your internet connection or see how to run the library in "
                "offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load text renderer for '{pretrained_model_name_or_path}'. If you were trying to load it "
                "from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a {TEXT_RENDERER_NAME} file"
            )

        try:
            # Load text_renderer dict
            with open(resolved_text_renderer_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            text_renderer_dict = json.loads(text)

        except json.JSONDecodeError:
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_text_renderer_file}' is not a valid JSON file."
            )

        if resolved_text_renderer_file == text_renderer_file:
            logger.info(f"loading text renderer configuration file {text_renderer_file}")
        else:
            logger.info(
                f"loading text renderer configuration file {text_renderer_file} from cache at {resolved_text_renderer_file}"
            )

        return text_renderer_dict, kwargs

    @classmethod
    def resolve_and_update_font_file(
        cls, pretrained_model_name_or_path: Union[str, os.PathLike], text_renderer_dict: Dict[str, Any], **kwargs
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        From a `pretrained_model_name_or_path`, resolve to a dictionary of parameters, to be used for instantiating a
        text renderer of type [`~text_rendering_utils.TextRenderingMixin`] using `from_dict`.
        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                The identifier of the pre-trained checkpoint from which we want the dictionary of parameters.
            text_renderer_dict (`Dict[str, Any]`):
                The resolved dictionary of parameters, to be used for instantiating a
                text renderer of type [`~text_rendering_utils.TextRenderingMixin`] using `from_dict`.
        Returns:
            `Tuple[Dict, Dict]`: The dictionary(ies) that will be used to instantiate the text renderer object.
        """

        font_file_name = text_renderer_dict.get("font_file")

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        use_auth_token = kwargs.pop("use_auth_token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)

        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)

        user_agent = {"file_type": "text renderer font file", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            font_file = os.path.join(pretrained_model_name_or_path, font_file_name)
        elif os.path.isfile(pretrained_model_name_or_path) or is_remote_url(pretrained_model_name_or_path):
            font_file = pretrained_model_name_or_path
        else:
            font_file = hf_bucket_url(
                pretrained_model_name_or_path, filename=font_file_name, revision=revision, mirror=None
            )

        try:
            # Load from URL or cache if already cached
            resolved_font_file = cached_path(
                font_file,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                resume_download=resume_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                user_agent=user_agent,
            )

        except RepositoryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} is not a local folder and is not a valid model identifier listed on "
                "'https://huggingface.co/models'\nIf this is a private repository, make sure to pass a token having "
                "permission to this repo with `use_auth_token` or log in with `huggingface-cli login` and pass "
                "`use_auth_token=True`."
            )
        except RevisionNotFoundError:
            raise EnvironmentError(
                f"{revision} is not a valid git identifier (branch name, tag name or commit id) that exists for this "
                f"model name. Check the model page at 'https://huggingface.co/{pretrained_model_name_or_path}' for "
                "available revisions."
            )
        except EntryNotFoundError:
            raise EnvironmentError(
                f"{pretrained_model_name_or_path} does not appear to have a file named {font_file}."
            )
        except HTTPError:
            raise EnvironmentError(
                "We couldn't connect to 'https://huggingface.co/' to load this model and it looks like "
                f"{pretrained_model_name_or_path} is not the path to a directory conaining a "
                f"{font_file} file.\nCheckout your internet connection or see how to run the library in "
                "offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'."
            )
        except EnvironmentError:
            raise EnvironmentError(
                f"Can't load text renderer font file for '{pretrained_model_name_or_path}'. If you were trying to load it "
                "from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. "
                f"Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a directory "
                f"containing a {font_file} file"
            )

        if resolved_font_file == font_file:
            logger.info(f"loading font file {font_file}")
        else:
            logger.info(f"loading font file {font_file} from cache at {resolved_font_file}")

        text_renderer_dict["font_file"] = resolved_font_file

        return text_renderer_dict, kwargs

    @classmethod
    def from_dict(cls, text_renderer_dict: Dict[str, Any], **kwargs) -> PreTrainedTextRenderer:
        """
        Instantiates a type of [`~text_rendering_utils.TextRenderingMixin`] from a Python dictionary of
        parameters.
        Args:
            text_renderer_dict (`Dict[str, Any]`):
                Dictionary that will be used to instantiate the text renderer object. Such a dictionary can be
                retrieved from a pretrained checkpoint by leveraging the
                [`~text_rendering_utils.TextRenderingMixin.to_dict`] method.
            kwargs (`Dict[str, Any]`):
                Additional parameters from which to initialize the text renderer object.
        Returns:
            [`~text_rendering_utils.TextRenderingMixin`]: The text renderer object instantiated from those
            parameters.
        """
        return_unused_kwargs = kwargs.pop("return_unused_kwargs", False)

        if "fallback_fonts_dir" in kwargs:
            fallback_fonts_dir = kwargs.pop("fallback_fonts_dir")
            text_renderer_dict.update({"fallback_fonts_dir": fallback_fonts_dir})

        text_renderer = cls(**text_renderer_dict)

        # Update text_renderer with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(text_renderer, key):
                setattr(text_renderer, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info(f"Text renderer {text_renderer}")
        if return_unused_kwargs:
            return text_renderer, kwargs
        else:
            return text_renderer

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.
        Returns:
            `Dict[str, Any]`: Dictionary of all the attributes that make up this text renderer instance.
        """
        output = copy.deepcopy(self.__getstate__())
        output["text_renderer_type"] = self.__class__.__name__

        return output

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PreTrainedTextRenderer:
        """
        Instantiates a text renderer of type [`~text_rendering_utils.TextRenderingMixin`] from the path to
        a JSON file of parameters.
        Args:
            json_file (`str` or `os.PathLike`):
                Path to the JSON file containing the parameters.
        Returns:
            A text renderer of type [`~text_rendering_utils.TextRenderingMixin`]: The text_Renderer
            object instantiated from that JSON file.
        """
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        text_renderer_dict = json.loads(text)
        return cls(**text_renderer_dict)

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.
        Returns:
            `str`: String containing all the attributes that make up this text_renderer instance in JSON format.
        """
        dictionary = self.to_dict()

        for key, value in dictionary.items():
            if isinstance(value, np.ndarray):
                dictionary[key] = value.tolist()

        # make sure only the basename of the font file is stored
        font_file = dictionary.pop("font_file", None)
        if font_file is not None:
            dictionary["font_file"] = os.path.basename(font_file)

        return json.dumps(dictionary, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.
        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this text_renderer instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def copy_font_file_to_save_dir(self, save_directory: Union[str, os.PathLike]):
        """
        Copy font file from resolved font filepath to save directory.
        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the font file will be saved.
        """
        if not os.path.isdir(save_directory):
            raise EnvironmentError(
                f"{save_directory} does not appear to exist. Please double-check the spelling"
                f"or create the directory if necessary"
            )

        if not os.path.isfile(self.font_file):
            raise EnvironmentError(
                f"{self.font_file} does not appear to exist. Please ensure the attribute is set"
                f"correctly and the font file exists."
            )

        try:
            destination_path = shutil.copy(self.font_file, save_directory)

            logger.info(f"Text renderer font file saved in {destination_path}")

        except shutil.SameFileError as e:
            logger.warning(
                f"Font file not copied to {save_directory} because {e}. If this is unintended, please check "
                f"the text renderer font file path and the save directory."
            )

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    @classmethod
    def register_for_auto_class(cls, auto_class="AutoTextRenderer"):
        """
        Register this class with a given auto class. This should only be used for custom text renderers as the ones
        in the library are already mapped with `AutoTextRenderer`.
        <Tip warning={true}>
        This API is experimental and may have some slight breaking changes in the next releases.
        </Tip>
        Args:
            auto_class (`str` or `type`, *optional*, defaults to `"AutoTextRenderer"`):
                The auto class to register this new text renderer with.
        """
        if not isinstance(auto_class, str):
            auto_class = auto_class.__name__

        import transformers.models.auto as auto_module

        if not hasattr(auto_module, auto_class):
            raise ValueError(f"{auto_class} is not a valid auto class.")

        cls._auto_class = auto_class


TextRenderingMixin.push_to_hub = copy_func(TextRenderingMixin.push_to_hub)
TextRenderingMixin.push_to_hub.__doc__ = TextRenderingMixin.push_to_hub.__doc__.format(
    object="text renderer", object_class="AutoTextRenderer", object_files="text renderer file"
)
