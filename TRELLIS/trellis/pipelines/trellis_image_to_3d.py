from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from torchvision import transforms
from PIL import Image
import rembg
from .base import Pipeline
from . import samplers
from ..modules import sparse as sp
from ..representations import Gaussian, Strivec, MeshExtractResult
import logging
import time

logger = logging.getLogger('trellis-api.pipeline')


class TrellisImageTo3DPipeline(Pipeline):
    """
    Pipeline for inferring Trellis image-to-3D models.

    Args:
        models (dict[str, nn.Module]): The models to use in the pipeline.
        sparse_structure_sampler (samplers.Sampler): The sampler for the sparse structure.
        slat_sampler (samplers.Sampler): The sampler for the structured latent.
        slat_normalization (dict): The normalization parameters for the structured latent.
        image_cond_model (str): The name of the image conditioning model.
    """

    def __init__(
        self,
        models: dict[str, nn.Module] = None,
        sparse_structure_sampler: samplers.Sampler = None,
        slat_sampler: samplers.Sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return
        super().__init__(models)
        self.sparse_structure_sampler = sparse_structure_sampler
        self.slat_sampler = slat_sampler
        self.sparse_structure_sampler_params = {}
        self.slat_sampler_params = {}
        self.slat_normalization = slat_normalization
        self.rembg_session = None
        self._init_image_cond_model(image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "TrellisImageTo3DPipeline":
        """
        Load a pretrained model.

        Args:
            path (str): The path to the model. Can be either local path or a Hugging Face repository.
        """
        pipeline = super(
            TrellisImageTo3DPipeline, TrellisImageTo3DPipeline
        ).from_pretrained(path)
        new_pipeline = TrellisImageTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(
            samplers, args["sparse_structure_sampler"]["name"]
        )(**args["sparse_structure_sampler"]["args"])
        new_pipeline.sparse_structure_sampler_params = args["sparse_structure_sampler"][
            "params"
        ]

        new_pipeline.slat_sampler = getattr(samplers, args["slat_sampler"]["name"])(
            **args["slat_sampler"]["args"]
        )
        new_pipeline.slat_sampler_params = args["slat_sampler"]["params"]

        new_pipeline.slat_normalization = args["slat_normalization"]

        new_pipeline._init_image_cond_model(args["image_cond_model"])

        return new_pipeline

    def _init_image_cond_model(self, name: str):
        """
        Initialize the image conditioning model.
        """
        dinov2_model = torch.hub.load("facebookresearch/dinov2", name, pretrained=True)
        dinov2_model.eval()
        self.models["image_cond_model"] = dinov2_model
        transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.image_cond_model_transform = transform

    def preprocess_image(self, input: Image.Image) -> Image.Image:
        """
        Preprocess the input image.
        """
        # if has alpha channel, use it directly; otherwise, remove background
        has_alpha = False
        if input.mode == "RGBA":
            alpha = np.array(input)[:, :, 3]
            if not np.all(alpha == 255):
                has_alpha = True
        if has_alpha:
            output = input
        else:
            input = input.convert("RGB")
            max_size = max(input.size)
            scale = min(1, 1024 / max_size)
            if scale < 1:
                input = input.resize(
                    (int(input.width * scale), int(input.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            if getattr(self, "rembg_session", None) is None:
                self.rembg_session = rembg.new_session("u2net")
            output = rembg.remove(input, session=self.rembg_session)
        output_np = np.array(output)
        alpha = output_np[:, :, 3]
        bbox = np.argwhere(alpha > 0.8 * 255)
        bbox = (
            np.min(bbox[:, 1]),
            np.min(bbox[:, 0]),
            np.max(bbox[:, 1]),
            np.max(bbox[:, 0]),
        )
        center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        size = int(size * 1.2)
        bbox = (
            center[0] - size // 2,
            center[1] - size // 2,
            center[0] + size // 2,
            center[1] + size // 2,
        )
        output = output.crop(bbox)  # type: ignore
        output = output.resize((518, 518), Image.Resampling.LANCZOS)
        output = np.array(output).astype(np.float32) / 255
        output = output[:, :, :3] * output[:, :, 3:4]
        output = Image.fromarray((output * 255).astype(np.uint8))
        return output

    @torch.no_grad()
    def encode_image(
        self, image: Union[torch.Tensor, list[Image.Image]]
    ) -> torch.Tensor:
        """
        Encode the image.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image to encode

        Returns:
            torch.Tensor: The encoded features.
        """
        if isinstance(image, torch.Tensor):
            assert image.ndim == 4, "Image tensor should be batched (B, C, H, W)"
        elif isinstance(image, list):
            assert all(
                isinstance(i, Image.Image) for i in image
            ), "Image list should be list of PIL images"
            image = [i.resize((518, 518), Image.LANCZOS) for i in image]
            image = [np.array(i.convert("RGB")).astype(np.float32) / 255 for i in image]
            image = [torch.from_numpy(i).permute(2, 0, 1).float() for i in image]
            image = torch.stack(image).to(self.device)
        else:
            raise ValueError(f"Unsupported type of image: {type(image)}")

        image = self.image_cond_model_transform(image).to(self.device)
        features = self.models["image_cond_model"](image, is_training=True)["x_prenorm"]
        patchtokens = F.layer_norm(features, features.shape[-1:])
        return patchtokens

    def get_cond(self, image: Union[torch.Tensor, list[Image.Image]]) -> dict:
        """
        Get the conditioning information for the model.

        Args:
            image (Union[torch.Tensor, list[Image.Image]]): The image prompts.

        Returns:
            dict: The conditioning information
        """
        cond = self.encode_image(image)
        neg_cond = torch.zeros_like(cond)
        return {
            "cond": cond,
            "neg_cond": neg_cond,
        }

    def sample_sparse_structure(
        self,
        cond: dict,
        num_samples: int = 1,
        sampler_params: dict = {},
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Sample sparse structures with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            num_samples (int): The number of samples to generate.
            sampler_params (dict): Additional parameters for the sampler.
        """
        # Sample occupancy latent
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(
            self.device
        )
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params, verbose=verbose
        ).samples

        # Decode occupancy latent
        decoder = self.models["sparse_structure_decoder"]
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()

        return coords

    def decode_slat(
        self,
        slat: sp.SparseTensor,
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
    ) -> dict:
        """
        Decode the structured latent.

        Args:
            slat (sp.SparseTensor): The structured latent.
            formats (List[str]): The formats to decode the structured latent to.

        Returns:
            dict: The decoded structured latent.
        """
        ret = {}
        if "mesh" in formats:
            ret["mesh"] = self.models["slat_decoder_mesh"](slat)
        if "gaussian" in formats:
            ret["gaussian"] = self.models["slat_decoder_gs"](slat)
        if "radiance_field" in formats:
            ret["radiance_field"] = self.models["slat_decoder_rf"](slat)
        return ret

    def sample_slat(
        self,
        cond: dict,
        coords: torch.Tensor,
        sampler_params: dict = {},
        progress_callback: Callable[[int], None] = None,
        verbose: bool = True,
    ) -> sp.SparseTensor:
        """
        Sample structured latent with the given conditioning.

        Args:
            cond (dict): The conditioning information.
            coords (torch.Tensor): The coordinates of the sparse structure.
            sampler_params (dict): Additional parameters for the sampler.
            progress_callback (Callable[[int], None]): Optional callback for progress tracking.
        """
        logger.debug(f"Starting SLAT sampling with {len(coords)} coordinates")
        
        # Sample structured latent
        flow_model = self.models["slat_flow_model"]
        noise = sp.SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels).to(self.device),
            coords=coords,
        )
        logger.debug(f"Created noise tensor with shape {noise.feats.shape}")
        
        sampler_params = {**self.slat_sampler_params, **sampler_params}
        slat = self.slat_sampler.sample(
            flow_model, noise, **cond, **sampler_params,
            verbose=verbose, progress_callback=progress_callback
        ).samples
        
        logger.debug("Applying normalization")
        std = torch.tensor(self.slat_normalization["std"])[None].to(slat.device)
        mean = torch.tensor(self.slat_normalization["mean"])[None].to(slat.device)
        slat = slat * std + mean
        
        logger.debug(f"SLAT sampling complete, tensor shape: {slat.feats.shape}")
        return slat

    @torch.no_grad()
    def run(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
    ) -> dict:
        """
        Run the pipeline.

        Args:
            image (Image.Image): The image prompt.
            num_samples (int): The number of samples to generate.
            sparse_structure_sampler_params (dict): Additional parameters for the sparse structure sampler.
            slat_sampler_params (dict): Additional parameters for the structured latent sampler.
            preprocess_image (bool): Whether to preprocess the image.
        """
        if preprocess_image:
            image = self.preprocess_image(image)
        cond = self.get_cond([image])
        torch.manual_seed(seed)
        coords = self.sample_sparse_structure(
            cond, num_samples, sparse_structure_sampler_params
        )
        slat = self.sample_slat(cond, coords, slat_sampler_params)
        return self.decode_slat(slat, formats)

    @torch.no_grad()
    def run_with_progress(
        self,
        image: Image.Image,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        slat_sampler_params: dict = {},
        formats: List[str] = ["mesh", "gaussian", "radiance_field"],
        preprocess_image: bool = True,
        progress_callback: Callable[[float], None] = None,
    ) -> dict:
        """
        Run the pipeline with progress tracking for each major step.
        Progress distribution (based on timing):
        - Preprocess & Conditioning: 0-3% (~2.4%)
        - Sparse Structure: 3-13% (~9.6%)
        - SLAT Sampling: 13-42% (~29.7%)
        - Format Decoding: 42-100% (~58.3%)
        """
        def update_progress(progress):
            if progress_callback:
                logger.debug(f"Pipeline update_progress called with {progress:.2f}%")
                progress_callback(min(progress, 100))

        phase_start = time.time()
        
        # Step 1: Preprocess & Conditioning (0-3%)
        preprocess_start = time.time()
        if preprocess_image:
            update_progress(1)
            image = self.preprocess_image(image)
            update_progress(2)
        
        update_progress(3)
        cond = self.get_cond([image])
        preprocess_time = time.time() - preprocess_start
        logger.debug(f"Preprocess and conditioning took {preprocess_time:.2f}s")

        # Step 2: Sparse Structure (3-13%)
        structure_start = time.time()
        torch.manual_seed(seed)
        logger.debug("Starting sparse structure sampling")
        
        flow_model = self.models["sparse_structure_flow_model"]
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
        sampler_params = {**self.sparse_structure_sampler_params, **sparse_structure_sampler_params}
        
        steps = sampler_params.get("steps", 20)
        def sparse_progress_callback(step):
            progress = 3 + (step / steps) * 10
            logger.debug(f"Sparse structure step {step}/{steps}, progress: {progress:.2f}%")
            update_progress(progress)
        
        z_s = self.sparse_structure_sampler.sample(
            flow_model, noise, **cond, **sampler_params,
            verbose=False, progress_callback=sparse_progress_callback
        ).samples
        
        decoder = self.models["sparse_structure_decoder"]
        logger.debug("Starting sparse structure decoding")
        coords = torch.argwhere(decoder(z_s) > 0)[:, [0, 2, 3, 4]].int()
        logger.debug(f"Found {len(coords)} coordinates")
        update_progress(13)
        
        structure_time = time.time() - structure_start
        logger.debug(f"Sparse structure sampling took {structure_time:.2f}s")

        # Step 3: SLAT Sampling (13-42%)
        slat_start = time.time()
        logger.debug("Starting SLAT sampling")
        steps = slat_sampler_params.get("steps", 20)
        def slat_progress_callback(step):
            progress = 13 + (step / steps) * 29
            logger.debug(f"SLAT sampling step {step}/{steps}, progress: {progress:.2f}%")
            update_progress(progress)
        
        slat = self.sample_slat(
            cond, coords, slat_sampler_params,
            verbose=False,
            progress_callback=slat_progress_callback
        )
        slat_time = time.time() - slat_start
        logger.debug(f"SLAT sampling took {slat_time:.2f}s")
        update_progress(42)

        # Step 4: Format Decoding (42-100%)
        decode_start = time.time()
        logger.debug("Starting format decoding")
        decode_range = 58  # 42-100%
        progress_per_format = decode_range / len(formats)
        
        results = {}
        format_times = {}
        for idx, fmt in enumerate(formats):
            format_start = time.time()
            start_progress = 42 + (idx * progress_per_format)
            end_progress = start_progress + progress_per_format
            logger.debug(f"Starting {fmt} decoding ({start_progress:.1f}% - {end_progress:.1f}%)")
            
            def format_callback(decoder_progress):
                overall_progress = start_progress + (decoder_progress / 100.0 * progress_per_format)
                logger.debug(f"{fmt} decoding progress: {decoder_progress}%, overall: {overall_progress:.2f}%")
                if progress_callback:
                    logger.debug(f"Calling progress_callback with {overall_progress:.2f}%")
                update_progress(overall_progress)
            
            if fmt == "gaussian":
                decoder = self.models["slat_decoder_gs"]
                results[fmt] = decoder(slat, callback=format_callback)
            elif fmt == "mesh":
                decoder = self.models["slat_decoder_mesh"]
                results[fmt] = decoder(slat, callback=format_callback)
            elif fmt == "radiance_field":
                decoder = self.models["slat_decoder_rf"]
                results[fmt] = decoder(slat, callback=format_callback)
            
            format_times[fmt] = time.time() - format_start
            logger.debug(f"{fmt} decoding took {format_times[fmt]:.2f}s")
        
        decode_time = time.time() - decode_start
        logger.debug(f"Total decoding took {decode_time:.2f}s")

        # Final timing summary
        total_time = time.time() - phase_start
        format_summary = "\n          ".join([f"- {fmt}: {t:.2f}s ({(t/total_time)*100:.1f}%)" 
                                            for fmt, t in format_times.items()])

        logger.debug(f"""Pipeline timing breakdown:
            Preprocess & Conditioning: {preprocess_time:.2f}s ({(preprocess_time/total_time)*100:.1f}%)
            Sparse Structure: {structure_time:.2f}s ({(structure_time/total_time)*100:.1f}%)
            SLAT Sampling: {slat_time:.2f}s ({(slat_time/total_time)*100:.1f}%)
            Format Decoding: {decode_time:.2f}s ({(decode_time/total_time)*100:.1f}%)
                  {format_summary}
            Total Pipeline: {total_time:.2f}s
        """)
        
        return results


