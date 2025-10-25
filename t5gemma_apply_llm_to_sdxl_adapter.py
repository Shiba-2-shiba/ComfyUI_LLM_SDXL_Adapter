import logging
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger("LLM-SDXL-Adapter")


class t5gemmaApplyLLMToSDXLAdapter:
    """ComfyUI node that formats T5Gemma adapter outputs for SDXL conditioning."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "llm_hidden_states": ("LLM_HIDDEN_STATES",),
                "llm_attention_mask": ("LLM_ATTENTION_MASK",),
                "llm_adapter": ("LLM_ADAPTER",),
            },
            "optional": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "target_height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "crop_w": ("INT", {"default": 0, "min": 0, "max": 8192}),
                "crop_h": ("INT", {"default": 0, "min": 0, "max": 8192}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"
    CATEGORY = "llm_sdxl"

    def apply(
        self,
        llm_hidden_states: torch.Tensor,
        llm_attention_mask: Optional[torch.Tensor],
        llm_adapter: Any,
        *,
        width: Optional[Any] = None,
        height: Optional[Any] = None,
        target_width: Optional[Any] = None,
        target_height: Optional[Any] = None,
        crop_w: Optional[Any] = None,
        crop_h: Optional[Any] = None,
    ) -> Tuple[Any, ...]:
        """Apply the adapter and package the result for ComfyUI."""

        try:
            prompt_embeds, pooled_output = self._run_adapter(
                llm_adapter, llm_hidden_states, llm_attention_mask
            )
        except Exception as exc:  # pragma: no cover - surfaced to ComfyUI
            logger.error("Failed to apply adapter: %s", exc)
            raise

        prompt_embeds = prompt_embeds.cpu().contiguous()
        pooled_output = pooled_output.cpu().contiguous()

        metadata = self._build_metadata(
            pooled_output=pooled_output,
            width=width,
            height=height,
            target_width=target_width,
            target_height=target_height,
            crop_w=crop_w,
            crop_h=crop_h,
        )

        conditioning = [[prompt_embeds, metadata]]
        return (conditioning,)

    @staticmethod
    def _run_adapter(
        llm_adapter: Any,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute the adapter with inference safeguards."""

        adapter_infer = getattr(llm_adapter, "infer", None)

        with torch.inference_mode():
            if callable(adapter_infer):
                return adapter_infer(hidden_states, attention_mask=attention_mask)
            return llm_adapter(hidden_states, attention_mask=attention_mask)

    @staticmethod
    def _build_metadata(
        *,
        pooled_output: torch.Tensor,
        width: Optional[Any],
        height: Optional[Any],
        target_width: Optional[Any],
        target_height: Optional[Any],
        crop_w: Optional[Any],
        crop_h: Optional[Any],
    ) -> Dict[str, Any]:
        """Assemble the metadata dictionary expected by ComfyUI."""

        metadata: Dict[str, Any] = {"pooled_output": pooled_output}

        def _as_int(value: Optional[Any]) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return int(value.item())
            return int(value)

        if width is not None and height is not None:
            metadata.update({"width": _as_int(width), "height": _as_int(height)})
        if target_width is not None and target_height is not None:
            metadata.update(
                {"target_width": _as_int(target_width), "target_height": _as_int(target_height)}
            )
        if crop_w is not None and crop_h is not None:
            metadata.update({"crop_w": _as_int(crop_w), "crop_h": _as_int(crop_h)})

        return metadata

NODE_CLASS_MAPPINGS = {
    "t5gemmaApplyLLMToSDXLAdapter": t5gemmaApplyLLMToSDXLAdapter
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "t5gemmaApplyLLMToSDXLAdapter": "Apply T5Gemma LLM to Adapter"
}
