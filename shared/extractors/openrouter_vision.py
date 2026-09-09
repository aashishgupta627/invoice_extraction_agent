import base64
import json
import os
import re
import time
from typing import Optional

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI

from shared.logging_config import get_logger, with_thread

from .base import InvoiceExtractor

load_dotenv()  # Load .env

logger = get_logger("openrouter_vision")

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-flash-1.5")

# Strips ```json ... ``` or ``` ... ``` fences some models wrap the JSON in
# even when told not to.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)


class OpenRouterExtractor(InvoiceExtractor):
    def __init__(self, model: str = None, api_key: str = None):
        self.model = model or DEFAULT_MODEL
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env or environment")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

    def _pdf_to_images(self, pdf_path: str, dpi: int = 150) -> list:
        """Convert PDF pages to images (base64)."""
        doc = pymupdf.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img_data = pix.tobytes("png")
            images.append(base64.b64encode(img_data).decode("utf-8"))
        return images

    def _image_to_data_url(self, b64: str) -> str:
        return f"data:image/png;base64,{b64}"

    def _parse_model_output(self, result_text: Optional[str], tlog) -> dict:
        """Parse the model's response into a dict, logging exactly what
        went wrong (and what the model actually sent) when it fails —
        free vision models on OpenRouter frequently deviate from a clean
        JSON-only response (empty content, markdown fences, truncation,
        refusals), and a bare 'Model output not valid JSON' gives no way
        to tell those cases apart after the fact."""
        if not result_text or not result_text.strip():
            tlog.error(
                "Model returned empty content. This usually means: the "
                "model hit a length/rate limit, silently refused the "
                "image, or doesn't honor response_format=json_object. "
                "Raw content repr: %r" % result_text
            )
            raise ValueError("Model returned empty content (no JSON to parse)")

        # First attempt: content is already clean JSON.
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            pass

        # Second attempt: strip markdown code fences, then retry.
        stripped = _FENCE_RE.sub("", result_text).strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

        # Third attempt: extract the first {...} span out of surrounding text.
        match = _BRACE_RE.search(stripped)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Every strategy failed — log the full raw output so the actual
        # failure mode (truncated JSON, prose refusal, garbage, etc.) is
        # visible in the logs rather than guessed at.
        tlog.error(
            "Model output could not be parsed as JSON by any strategy "
            "(direct parse, fence-stripped parse, brace-extraction). "
            f"Raw output ({len(result_text)} chars):\n{result_text}"
        )
        raise ValueError("Model output not valid JSON")

    def extract(self, file_path: str) -> dict:
        tlog = with_thread(logger, file_path)

        # Convert to images
        if file_path.lower().endswith(".pdf"):
            images_b64 = self._pdf_to_images(file_path)
        else:
            with open(file_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            images_b64 = [b64]

        tlog.info(
            f"Calling {self.model} with {len(images_b64)} image(s), "
            f"total payload ~{sum(len(b) for b in images_b64) // 1024} KB (base64)"
        )

        # Build prompt – force JSON schema
        prompt = """
You are an invoice data extraction system. Extract the following fields from the invoice image.
Return ONLY a JSON object with this exact structure (no extra text):
{
  "seller": {"name": "", "gstin": "", "address": "", "state_code": ""},
  "buyer": {"name": "", "gstin": "", "address": "", "state_code": ""},
  "invoice_number": "",
  "invoice_date": "",
  "line_items": [{"description": "", "hsn_sac": "", "quantity": 0, "rate": 0.0, "amount": 0.0, "gstrate": 0.0}],
  "tax_breakup": {"taxable_value": 0.0, "cgst": 0.0, "sgst": 0.0, "igst": 0.0, "total_gst": 0.0},
  "total_amount": 0.0
}

**Instructions:** 
- The invoice number is usually labeled "Invoice No" or "Inv No". 
- The date is usually labeled "Date" or "Dated".
- For line items, the "amount" must be the **taxable amount before tax**, not the total including tax.
- For missing fields, use empty string or 0.0. Do not add any extra keys.

**CRITICAL:** 
- Extract the text EXACTLY as shown on the invoice.
- Do NOT correct or modify any data. Extract the text EXACTLY as shown on the invoice.
- If a GSTIN appears with 14 characters, extract those 14 characters – do NOT add or change anything.
- If a number is unreadable, use an empty string or 0.0, but do NOT guess or correct.
- - Return ONLY the JSON object, with no additional commentary or corrections.
        """
        content = [{"type": "text", "text": prompt}]
        for b64 in images_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": self._image_to_data_url(b64)}
            })

        start = time.monotonic()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
        except Exception:
            tlog.error(
                f"OpenRouter API call failed after {time.monotonic() - start:.2f}s",
                exc_info=True,
            )
            raise
        elapsed = time.monotonic() - start

        choice = response.choices[0]
        result_text = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)
        usage = getattr(response, "usage", None)
        tlog.debug(
            f"Model call took {elapsed:.2f}s, finish_reason={finish_reason}, "
            f"usage={usage}, content_len={len(result_text) if result_text else 0}"
        )
        if finish_reason and finish_reason != "stop":
            tlog.warning(
                f"Non-'stop' finish_reason={finish_reason!r} — response may be "
                "truncated or filtered."
            )

        data = self._parse_model_output(result_text, tlog)
        tlog.debug(f"Parsed extraction fields: {list(data.keys())}")

        return self._ensure_fields(data)

    def _ensure_fields(self, data: dict) -> dict:
        template = {
            "seller": {"name": "", "gstin": "", "address": "", "state_code": ""},
            "buyer": {"name": "", "gstin": "", "address": "", "state_code": ""},
            "invoice_number": "",
            "invoice_date": "",
            "line_items": [],
            "tax_breakup": {"taxable_value": 0.0, "cgst": 0.0, "sgst": 0.0, "igst": 0.0, "total_gst": 0.0},
            "total_amount": 0.0
        }
        # Recursively merge
        def merge(base, override):
            if isinstance(base, dict) and isinstance(override, dict):
                for k, v in base.items():
                    if k in override:
                        base[k] = merge(v, override[k])
                return base
            elif isinstance(base, list):
                # We only overwrite if override is list
                return override if isinstance(override, list) else base
            else:
                return override if override is not None and override != "" else base
        return merge(template, data)
