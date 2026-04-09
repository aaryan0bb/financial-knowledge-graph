# PDF to Text Converter with Figure Analysis
#
# This script converts PDFs to enriched text format by:
# 1. Extracting text using LLMWhisperer (high quality mode)
# 2. Detecting and analyzing figures using MinerU and Gemini Vision
# 3. Providing 3-page context window for each figure analysis
# 4. Merging text and figure analysis into a structured output
#
# Output format:
# - Main text preserved with page delimiters
# - Figure analysis embedded at appropriate locations
# - Rich context from surrounding pages included
#
# Dependencies:
# - MinerU: For figure detection
# - Gemini Vision: For figure analysis
# - LLMWhisperer: For high-quality text extraction

from __future__ import annotations

import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Dict, List, Tuple, NamedTuple

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    class _LoggerShim:
        def __getattr__(self, name):
            return getattr(logging, name.upper(), logging.info)

    logger = _LoggerShim()  # type: ignore

# --------------------------- Prompt & Constants ---------------------------
PROMPT_TEMPLATE = """
SYSTEM:
You are ChartGPT – a financial analyst and visual chart interpreter.
Your job is to convert financial or economic figures into structured data and rich, narrative insights.
Ignore logos, icons, or decorative shapes. Focus only on visual elements with structured data (e.g., time series, bar/line plots, labeled tables, annotated charts).

USER:
For the attached figure **{image_filename_placeholder}**, and using the **full-page context provided**, return:

1. A detailed **Markdown table** showing the key structured data from the figure.
   - Use meaningful column headers (e.g., Date, Index, Exposure, Weekly Change, Region, etc.)
   - Do **not limit to 3 columns**. Include all visible columns in the chart.
   - Use **up to 30 rows**, rounded and abbreviated as needed (e.g., +3.1%, $2.2bn, ‑12 bps)

2. A **rich summary paragraph (4–6 lines)** that explains:
   - The key dimensions of the chart (e.g., X/Y axes, categories/legends, time range)
   - The primary trends or shifts (e.g., rising CTA exposure, regional divergence)
   - Any inflection points, peaks, or unusual values

3. A **second paragraph (2–3 lines)** giving the broader context or implication of what’s shown, based on the chart and the surrounding page text.

Do **not** include entity-relationship lists or any triple backtick code blocks.

Use the format below exactly (no preamble):

| Column1 | Column2 | Column3 | ... |
|---------|---------|---------|-----|
| …       | …       | …       | …   |
...

Summary: <clear, rich multi-line description of the chart>

Explanation: <broader context or takeaway insight>
"""

PAGE_DELIM = "page_end"
INDENT = "    "

# --------------------------- Configuration -------------------------------
from pathlib import Path

# Configuration
PDF_DIR = Path(os.getenv("PDF_INPUT_DIR", "./data/pdfs"))
OUT_DIR = Path(os.getenv("CONVERTED_TEXT_DIR", "./data/converted"))
DEVICE = 'cpu'
DPI = 300
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
LLMW_KEY = os.getenv("LLMWHISPERER_API_KEY", "")
TIMEOUT = 300

# ----------------------- MinerU utilities --------------------------------

def load_layout_model(device: str):
    try:
        from mineru.backend.pipeline.model_init import AtomModelSingleton
        from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
        from mineru.utils.enum_class import ModelPath
    except ImportError:
        raise RuntimeError("Please install mineru: pip install git+https://github.com/mineru/mineru.git")

    atom_models = AtomModelSingleton()
    weight_root = Path(auto_download_and_get_model_root_path(ModelPath.doclayout_yolo))
    weights_path = weight_root / ModelPath.doclayout_yolo
    return atom_models.get_atom_model(
        atom_model_name="layout",
        doclayout_yolo_weights=str(weights_path),
        device=device,
    )


def is_figure_body(det: dict) -> bool:
    return int(det.get("category_id", -1)) == 3

# ----------------------- Core Pipeline -----------------------------------
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from PIL import Image

def detect_and_crop(pdf: Path, layout_model, dpi: int = DPI) -> List[Tuple[int, int, "Image"]]:
    from mineru.utils.pdf_image_tools import load_images_from_pdf
    pdf_bytes = pdf.read_bytes()
    images_list, _ = load_images_from_pdf(pdf_bytes, dpi=dpi)
    page_imgs = [d["img_pil"] for d in images_list]
    dets_per_page = layout_model.batch_predict(page_imgs, batch_size=4)
    crops = []
    for pidx, (pil, dets) in enumerate(zip(page_imgs, dets_per_page), 1):
        fidx = 0
        for det in dets:
            if not is_figure_body(det):
                continue
            fidx += 1
            x0, y0, x1, y1 = det["poly"][0], det["poly"][1], det["poly"][4], det["poly"][5]
            crops.append((pidx, fidx, pil.crop((x0, y0, x1, y1))))
    return crops

class GeminiStats(NamedTuple):
    prompt_tokens: int
    response_tokens: int
    total_tokens: int


def get_window_context(page_texts: Dict[int, str], current_page: int) -> str:
    """Get context from previous, current, and next page."""
    context_parts = []
    
    # Previous page
    if current_page - 1 in page_texts:
        context_parts.append(f"Previous page {current_page-1}:\n{page_texts[current_page-1]}\n")
    
    # Current page
    if current_page in page_texts:
        context_parts.append(f"Current page {current_page}:\n{page_texts[current_page]}\n")
    
    # Next page
    if current_page + 1 in page_texts:
        context_parts.append(f"Next page {current_page+1}:\n{page_texts[current_page+1]}\n")
    
    return "\n".join(context_parts)

def summarise_figures(crops, page_texts: Dict[int, str]) -> Tuple[Dict[int, List[str]], GeminiStats]:
    try:
        import google.generativeai as genai
    except ModuleNotFoundError:
        raise RuntimeError("Install google-generativeai")

    genai.configure(api_key=GEMINI_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL)

    summaries: Dict[int, List[str]] = {}
    p_tok = r_tok = t_tok = 0
    for idx, (page, fig, img) in enumerate(crops, 1):
        base = PROMPT_TEMPLATE.replace("{image_filename_placeholder}", f"page_{page}_fig_{fig}.jpg")
        window_context = get_window_context(page_texts, page)
        prompt = f"{base}\n\nContext (3-page window):\n{window_context}\n"
        resp = model.generate_content([prompt, img])
        summaries.setdefault(page, []).append(resp.text.strip())
        usage = getattr(resp, "usage_metadata", None)
        if usage:
            p_tok += getattr(usage, "prompt_token_count", 0)
            r_tok += getattr(usage, "candidates_token_count", 0)
            t_tok += getattr(usage, "total_tokens", 0)
            time.sleep(0.3)          # 200 calls/min
    return summaries, GeminiStats(p_tok, r_tok, t_tok)


def extract_text(pdf: Path) -> str:
    try:
        from unstract.llmwhisperer import LLMWhispererClientV2
    except ModuleNotFoundError:
        raise RuntimeError("Install llmwhisperer-client")

    client = LLMWhispererClientV2(api_key=LLMW_KEY)

    max_attempts = 4
    backoff = 2  # seconds (will double each retry)
    for attempt in range(1, max_attempts + 1):
        try:
            res = client.whisper(
                file_path=str(pdf),
                wait_for_completion=True,
                wait_timeout=TIMEOUT,
                mode="form",
                page_seperator=PAGE_DELIM,
            )
            return res.get("extraction", {}).get("result_text", "")
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout) as exc:
            if attempt == max_attempts:
                logger.error(f"LLMWhisperer failed after {attempt} attempts: {exc}")
                raise
            sleep_for = backoff * attempt
            logger.warning(f"LLMWhisperer connection error (attempt {attempt}/{max_attempts}). Retrying in {sleep_for}s…")
            time.sleep(sleep_for)


def page_text_map(llm_text: str) -> Dict[int, str]:
    lines = llm_text.splitlines()
    mp: Dict[int, str] = {}
    buf: List[str] = []
    page = 1
    for line in lines:
        if line.startswith(PAGE_DELIM):
            mp[page] = "\n".join(buf)
            buf = []
            page += 1
        else:
            buf.append(line)
    if buf:
        mp[page] = "\n".join(buf)
    return mp


def merge(llm_text: str, summaries_by_page: Dict[int, List[str]]) -> str:
    lines = llm_text.splitlines(keepends=True)
    out: List[str] = []
    buf: List[str] = []
    page = 1

    def flush(p):
        if buf:
            out.extend(buf)
        if summaries := summaries_by_page.get(p):
            block = "\n\n".join(textwrap.indent(s, INDENT) for s in summaries)
            out.append("\n" + block + "\n\n")
        buf.clear()

    for line in lines:
        if line.startswith(PAGE_DELIM):
            flush(page)
            out.append(line)
            page += 1
        else:
            buf.append(line)
    if buf:
        flush(page)
    return "".join(out)

# --------------------------- Driver ---------------------------------------

def process_one(pdf: Path, layout_model):
    start = time.time()
    llm_text = extract_text(pdf)
    texts = page_text_map(llm_text)
    crops = detect_and_crop(pdf, layout_model)
    summaries, stats = summarise_figures(crops, texts)
    final = merge(llm_text, summaries)
    out = OUT_DIR / f"{pdf.stem}_converted_with_figures.txt"
    out.write_text(final, encoding="utf-8")
    logger.success(
        f"{pdf.name} | prompt_tok={stats.prompt_tokens} resp_tok={stats.response_tokens} "
        f"total_tok={stats.total_tokens} time={time.time()-start:.1f}s → {out}"
    )


def main():
    pdf_dir = PDF_DIR.resolve()
    if not pdf_dir.is_dir():
        logger.error("PDF directory missing")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    layout_model = load_layout_model(DEVICE)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    for pdf in pdfs:
        process_one(pdf, layout_model)


if __name__ == "__main__":
    main() 