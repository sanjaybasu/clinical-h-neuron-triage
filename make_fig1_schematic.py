#!/usr/bin/env python3
"""
Generate Figure 1 (CETT pipeline schematic) using Gemini image generation API.
Replaces the previous version which had overlapping text/graphics.

Three-stage pipeline diagram for Nature Medicine:
  Stage 1 – Contrasting query generation (TriviaQA pairs)
  Stage 2 – CETT sparse probing → h-neuron identification
  Stage 3 – Inference-time suppression (ablation sweep)
"""

import os, sys, json, base64, io, textwrap
from pathlib import Path
from PIL import Image

API_KEY = "REDACTED_API_KEY"

OUTPUT_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/h_neuron_triage/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PKG_DIR   = Path("/Users/sanjaybasu/waymark-local/packaging/h_neuron_triage/figures")
PKG_DIR.mkdir(parents=True, exist_ok=True)

# ── Review log ──────────────────────────────────────────────────────────────
REVIEW_LOG = []

def save_review_log():
    log_path = OUTPUT_DIR / "fig1_cett_pipeline_review_log.json"
    with open(log_path, "w") as f:
        json.dump(REVIEW_LOG, f, indent=2)
    print(f"Review log: {log_path}")


# ── Gemini image generation ──────────────────────────────────────────────────
def generate_image_gemini(prompt: str, iteration: int) -> Image.Image | None:
    """Call Gemini 2.0 Flash image generation."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("Run: pip install google-genai")

    client = genai.Client(api_key=API_KEY)

    print(f"\n[Iteration {iteration}] Calling Gemini image generation...")
    resp = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE"]
        ),
    )

    for part in resp.candidates[0].content.parts:
        if part.inline_data is not None:
            data = base64.b64decode(part.inline_data.data) \
                   if isinstance(part.inline_data.data, str) \
                   else part.inline_data.data
            return Image.open(io.BytesIO(data))
    return None


# ── Gemini quality review ────────────────────────────────────────────────────
def review_image_gemini(image: Image.Image, iteration: int) -> tuple[float, str, bool]:
    """Review image quality with Gemini 2.0 Flash. Returns (score, critique, needs_improvement)."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("Run: pip install google-genai")

    client = genai.Client(api_key=API_KEY)

    # Convert PIL image to bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_bytes = buf.getvalue()

    review_prompt = """You are a scientific figure quality reviewer for Nature Medicine.

Evaluate this scientific pipeline diagram on these 5 criteria (0-2 points each, max 10):

1. NO OVERLAPS (most critical): Are any text labels overlapping boxes, arrows, or other text?
   - 0: Multiple overlaps making it unreadable
   - 1: Minor overlaps but mostly readable
   - 2: Zero overlaps, all elements cleanly separated

2. CLARITY: Is the three-stage pipeline flow clearly communicated?
   - 0: Hard to follow
   - 1: Partially clear
   - 2: Very clear left-to-right flow with numbered stages

3. LABEL QUALITY: Are all key elements labeled with readable text?
   - 0: Missing labels or unreadable
   - 1: Some labels missing or small
   - 2: All elements clearly labeled, good font size

4. VISUAL HIERARCHY: Is there clear visual distinction between stages and elements?
   - 0: No hierarchy
   - 1: Some hierarchy
   - 2: Strong hierarchy with color-coding and spacing

5. PUBLICATION QUALITY: Would this be accepted in Nature Medicine?
   - 0: Looks draft/amateur
   - 1: Acceptable but needs polish
   - 2: Publication-ready professional quality

Respond in exactly this format:
SCORE: X.X

STRENGTHS:
- [strength 1]
- [strength 2]

ISSUES:
- [issue 1 with specific location]
- [issue 2 with specific location]

VERDICT: [PASS if score >= 8.0, else NEEDS_IMPROVEMENT]"""

    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
            types.Part.from_text(text=review_prompt)
        ]
    )

    critique = resp.text.strip()
    # Parse score
    score = 7.0
    for line in critique.split("\n"):
        if line.startswith("SCORE:"):
            try:
                score = float(line.split(":")[1].strip())
            except ValueError:
                pass
            break

    threshold = 8.0  # Nature Medicine standard
    needs_improvement = score < threshold

    print(f"  Review score: {score}/10 ({'PASS' if not needs_improvement else 'NEEDS IMPROVEMENT'})")

    REVIEW_LOG.append({
        "iteration": iteration,
        "score": score,
        "needs_improvement": needs_improvement,
        "critique": critique
    })

    return score, critique, needs_improvement


# ── Prompt construction ──────────────────────────────────────────────────────
BASE_PROMPT = """
Create a clean, professional scientific figure for a Nature Medicine manuscript.
This is a horizontal three-stage pipeline diagram showing CETT (Conditional Empirical Token-Level
Transparency) sparse probing to identify over-compliance neurons in language models.

CRITICAL LAYOUT RULES:
- Do NOT include any pixel measurements or spacing dimensions as text in the figure
- All text must be actual scientific content only — no layout annotations, no "px" values, no arrows described as pixel widths
- All elements must have generous whitespace between them — nothing touching anything else
- Each stage is a distinct vertical column; the three columns do not overlap each other
- Arrows go BETWEEN stages only, never over text
- Font: stage titles 16pt bold, content labels 11pt, key numbers 13pt bold
- Figure is wide landscape (about 2:1 width:height), white background, clean sans-serif font
- Color palette: Stage 1 column header = steel blue, Stage 2 column header = forest green, Stage 3 column header = burnt orange
- Flat design: no shadows, no gradients, no 3D effects

STAGE 1 (left column, blue header):
  Column header band (blue): "Stage 1: Contrasting Query Pairs" (bold title) + "TriviaQA dataset" (subtitle below)
  Below the header, two content boxes stacked vertically with generous space between:
    Top box (solid blue outline): "Factual question + correct answer"
    Bottom box (dashed blue outline, lighter fill): "Same question + hallucinated answer"
  Footer text at very bottom of column (small, gray): "1,024 contrasting pairs"

STAGE 2 (center column, green header):
  Column header band (green): "Stage 2: CETT Sparse Probing" (bold) + "32-layer Llama-3.1-8B" (subtitle)
  Below the header, four content boxes connected by downward arrows, each box clearly separated:
    Box 1 (green outline): "Forward pass → SwiGLU activations"
    Downward arrow
    Box 2 (green outline): "Logistic probe per neuron"
    Downward arrow
    Box 3 (dark green filled, white text bold): "AUC threshold = 0.847"
    Downward arrow
    Box 4 (orange filled, bold): "213 h-neurons identified" with a second line of smaller text: "Layers 10–14  ·  0.046% of neurons"
  Footer text at very bottom (small, gray, italic): "Jaccard overlap = 0.000 (base vs. fine-tuned)"

STAGE 3 (right column, orange header):
  Column header band (orange): "Stage 3: Inference-Time Suppression" (bold) + "Ablation sweep" (subtitle)
  Below the header, three sub-rows showing suppression effect. Each row has two parts side by side:
    LEFT part of row (narrow, white background): Greek alpha label with value
    RIGHT part of row (wider): a horizontal bar chart

    Row A: Left label "α = 0.0", bar chart shows a very short bar labeled "5%" with title "Suppress" above it
    Row B: Left label "α = 1.0", bar chart shows a medium bar labeled "41%" with title "Baseline" above it
    Row C: Left label "α = 3.0", bar chart shows a longer bar labeled "65%" with title "Amplify" above it

    The three rows are evenly spaced with clear white gaps between them. The alpha labels are strictly in their own left column; bars are strictly in the right column. No overlap.

  Footer text at very bottom of column (small, gray): "Physician test set, N = 132"

CONNECTING ARROWS between stages:
  Arrow 1: Thick horizontal arrow pointing right from Stage 1 to Stage 2. Label centered above the arrow: "Apply to model"
  Arrow 2: Thick horizontal arrow pointing right from Stage 2 to Stage 3. Label centered above the arrow: "Identify neurons"

The arrows must occupy the white space between the column borders — not overlapping any column content.

ABSOLUTE PROHIBITION:
- Do not write any pixel dimensions (px, pt) in the figure
- Do not write any layout instructions as text
- Only scientific content labels should appear

Produce a publication-ready scientific diagram with perfect whitespace and zero overlapping elements.
""".strip()


def make_improvement_prompt(critique: str, iteration: int) -> str:
    """Enhance prompt based on review critique."""
    issues = []
    in_issues = False
    for line in critique.split("\n"):
        if line.strip().startswith("ISSUES:"):
            in_issues = True
            continue
        if line.strip().startswith("VERDICT:"):
            in_issues = False
        if in_issues and line.strip().startswith("-"):
            issues.append(line.strip("- ").strip())

    fix_instruction = "\n".join(f"MUST FIX: {i}" for i in issues[:5])
    return f"""{BASE_PROMPT}

SPECIFIC CORRECTIONS REQUIRED (from quality review #{iteration}):
{fix_instruction}

Strict enforcement: No element may overlap any other element. Use generous whitespace.
"""


# ── Main pipeline ────────────────────────────────────────────────────────────
def run():
    MAX_ITERATIONS = 2
    prompt = BASE_PROMPT
    best_image = None
    best_score = 0.0

    for iteration in range(1, MAX_ITERATIONS + 1):
        image = generate_image_gemini(prompt, iteration)
        if image is None:
            print(f"  Iteration {iteration}: No image returned, skipping.")
            continue

        # Save intermediate
        tmp_path = OUTPUT_DIR / f"fig1_v{iteration}.png"
        image.save(tmp_path, dpi=(300, 300))
        print(f"  Saved: {tmp_path}")

        score, critique, needs_improvement = review_image_gemini(image, iteration)

        if score > best_score:
            best_score = score
            best_image = image

        if not needs_improvement:
            print(f"\n  Quality threshold met at iteration {iteration} (score={score}).")
            break

        if iteration < MAX_ITERATIONS:
            print(f"  Below threshold ({score} < 8.0), refining prompt for iteration {iteration+1}...")
            prompt = make_improvement_prompt(critique, iteration)

    # Save final
    if best_image is not None:
        for out_dir in [OUTPUT_DIR, PKG_DIR]:
            png_path = out_dir / "fig_cett_pipeline_schematic.png"
            best_image.save(png_path, dpi=(300, 300))
            print(f"Saved: {png_path}")

        REVIEW_LOG.append({"final_score": best_score, "early_stop": best_score >= 8.0})
        save_review_log()
        print(f"\nFinal score: {best_score}/10")
    else:
        print("ERROR: No image generated.")
        sys.exit(1)


if __name__ == "__main__":
    os.chdir("/Users/sanjaybasu/waymark-local/packaging/h_neuron_triage")
    run()
