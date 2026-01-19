# Images and vision (OpenAI API docs)

Source: https://platform.openai.com/docs/guides/images-vision\
Snapshot date (UTC): 2026-01-16

## Overview

In this guide, you will learn about building applications involving images with
the OpenAI API.

### A tour of image-related use cases

Recent language models can process image inputs and analyze them — a capability
known as **vision**. With `gpt-image-1`, they can both analyze visual inputs and
create images.

The OpenAI API offers several endpoints to process images as input or generate
them as output, enabling you to build powerful multimodal applications.

**API — supported use cases**

- **Responses API**: Analyze images and use them as input and/or generate images
  as output
- **Images API**: Generate images as output, optionally using images as input
- **Chat Completions API**: Analyze images and use them as input to generate
  text or audio

## Generate or edit images

You can generate or edit images using the **Images API** or the **Responses
API**.

Our latest image generation model, `gpt-image-1`, is a natively multimodal large
language model. It can understand text and images and leverage its broad world
knowledge to generate images with better instruction following and contextual
awareness.

In contrast, we also offer specialized image generation models (DALL·E 2 and 3)
which don't have the same inherent understanding of the world as GPT Image.

### Generate images with Responses (example)

#### JavaScript (Node)

```js
import OpenAI from "openai";
const openai = new OpenAI();

const response = await openai.responses.create({
  model: "gpt-4.1-mini",
  input:
    "Generate an image of gray tabby cat hugging an otter with an orange scarf",
  tools: [{ type: "image_generation" }],
});

// Save the image to a file
const imageData = response.output
  .filter((output) => output.type === "image_generation_call")
  .map((output) => output.result);

if (imageData.length > 0) {
  const imageBase64 = imageData[0];
  const fs = await import("fs");
  fs.writeFileSync("cat_and_otter.png", Buffer.from(imageBase64, "base64"));
}
```

#### Python

```py
from openai import OpenAI
import base64

client = OpenAI()

response = client.responses.create(
    model="gpt-4.1-mini",
    input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
    tools=[{"type": "image_generation"}],
)

image_data = [
    output.result
    for output in response.output
    if output.type == "image_generation_call"
]

if image_data:
    image_base64 = image_data[0]
    with open("cat_and_otter.png", "wb") as f:
        f.write(base64.b64decode(image_base64))
```

### Using world knowledge for image generation

A natively multimodal language model can use its visual understanding of the
world to generate lifelike images including real-life details without a
reference.

Example given in the docs: if you prompt GPT Image to generate an image of a
glass cabinet with popular semi‑precious stones, the model can select and depict
gemstones like amethyst, rose quartz, jade, etc.

## Analyze images

**Vision** is the ability for a model to “see” and understand images. If there
is text in an image, the model can also understand the text. It can understand
most visual elements (objects, shapes, colors, textures), with some limitations.

### Giving a model images as input

You can provide images as input in multiple ways:

- Fully qualified **URL** to an image file
- **Base64-encoded data URL**
- **File ID** (created with the Files API)

You can provide multiple images as input in a single request by including
multiple images in the `content` array, but keep in mind that **images count as
tokens** and will be billed accordingly.

### Passing a URL (example)

```js
import OpenAI from "openai";
const openai = new OpenAI();

const response = await openai.responses.create({
  model: "gpt-4.1-mini",
  input: [{
    role: "user",
    content: [
      { type: "input_text", text: "what's in this image?" },
      {
        type: "input_image",
        image_url:
          "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
      },
    ],
  }],
});

console.log(response.output_text);
```

```py
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {
                "type": "input_image",
                "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
            },
        ],
    }],
)
print(response.output_text)
```

```bash
curl https://api.openai.com/v1/responses   -H "Content-Type: application/json"   -H "Authorization: Bearer $OPENAI_API_KEY"   -d '{
    "model": "gpt-4.1-mini",
    "input": [
      {
        "role": "user",
        "content": [
          {"type": "input_text", "text": "what is in this image?"},
          {
            "type": "input_image",
            "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
          }
        ]
      }
    ]
  }'
```

### Passing a Base64-encoded image (example)

```js
import fs from "fs";
import OpenAI from "openai";

const openai = new OpenAI();

const imagePath = "path_to_your_image.jpg";
const base64Image = fs.readFileSync(imagePath, "base64");

const response = await openai.responses.create({
  model: "gpt-4.1-mini",
  input: [
    {
      role: "user",
      content: [
        { type: "input_text", text: "what's in this image?" },
        {
          type: "input_image",
          image_url: `data:image/jpeg;base64,${base64Image}`,
        },
      ],
    },
  ],
});

console.log(response.output_text);
```

```py
import base64
from openai import OpenAI

client = OpenAI()

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "path_to_your_image.jpg"
base64_image = encode_image(image_path)

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what's in this image?"},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        }
    ],
)

print(response.output_text)
```

### Passing a file ID (example)

```py
from openai import OpenAI

client = OpenAI()

def create_file(file_path: str) -> str:
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
        return result.id

file_id = create_file("path_to_your_image.jpg")

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "what's in this image?"},
            {"type": "input_image", "file_id": file_id},
        ],
    }],
)

print(response.output_text)
```

### Image input requirements

Input images must meet the following requirements to be used in the API.

**Supported file types**

- PNG (.png)
- JPEG (.jpeg and .jpg)
- WEBP (.webp)
- Non-animated GIF (.gif)

**Size limits**

- Up to 50 MB total payload size per request
- Up to 500 individual image inputs per request

**Other requirements**

- No watermarks or logos
- No NSFW content
- Clear enough for a human to understand

### Specify image input detail level

The `detail` parameter tells the model what level of detail to use when
processing and understanding the image (`low`, `high`, or `auto`).

Example:

```json
{
  "type": "input_image",
  "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
  "detail": "high"
}
```

Using `"detail": "low"` can save tokens and speed up responses by processing a
low-resolution 512×512 version of the image with a budget of **85 tokens**.

## Limitations

Known limitations mentioned in the docs:

- Medical images: not suitable for interpreting specialized medical images
  (e.g., CT scans) and shouldn't be used for medical advice.
- Non-English: may not perform optimally for non‑Latin alphabets (e.g.,
  Japanese, Korean).
- Small text: enlarge text to improve readability, but avoid cropping important
  details.
- Rotation: may misinterpret rotated or upside-down text and images.
- Visual elements: may struggle with graphs or text where colors/styles
  (solid/dashed/dotted lines) vary.
- Spatial reasoning: struggles with precise spatial localization (e.g., chess
  positions).
- Accuracy: may generate incorrect descriptions/captions in certain scenarios.
- Image shape: struggles with panoramic and fisheye images.
- Metadata/resizing: doesn't process original filenames/metadata; images are
  resized before analysis.
- Counting: may give approximate object counts.
- CAPTCHAs: submissions are blocked for safety reasons.

## Calculating costs

Image inputs are metered and charged in tokens (like text). How images are
converted to tokens varies by model.

### GPT-4.1-mini, GPT-4.1-nano, o4-mini

Token cost is based on image dimensions:

A. Number of 32×32 patches needed to cover the image:

```
raw_patches = ceil(width/32) × ceil(height/32)
```

B. If patches exceed 1536, scale down so the image can be covered by ≤1536
patches:

```
r = √(32²×1536/(width×height))
r = r × min(
  floor(width×r/32) / (width×r/32),
  floor(height×r/32) / (height×r/32)
)
```

C. Token cost is patches (capped at 1536 tokens):

```
image_tokens = ceil(resized_width/32) × ceil(resized_height/32)
```

D. Apply a model multiplier:

- `gpt-5-mini` — 1.62
- `gpt-5-nano` — 2.46
- `gpt-4.1-mini` — 1.62
- `gpt-4.1-nano` — 2.46
- `o4-mini` — 1.72

**Cost examples**

- 1024×1024 image → 1024 tokens
- 1800×2400 image → 1452 tokens (after scaling to fit token budget)

### GPT-4o, GPT-4.1, GPT-4o-mini, CUA, and o-series (except o4-mini)

Token cost depends on **size** and **detail**:

- Any image with `"detail": "low"` costs a fixed base number of tokens (varies
  by model).
- For `"detail": "high"`:
  - Scale to fit within 2048×2048 (preserve aspect ratio)
  - Scale so shortest side is 768px
  - Count number of 512px tiles; each costs a fixed number of tokens
  - Add base tokens

Model token chart:

- gpt-5 / gpt-5-chat-latest — base 70, tile 140
- 4o / 4.1 / 4.5 — base 85, tile 170
- 4o-mini — base 2833, tile 5667
- o1 / o1-pro / o3 — base 75, tile 150
- computer-use-preview — base 65, tile 129

**Example (gpt-4o)**

- 1024×1024 in `"detail": "high"` → 765 tokens (`170*4 + 85`)
- 2048×4096 in `"detail": "high"` → 1105 tokens (`170*6 + 85`)
- 4096×8192 in `"detail": "low"` → 85 tokens

### GPT Image 1

Calculated similarly to the above, except the shortest side is scaled to
**512px** (instead of 768px). Pricing also depends on input fidelity.

- Low input fidelity: base 65 image tokens, each tile 129 image tokens.
- High input fidelity: adds extra tokens based on aspect ratio:
  - Square images: +4160 tokens
  - Portrait/landscape-like: +6240 tokens

### Notes

- Images count toward your tokens-per-minute (TPM) limit.
- For up-to-date estimates, use the image pricing calculator linked from the
  pricing page.

## Here’s a summary of known OpenAI defaults for image generation sizes and quality options across the main OpenAI models (based on available docs and community/third-party references):

⚠️ Official OpenAI docs do not currently publish a consolidated “supported
sizes/qualities per model” table, so the info below reflects documented API
constraints from multiple sources. ￼

⸻

📸 Supported Image Sizes & Qualities (per Model)

🧠 DALL·E 2 •	Supported sizes: •	256×256 •	512×512 •	1024×1024 •	Quality: •	Only
standard quality is supported (no HD option). •	Defaults: •	If unspecified, many
API docs/tools default to 1024×1024. ￼

⸻

🧠 DALL·E 3 •	Supported sizes: •	1024×1024 •	1792×1024 (landscape) •	1024×1792
(portrait) •	Quality options: •	standard (default) •	hd (higher quality, more
compute/cost) •	Defaults: •	1024×1024 & standard quality by default. ￼

⸻

🧠 GPT Image Models (e.g., gpt-image-1) •	Supported sizes (documented via
community/encyclopedic sources): •	1024×1024 •	1536×1024 (landscape) •	1024×1536
(portrait) •	Quality/other: •	Parameters like quality may vary with
implementation; standard/hd control is generally associated with DALL·E 3. ￼

⸻

📌 Notes & Behavior •	DALL·E 2 doesn’t support a quality parameter (only
standard). ￼ •	DALL·E 3 supports quality (standard vs. hd) and style options
(e.g., vivid, natural), but defaults are typically used if not specified. ￼
•	The “n” parameter (number of images) is limited on DALL·E 3 to 1. ￼

## OpenAI Image Generation – Known Defaults (sizes & qualities)

## DALL·E 2

- **Sizes:** 256×256, 512×512, 1024×1024
- **Quality:** standard only
- **Default Size:** 1024×1024 (if unspecified)

## DALL·E 3

- **Sizes:** 1024×1024, 1792×1024, 1024×1792
- **Quality:** standard (default), hd
- **Default:** 1024×1024 @ standard

## GPT Image (e.g., gpt-image-1)

- **Sizes:** 1024×1024, 1536×1024, 1024×1536
- **Quality:** varies by implementation (often standard)

## Here’s an expanded OpenAI Image Generation Defaults & Supported Sizes/Qualities (with API examples) in Markdown format, based on official docs and reliable sources:

⚠️ OpenAI does not publish an official “one table per model” in their main docs,
but we can reconstruct known behavior from API references and community
resources. ￼

⸻

### Supported Image Sizes & Quality by Model

🏷️ DALL·E 2 •	Supported sizes: 256x256, 512x512, 1024x1024 ￼ •	Quality: Only
standard quality (no hd parameter). ￼ •	Default: 1024x1024 if unspecified
(common de facto default). •	Notes: DALL·E 2 does not support a quality option.
￼

🏷️ DALL·E 3 •	Supported sizes: •	1024x1024 •	1792x1024 •	1024x1792 ￼ •	Quality
Options: •	standard (default) •	hd (higher detail) ￼ •	Defaults: •	1024x1024
•	quality: standard

🏷️ GPT-Image Models (e.g., gpt-image-1, gpt-image-1-mini, gpt-image-1.5)
•	Supported / typical sizes: •	1024x1024 (square) •	1536x1024 (landscape)
•	1024x1536 (portrait) ￼ •	Quality: Varies by implementation; quality parameters
(like low, medium, high, auto) are reported in community guides — but the effect
is mostly on compression/efficiency, not inherent resolution. ￼ •	Defaults:
•	size: 1024x1024 (common default) •	quality: auto (if supported)

⸻

### 📋 Summary (Compact Table in Markdown)

| Model           | Supported Sizes                 | Quality Options         | Common Default       |
| --------------- | ------------------------------- | ----------------------- | -------------------- |
| **DALL·E 2**    | 256x256, 512x512, 1024x1024     | standard only           | 1024x1024 @ standard |
| **DALL·E 3**    | 1024x1024, 1792x1024, 1024x1792 | standard, hd            | 1024x1024 @ standard |
| **GPT-Image-1** | 1024x1024, 1536x1024, 1024x1536 | (low/medium/high/auto)¹ | 1024x1024 @ auto     |

¹ Quality for GPT-Image models affects compression/efficiency, not inherent
pixel dimensions.
[oai_citation:8‡OpenAI Cookbook](https://cookbook.openai.com/examples/generate_images_with_gpt_image?utm_source=chatgpt.com)

⸻

💡 Example API Calls

DALL·E 2 (size only)

POST https://api.openai.com/v1/images/generations { "model": "dall-e-2",
"prompt": "A red sports car on the moon", "size": "1024x1024", "n": 1 }

DALL·E 3 (size + quality)

POST https://api.openai.com/v1/images/generations { "model": "dall-e-3",
"prompt": "A futuristic city skyline at sunset", "size": "1792x1024", "quality":
"hd", "n": 1 }

GPT-Image-1 (size + quality)

POST https://api.openai.com/v1/images/generations { "model": "gpt-image-1",
"prompt": "A photorealistic portrait of a sea turtle", "size": "1536x1024",
"quality": "high", "n": 1 }

⸻

📌 Notes & Guidance •	n parameter: Many guides note that for some models (like
DALL·E 3) only n=1 is allowed per call, while older models (like DALL·E 2) can
generate more. ￼ •	Quality semantics: •	For DALL·E 3, hd produces more detail
and consistency at higher cost relative to standard. ￼ •	For GPT-Image models,
quality controls often relate to compression/level of detail in encoding, not
changing the resolution itself. ￼ •	“auto” size: Some integrations support size:
"auto" (e.g., GPT-Image examples). ￼
