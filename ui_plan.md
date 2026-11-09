# CVA --- Visual Reliability Lab

## Product Requirements, UI/UX Specification, Technical Architecture & Antigravity Build Instructions

**Purpose:** Build a polished, public Computer Vision product that lets
visitors inspect how image-classification models respond to controlled
visual interventions.\
**Build window:** 2--4 weeks.\
**Frontend:** React + TypeScript + Tailwind, deployed on Vercel.\
**Backend:** FastAPI deployed separately from Vercel.\
**ML:** PyTorch + torchvision.\
**Development hardware:** Windows, RTX 3050 Laptop GPU (6 GB VRAM),
Ryzen 5 7640HS.\
**Audience:** Computer vision students, ML practitioners, researchers,
and a professor evaluating the project.

------------------------------------------------------------------------

# 1. Product vision

CVA (Visual Reliability Lab) is an interactive visual model-auditing
platform. A user selects a supported image-classification model, uploads
an image, and runs controlled experiments to inspect how the model's
predictions change when foreground/background information is modified.

**Product promise:** Understand what your vision model is relying
on---not just what it predicts.

The product should feel like a polished research tool, not a generic
admin dashboard or a thin upload form.

A first-time visitor must be able to try a built-in demo without
registering or uploading a model. Advanced users should be able to
upload a compatible custom model through a constrained, secure workflow.

## Core capabilities

-   Original image classification and top-k predictions.
-   Foreground mask visualization.
-   Background replacement/blur intervention.
-   Foreground suppression intervention.
-   Side-by-side prediction comparison.
-   Supported custom model upload.
-   Benchmark dashboard backed by actual experiment results.
-   Methodology and limitations pages.
-   Downloadable audit report.

## Scientific positioning

CVA is an auditing and exploratory analysis tool. It does not prove
causal reliance from one image intervention. Masking/compositing can
introduce artifacts, segmentation can be imperfect, and prediction
changes can have multiple explanations. Explain these limitations in the
UI and every report.

Do not market CVA as definitively explaining a model's internal
reasoning.

------------------------------------------------------------------------

# 2. Product scope

## Required MVP

1.  Public, responsive website deployed on Vercel.
2.  Polished landing page and Studio.
3.  Built-in, real, trained demo checkpoint.
4.  Image upload and validation.
5.  Actual PyTorch inference through FastAPI.
6.  At least one reliable intervention; target two.
7.  Before/after image and prediction comparison.
8.  Benchmark page populated from actual reproducible experiment
    outputs.
9.  Methodology and limitations page.
10. JSON report export; PDF/printable report if time permits.
11. Deployment and local setup documentation.

## Stretch goals

-   Custom model upload for a strict architecture allowlist.
-   Multiple model architectures.
-   PDF reports.
-   Audit history for the current browser session.
-   Streamlit is NOT needed; the public React website is the product.
-   Extra benchmark datasets and ablations.

## Explicitly out of scope

-   Arbitrary user-supplied Python code or model architectures.
-   Arbitrary pickled PyTorch objects.
-   User accounts, billing, social features, or teams.
-   Training custom models in the browser.
-   Real-time video.
-   Large-scale distributed inference.
-   Claims of state-of-the-art or causal explanation.
-   Complex database/microservice architecture before the core workflow
    works.

**Priority order:** real built-in demo → image upload → inference →
interventions → comparison → benchmark → custom model upload → polish.

------------------------------------------------------------------------

# 3. Target user journeys

## A. First-time visitor --- most important

1.  Opens landing page.
2.  Understands CVA within 10 seconds.
3.  Clicks **Try Demo**.
4.  Selects built-in demo model.
5.  Uploads an image or chooses a sample.
6.  Sees actual model prediction.
7.  Runs background/foreground intervention.
8.  Inspects transformed image and prediction changes.
9.  Reads limitations.
10. Downloads an audit report or opens benchmark results.

This flow must work without registration and without model upload.

## B. Custom model user

1.  Opens Custom Audit.
2.  Reads supported architecture and format requirements.
3.  Selects an allowlisted architecture.
4.  Uploads weights, config, and class labels.
5.  Backend validates the artifact.
6.  User uploads an image.
7.  Runs inference and supported interventions.
8.  Exports results.

Never silently infer architecture, label order, preprocessing, or
normalization.

## C. Professor/evaluator

1.  Opens live link.
2.  Tries demo without setup.
3.  Inspects intervention process.
4.  Opens benchmark page and sees real metrics and dataset group
    definitions.
5.  Reads methodology and limitations.
6.  Downloads an example report.
7.  Optionally inspects GitHub source.

No fake metrics, fake users, fake testimonials, fake affiliations, or
fabricated endorsements.

------------------------------------------------------------------------

# 4. Information architecture

Implement these routes:

  Route              Purpose
  ------------------ ----------------------------------------
  `/`                Landing page
  `/studio`          Main audit workspace and built-in demo
  `/studio/custom`   Custom model upload and validation
  `/benchmarks`      Real benchmark results
  `/methodology`     Interventions, metrics, limitations
  `/about`           Project, technology, source repository

Global navigation: - CVA logo/wordmark. - Studio, Benchmarks,
Methodology. - Primary CTA: Launch Studio. - Backend/API availability
indicator. - Configurable GitHub link. - Responsive mobile navigation.

Every visible control must work. Do not show dead buttons.

------------------------------------------------------------------------

# 5. Visual design system

## Design concept: premium dark research workstation

The UI should feel precise, technical, modern, restrained, and visually
distinctive. Do not copy a generic shadcn dashboard or default Tailwind
template.

### Palette

Use CSS variables.

-   App background: `#0A0D14`
-   Primary panel: `#111827`
-   Secondary panel: `#151D2B`
-   Elevated panel: `#1B2638`
-   Main text: `#F3F6FC`
-   Secondary text: `#A7B1C2`
-   Muted text: `#718096`
-   Border: `rgba(148, 163, 184, 0.16)`
-   Accent: `#818CF8`
-   Accent hover: `#6366F1`
-   Success: `#34D399`
-   Warning: `#FBBF24`
-   Error: `#FB7185`

Use color semantically; never rely on color alone for status.

### Style

-   Modern sans-serif: Geist, Inter, or Manrope.
-   Monospace only for model IDs and technical metadata.
-   Strong typographic hierarchy and generous spacing.
-   Fine borders, subtle shadows, restrained gradients.
-   Accent colors only for actions/active states.
-   Avoid neon everywhere, excessive glassmorphism, glowing borders, and
    visual clutter.
-   Subtle transitions and skeleton loading states.
-   Respect `prefers-reduced-motion`.

### Responsive targets

Test at 1440, 1280, 1024, 768, 390, and 360px. Do not merely shrink the
desktop layout. Stack image comparisons and controls appropriately on
mobile.

------------------------------------------------------------------------

# 6. Page requirements

## 6.1 Landing page (`/`)

### Hero copy

Headline: **See what your vision model actually sees.**

Supporting copy: "Audit image classifiers with controlled visual
interventions. Compare predictions, inspect foreground and background
sensitivity, and explore model reliability."

Primary CTA: **Launch Studio**\
Secondary CTA: **Explore Methodology**

Hero visual: - Original image. - Foreground mask. - Counterfactual
result. - Use real sample/intervention outputs once implemented. -
During early UI development, static placeholders may be used but must be
replaced before production. - Do not invent accuracy percentages or
model improvements.

### Sections

1.  Hero with product preview.
2.  How it works: Select model/image → Apply intervention → Inspect
    prediction sensitivity.
3.  Feature cards: Counterfactual workspace, custom models, benchmarks,
    reports.
4.  Methodology/trust section explaining limitations.
5.  Footer with Studio, Benchmarks, Methodology, GitHub, attribution,
    and disclaimer.

## 6.2 Studio (`/studio`) --- highest priority

Desktop layout: - Left: model and image setup. - Center: image
comparison workspace. - Right: predictions, metrics, intervention
controls. - Panels can collapse; image workspace remains dominant.

### Model selector

**Demo model** - Select project-trained, validated model. - Show
architecture, class count, checkpoint version, input size,
preprocessing, and readiness. - Show training dataset and limitations.

**Custom model** - Link to guided custom upload. - Display accepted
architectures/formats. - Never promise "upload any model."

### Image input

-   Drag/drop and browse.
-   JPEG, PNG, WEBP.
-   Configurable max size (start at 10 MB).
-   Show preview and dimensions.
-   Replace/remove image.
-   Provide a few properly attributed sample images.
-   Validate actual file content, not just extension.
-   Reject corrupt, unsupported, and oversized images with useful
    messages.

### Original prediction panel

Show: - Predicted class. - Confidence/softmax score. - Top-k predictions
(3--5). - Model architecture and preprocessing metadata. - Inference
duration only if accurately measured. - Explain that softmax confidence
is not necessarily calibrated probability of correctness.

### Intervention controls

1.  **Background replacement/blur:** Preserve foreground mask, modify
    background.
2.  **Foreground suppression:** Mask or blur foreground, preserve
    background.
3.  **Mask preview:** Show foreground mask overlay and allow inspection.
4.  Expose only settings actually supported by the backend.

Explain each intervention with a tooltip.

### Comparison results

For each intervention show: - Original image. - Modified image. -
Original and modified top prediction. - Original-predicted-class score
change. - True-class score change only when ground truth is supplied. -
Prediction-flip indicator. - Optional Jensen--Shannon divergence if
implemented correctly. - Mask/segmentation warning where relevant.

Use clear comparison cards or a before/after slider that works on touch
screens.

### Audit summary

Generate deterministic text from actual measured results. Example:

"The predicted class remained unchanged after background blur, but its
score decreased by 0.18. The prediction changed after foreground
masking. This indicates sensitivity to these interventions; it does not
establish causal feature reliance."

Do not use an LLM to invent explanations.

### Export

-   JSON audit record required.
-   PDF or printable report if time permits.
-   Include model metadata, image metadata, intervention configuration,
    predictions, metrics, timestamp, and caveats.
-   Never expose private uploads through public URLs.

### Required states

Initial, uploading, validating, inference, segmentation, intervention,
completed, partial failure, API unavailable, unsupported model, invalid
image, rate limited, session expired.

No infinite spinner. Explain progress and errors.

## 6.3 Custom model (`/studio/custom`)

### Supported format strategy

Never accept arbitrary `.py` model definitions or arbitrary pickled
model objects.

Use a constrained package: - Architecture selected from a server-side
allowlist. - `.safetensors` preferred for weights. - `.pth`/`.pt` only
if the backend safely loads a plain state dict with restricted loading
(`weights_only=True`) and validates all tensors. - JSON model
configuration. - JSON class labels.

Start with ResNet-18 only. Add more architectures after the full flow
works.

Potential allowlist: - ResNet-18 - ResNet-50 (optional) -
MobileNetV3-Large - EfficientNet-B0 (optional)

### Example package

``` text
my-model/
├── weights.safetensors
├── model_config.json
└── labels.json
```

`model_config.json`:

``` json
{
  "architecture": "resnet18",
  "num_classes": 2,
  "input_size": [224, 224],
  "color_mode": "RGB",
  "normalization": {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "resize_mode": "resize"
}
```

`labels.json`:

``` json
{
  "0": "class_a",
  "1": "class_b"
}
```

Explain that class order must exactly match training order.

### Validation

Validate: - File size and content. - JSON schema. - Architecture
allowlist. - Class count and labels. - Tensor names/shapes. -
Missing/unexpected state-dict keys. - Input dimensions and
normalization. - Memory requirements. - Checkpoint compatibility.

Never use unrestricted `torch.load()` on an untrusted upload. Do not
execute user-supplied Python. Treat every upload as untrusted.

Custom model uploads are temporary by default and deleted after a
configurable TTL. If safe validation cannot be implemented in time,
disable custom uploads in the public MVP rather than weakening security.

## 6.4 Benchmarks (`/benchmarks`)

Display only actual saved experiment outputs: - Dataset/version/split. -
Sample counts and group definitions. - Model architectures. - Overall
accuracy. - Worst-group accuracy. - Accuracy per group. - Intervention
prediction-flip rates. - Score-change distributions. - Experiment
configuration/date. - Download metrics as CSV/JSON.

Charts need axis labels, units, legends, and clear titles. If results
are unavailable, show "Benchmark results are being prepared", not fake
numbers.

## 6.5 Methodology (`/methodology`)

Explain: - Shortcut learning and background correlation. -
Foreground/background interventions. - Segmentation masks. - Prediction
sensitivity. - Worst-group accuracy. - Distribution shift. - Confounders
and limitations.

Flow: Input image → Model prediction → Intervention → Re-inference →
Comparison.

Add verifiable papers/dataset references.

## 6.6 About (`/about`)

Explain project motivation, technology, source link, and research
disclaimer. Do not invent university/lab affiliations, awards,
endorsements, or personal details.

------------------------------------------------------------------------

# 7. Frontend technical requirements

## Recommended stack

-   React + TypeScript + Vite.
-   Tailwind CSS.
-   shadcn/ui selectively, customized to CVA's design system.
-   Lucide icons.
-   TanStack Query for server state.
-   React Hook Form + Zod where useful.
-   Recharts or Plotly for charts.
-   Framer Motion/Motion for restrained transitions.
-   React Router for SPA routes.

Do not install redundant libraries.

## Standards

-   Strict TypeScript; avoid `any`.
-   Reusable components.
-   Central API client.
-   Environment-based API URL.
-   Explicit loading/error/empty states.
-   Upload progress where supported.
-   Accessibility: labels, keyboard navigation, focus states, alt text.
-   No API secrets in browser bundles.
-   No fake predictions or hardcoded benchmark values.
-   No direct browser-to-model-provider calls.
-   Revoke image object URLs when no longer needed.

Suggested structure:

``` text
frontend/
├── public/samples/
├── src/
│   ├── app/
│   ├── components/
│   │   ├── layout/
│   │   ├── navigation/
│   │   ├── upload/
│   │   ├── image-viewer/
│   │   ├── charts/
│   │   ├── audit/
│   │   └── ui/
│   ├── pages/
│   ├── hooks/
│   ├── lib/
│   ├── types/
│   └── styles/
├── .env.example
├── package.json
└── README.md
```

------------------------------------------------------------------------

# 8. Backend and ML architecture

## Important deployment decision

**Vercel hosts the frontend. FastAPI/PyTorch inference should be
deployed separately.**

Vercel supports Python Functions and FastAPI, but function bundle size,
memory, execution duration, and lack of dedicated CUDA GPU make it a
poor default for this ML inference service. Do not put a large PyTorch
model and GPU workload into Vercel just because Python is supported.

Architecture:

``` text
Browser
  |
  v
Vercel: React + TypeScript frontend
  |
  | HTTPS API
  v
Separate Python host: FastAPI
  |
  +---- PyTorch model registry / inference
  |
  +---- segmentation and interventions
  |
  +---- temporary uploads / audit artifacts
  |
  +---- benchmark JSON/CSV results
```

Investigate current free-tier capabilities before selecting a backend
host. Candidates to evaluate include Hugging Face Spaces, Render, and
Railway. Verify sleeping, CPU/RAM, disk persistence, request limits, and
pricing. Do not promise permanent free hosting. CPU inference is
acceptable for a small demo if latency is documented. Do not expose the
developer's local RTX 3050 as the production server.

## Backend stack

-   Python 3.10 or 3.11 based on dependency compatibility.
-   FastAPI, Uvicorn, Pydantic.
-   PyTorch, torchvision.
-   Pillow/OpenCV, NumPy.
-   safetensors.
-   python-multipart.
-   pytest.

Suggested structure:

``` text
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── health.py
│   │   ├── models.py
│   │   ├── uploads.py
│   │   ├── audits.py
│   │   └── benchmarks.py
│   ├── core/
│   ├── ml/
│   │   ├── model_registry.py
│   │   ├── model_loader.py
│   │   ├── preprocessing.py
│   │   ├── inference.py
│   │   ├── segmentation.py
│   │   ├── interventions.py
│   │   └── metrics.py
│   ├── services/
│   └── schemas/
├── checkpoints/
├── tests/
├── requirements.txt
├── Dockerfile
└── README.md
```

Keep routes, validation, model loading, preprocessing, inference,
interventions, metrics, and report generation separate.

## Built-in demo model

Include at least one real, trained, compatible checkpoint. Preferred
initial model: ResNet-18.

Document: - Actual training dataset. - Class mapping. - Model version. -
Input size and normalization. - Evaluation split and metrics.

If a Waterbirds model is not ready, do not label an arbitrary ImageNet
model as Waterbirds. Use an honestly labeled ImageNet demo with its real
class labels, or finish training the intended benchmark model.

## Image preprocessing

-   Decode and validate input.
-   Convert to RGB.
-   Apply exact model-specific resize/crop.
-   Apply exact normalization.
-   Preserve original image for display.
-   Record preprocessing config.

Never assume all models use ImageNet normalization.

## Segmentation/interventions

For MVP: - Prefer dataset masks where available. - Otherwise use a
pretrained segmentation model that fits host resources. - Cache masks
during an audit. - Display mask to user. - Implement
foreground-preserving background blur/replacement and foreground
suppression. - Optional matched background swap only if reliable masks
and sample backgrounds exist. - Document that composites can be
out-of-distribution.

------------------------------------------------------------------------

# 9. API contract

Use `/api/v1`.

### `GET /api/v1/health`

Return API status, version, inference device, demo-model readiness, and
optional cold-start status. Do not expose filesystem paths or secrets.

### `GET /api/v1/models`

Return actual available demo models and supported custom architectures.
Never report a model as ready unless its checkpoint exists and loads.

### `POST /api/v1/models/validate`

Accept a supported model package and return validation results. Do not
load arbitrary model code.

### `POST /api/v1/audits`

Accept image, built-in model ID or validated custom model ID, optional
ground truth, and intervention options. Return a job ID or completed
result depending on execution duration.

### `GET /api/v1/audits/{audit_id}`

Return job status and result.

### `GET /api/v1/benchmarks`

Return benchmark metadata and actual saved results.

### `GET /api/v1/benchmarks/{benchmark_id}`

Return detailed metrics and group definitions.

### `GET /api/v1/reports/{audit_id}`

Return JSON report or PDF if implemented.

### `DELETE /api/v1/uploads/{upload_id}`

Delete temporary upload if supported.

### API requirements

-   Pydantic validation.
-   Consistent error response format.
-   Correct HTTP status codes.
-   Upload size limits.
-   CORS restricted to frontend origin.
-   Rate limiting/conservative per-IP limits.
-   No traceback in browser responses.
-   Avoid storing large blobs in global in-memory dictionaries.
-   Use job/status polling for longer operations.
-   Opaque unguessable audit IDs.
-   No cross-user artifact access.

------------------------------------------------------------------------

# 10. Audit result schema

Example shape (values must be real):

``` json
{
  "audit_id": "opaque-id",
  "created_at": "ISO-8601",
  "model": {
    "id": "waterbirds-resnet18",
    "architecture": "resnet18",
    "num_classes": 2,
    "version": "checkpoint-version"
  },
  "input": {
    "filename": "sample.jpg",
    "width": 640,
    "height": 480
  },
  "original_prediction": {
    "class_index": 1,
    "label": "waterbird",
    "score": 0.91,
    "top_k": []
  },
  "interventions": [
    {
      "type": "background_blur",
      "prediction": {
        "class_index": 1,
        "label": "waterbird",
        "score": 0.73
      },
      "original_predicted_class_score_change": -0.18,
      "prediction_changed": false,
      "artifact_warning": null
    }
  ],
  "limitations": [
    "Intervention artifacts may affect predictions.",
    "Prediction sensitivity is not proof of causal feature reliance."
  ]
}
```

Define score changes consistently (absolute difference vs relative
change). Do not mix definitions.

------------------------------------------------------------------------

# 11. Security and privacy

Custom uploads are a security boundary.

Required: - Strict file-size and content validation. - Allowlisted
architectures. - No user-supplied Python execution. - No arbitrary
pickle loading. - Prefer safetensors. - If `.pth` is supported, use
restricted state-dict loading and validate every tensor. - Temporary
private storage and automatic TTL cleanup. - Per-IP/session rate
limits. - Limit image dimensions and decoded pixel count. - Randomized
temporary filenames; never trust user paths. - No secrets in client
bundles/logs. - Restrictive CORS. - Do not log image bytes or weights. -
No public upload buckets. - HTTPS. - Clear privacy copy.

If secure custom loading cannot be delivered in time, disable custom
upload in public MVP. Built-in demo is mandatory; custom uploads are not
worth compromising security.

------------------------------------------------------------------------

# 12. Storage

Do not build accounts/database for the MVP.

Use: - Browser state for active workspace. - Backend temporary storage
for uploaded files. - JSON/CSV for benchmark results. - Configurable
cleanup TTL.

Do not assume serverless local disk is persistent. If saved audit
history is added later, implement explicit session ownership and
persistent storage.

------------------------------------------------------------------------

# 13. Vercel deployment

## Frontend deployment

-   Connect frontend repository to Vercel.
-   Set correct root directory for monorepo.
-   Configure build command/output.
-   Set `VITE_API_BASE_URL` to deployed FastAPI URL.
-   Configure preview/production env separately.
-   Use HTTPS.
-   Configure SPA rewrites if needed.
-   Test direct refresh/navigation on `/studio`, `/benchmarks`,
    `/methodology`.

## Backend deployment

Deploy FastAPI to a separate Python/container host. Set environment
variables, CORS, upload limits, health checks, demo checkpoint
availability, and cold-start handling. Test actual inference from public
frontend.

## Free tier caveat

Vercel Hobby is free for personal/non-commercial use subject to current
fair-use and usage limits. It is appropriate for the frontend of this
student project, not a guarantee that all ML inference can run free
indefinitely.

Check current official docs before deployment: -
https://vercel.com/docs/plans/hobby - https://vercel.com/docs/limits -
https://vercel.com/docs/functions/limitations -
https://vercel.com/docs/frameworks/backend/fastapi

Do not embed large PyTorch weights into the frontend or rely on Vercel
serverless functions for GPU inference.

## Environment variables

Frontend `.env.example`:

``` bash
VITE_API_BASE_URL=http://localhost:8000
VITE_GITHUB_URL=
```

Backend `.env.example`:

``` bash
APP_ENV=development
CORS_ORIGINS=http://localhost:5173
MAX_IMAGE_UPLOAD_MB=10
MAX_MODEL_UPLOAD_MB=100
TEMP_UPLOAD_TTL_MINUTES=60
RATE_LIMIT_PER_MINUTE=10
MODEL_DEVICE=auto
```

Adjust limits to actual host capacity. Never commit real secrets.

------------------------------------------------------------------------

# 14. Performance and reliability

-   Lazy-load heavy routes/charts.
-   Resize analysis copy where appropriate; preserve original for
    display.
-   Inference runs on backend.
-   Cache built-in model in process when possible.
-   Do not reload model checkpoint for every request.
-   Communicate cold starts.
-   Use job ID/status polling if necessary.
-   Add timeouts and retry only safe operations.
-   Prevent duplicate submissions.
-   Handle segmentation failure gracefully if classification succeeded.
-   Document latency and host limits.
-   Never claim real-time performance without measurement.

------------------------------------------------------------------------

# 15. Testing and acceptance criteria

## Frontend

-   [ ] Polished landing page at desktop/mobile.
-   [ ] All routes work with direct refresh.
-   [ ] Studio supports upload and sample images.
-   [ ] Loading, empty, success, and error states.
-   [ ] Model selection comes from API.
-   [ ] Predictions are real API results.
-   [ ] Intervention controls trigger actual backend work.
-   [ ] Displayed images match returned artifacts.
-   [ ] No fake benchmark metrics.
-   [ ] Charts have labels/legends.
-   [ ] Keyboard navigation and focus states.
-   [ ] No horizontal overflow at 360px.

## Backend

-   [ ] Health endpoint works.
-   [ ] Built-in checkpoint loads.
-   [ ] Invalid images rejected.
-   [ ] Incompatible model checkpoints rejected.
-   [ ] No arbitrary pickle/code execution.
-   [ ] Inference returns valid predictions.
-   [ ] Interventions return valid images.
-   [ ] Metrics tested on hand-computable examples.
-   [ ] Temporary files cleaned.
-   [ ] CORS/upload limits configured.
-   [ ] No traceback/path leaks.

## End-to-end

-   [ ] First-time user completes demo without account.
-   [ ] Real image produces real prediction.
-   [ ] Intervention modifies actual pixels.
-   [ ] Modified image is passed through same model pipeline.
-   [ ] Report matches UI results.
-   [ ] Public Vercel URL reaches backend.
-   [ ] Cold start tested.
-   [ ] Fresh browser session tested.
-   [ ] Mobile tested.

## Scientific quality

-   [ ] Dataset/model/checkpoint provenance documented.
-   [ ] Benchmark metrics generated by reproducible scripts.
-   [ ] Group definitions/splits documented.
-   [ ] Interventions and limitations explained.
-   [ ] No unsupported causal claims.
-   [ ] No fabricated results/citations.

------------------------------------------------------------------------

# 16. Development schedule

## Week 1 --- Foundation and UI

1.  Inspect existing repo and preserve ML code.
2.  Create frontend and design tokens.
3.  Build landing page, navigation, responsive shell.
4.  Build Studio layout and upload components.
5.  Define API contract and backend skeleton.
6.  Verify real demo checkpoint and health/model endpoints.

**Deliverable:** polished UI shell and working local backend.

## Week 2 --- Real audit workflow

1.  Image upload/validation.
2.  Built-in inference.
3.  Top-k predictions.
4.  Foreground mask generation.
5.  Background and foreground interventions.
6.  Re-inference and comparison.
7.  Tests and correctness fixes.

**Deliverable:** end-to-end real audit locally.

## Week 3 --- Product completion

1.  Constrained custom checkpoint support if safe.
2.  Validation and useful errors.
3.  Benchmark dashboard from real outputs.
4.  Methodology and limitations.
5.  Report export.
6.  Responsive refinements/accessibility.

**Deliverable:** complete tested product locally.

## Week 4 --- Deployment and polish

1.  Select backend host based on current limits.
2.  Deploy frontend to Vercel.
3.  Deploy FastAPI separately.
4.  Configure env/CORS.
5.  Test public workflow and cold starts.
6.  Optimize inference and UI.
7.  README, screenshots, demo script.
8.  Test from fresh browser/network.

**Deliverable:** public, working product.

### If only two weeks

Do not let custom upload delay the product. Deliver: - Landing page. -
Built-in model. - Image upload. - Real inference. - One robust
intervention. - Side-by-side comparison. - Basic
benchmark/methodology. - Public deployment.

------------------------------------------------------------------------

# 17. Product quality bar

The finished product must: - Look like a credible research tool. - Have
deliberate typography, spacing, hierarchy, and responsive behavior. -
Make image comparisons the visual centerpiece. - Have no dead buttons or
fake charts. - Have no placeholder text in production. - Have no console
errors or broken routes. - Load and use a real PyTorch model. - Modify
real image pixels during interventions. - Re-run inference on modified
images. - Compute metrics instead of hardcoding them. - Produce reports
matching the actual audit. - Explain limitations.

A polished website without real inference is not complete. A correct
notebook without a usable product is also not the target.

------------------------------------------------------------------------

# 18. Instructions to Claude / Antigravity coding agent

You are acting as a senior product designer, frontend engineer, backend
engineer, and applied computer vision engineer. Build CVA as a complete
deployable product following this specification.

## Phase 0 --- Inspect before changing

1.  Inspect current repository and files.
2.  Identify existing training, checkpoint, dataset, intervention, and
    evaluation code.
3.  Reuse correct working ML code; do not rewrite it unnecessarily.
4.  Identify what is implemented, missing, or placeholder.
5.  Verify current Vercel Hobby plan and deployment constraints using
    official docs.
6.  Evaluate backend hosts based on current free tiers, sleeping,
    memory, disk, CPU, and limits.
7.  Produce implementation plan and file tree.
8.  Do not overwrite unrelated files.

## Phase 1 --- Design/frontend

1.  Implement visual design system first.
2.  Build premium dark research-lab UI, not generic dashboard.
3.  Build landing page and all routes.
4.  Build complete Studio with model/image selection, comparison, and
    results.
5.  Make responsive.
6.  Use real assets/outputs when available.
7.  Do not hardcode fake metrics/predictions.
8.  Every visible button must work.

## Phase 2 --- Backend/ML

1.  Build FastAPI backend and typed schemas.
2.  Implement built-in model registry/loading.
3.  Implement real preprocessing and inference.
4.  Implement segmentation/masks and interventions.
5.  Re-run inference on transformed images.
6.  Return structured results and artifacts.
7.  Add strict safe custom model validation.
8.  Implement temporary file cleanup.
9.  Add tests.

## Phase 3 --- Product features

1.  Custom model flow only for allowlisted architectures and safe
    weights.
2.  Benchmark dashboard from actual experiment outputs.
3.  Methodology/limitations.
4.  JSON and optional PDF report.
5.  Progress/error states and API status.
6.  Attribution and privacy copy.

## Phase 4 --- Deployment

1.  Deploy frontend to Vercel.
2.  Deploy backend separately.
3.  Configure environment variables and restrictive CORS.
4.  Test public end-to-end workflow.
5.  Test cold start.
6.  Test all direct routes and mobile.
7.  Document free-tier limitations.

## Working rules

-   Work in small verifiable increments.
-   Run the app and inspect it visually.
-   Run tests after meaningful changes.
-   Do not stop at static mockups.
-   Do not substitute mock API responses for real functionality.
-   Clearly label incomplete features.
-   Avoid unnecessary dependencies.
-   Do not spend all the schedule on animations before inference works.
-   Prioritize working built-in demo over custom uploads.
-   Keep code modular.
-   Explain major architectural decisions.
-   Ask questions only for genuine blockers.

## Final handoff

Provide: 1. Local setup commands. 2. Frontend/backend start commands. 3.
Deployment instructions. 4. Required environment variables. 5. Public
URL only after successful deployment and verification. 6. Tests actually
run and outcomes. 7. Supported model architectures/formats. 8. Known
limitations and unfinished features. 9. Actual benchmark results and
sources. 10. A short professor demo script.

Never claim deployment succeeded unless the public URL has been tested.

------------------------------------------------------------------------

# 19. Definition of done

CVA is complete when a first-time visitor can open the public website,
launch Studio, select a real built-in model, upload an image, receive a
real prediction, run at least one valid counterfactual intervention,
inspect the modified image and changed prediction, and understand the
limitations.

**The goal is a credible, interactive Computer Vision product backed by
reproducible deep learning experiments.**
