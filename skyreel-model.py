import json
import logging
import os
import secrets
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

import modal


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "skyreels-video-generation"

MODEL_ID = "modal/skyreels-v2-t2v-14b"
MODEL_NAME = "SkyReels V2 T2V 14B"

HF_MODEL_ID = "Skywork/SkyReels-V2-T2V-14B-720P-Diffusers"

MODEL_CACHE = "/models"
JOBS_DIR = "/jobs"

KEEP_WARM_SECONDS = 120

FPS = 24

MIN_DURATION = 1
MAX_DURATION = 6

# SkyReels official/recommended settings
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 6.0
FLOW_SHIFT = 8.0

# Modal A100-80GB current GPU price
L40S_USD_PER_SECOND = 0.000542

STATUS_POLL_RETRY_SECONDS = 2


# ============================================================
# RESOLUTIONS
# ============================================================

SUPPORTED_RESOLUTIONS = [
    "480p",
    "720p",
]

SUPPORTED_ASPECT_RATIOS = [
    "9:16",
    "16:9",
]


SIZE_MAP = {
    ("480p", "9:16"): (480, 832),
    ("480p", "16:9"): (832, 480),

    ("720p", "9:16"): (720, 1280),
    ("720p", "16:9"): (1280, 720),
}


# ============================================================
# MODAL
# ============================================================

app = modal.App(APP_NAME)


model_volume = modal.Volume.from_name(
    "skyreels-v2-model-cache",
    create_if_missing=True,
)


jobs_volume = modal.Volume.from_name(
    "skyreels-video-generation-jobs",
    create_if_missing=True,
)


api_secret = modal.Secret.from_name(
    "video-api-secret",
    required_keys=[
        "MODAL_VIDEO_API_KEY"
    ],
)


# ============================================================
# IMAGES
# ============================================================

gpu_image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .apt_install(
        "ffmpeg",
        "git",
    )
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",

        # SkyReels support
        "diffusers>=0.35.0",
        "transformers>=4.49.0",
        "accelerate>=1.2.0",

        "huggingface_hub>=0.29.0",
        "safetensors",
        "sentencepiece",
        "protobuf",

        "numpy",
        "pillow",
        "imageio",
        "imageio-ffmpeg",
        "ftfy",
    )
)


api_image = (
    modal.Image.debian_slim(
        python_version="3.11"
    )
    .pip_install(
        "fastapi[standard]",
    )
)


# ============================================================
# HELPERS
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat(
        timespec="seconds"
    )


def log_event(
    event: str,
    **fields,
):
    """
    Compact application logging.

    We intentionally do not log GET polling/content/model
    requests here.
    """

    pieces = [
        utc_now(),
        event,
    ]

    for key, value in fields.items():
        if value is not None:
            pieces.append(
                f"{key}={value}"
            )

    print(
        " | ".join(pieces),
        flush=True,
    )


def job_json_path(
    job_id: str,
) -> Path:

    return Path(
        JOBS_DIR,
        f"{job_id}.json",
    )


def job_video_path(
    job_id: str,
) -> Path:

    return Path(
        JOBS_DIR,
        f"{job_id}.mp4",
    )


def _write_job_file(
    job_id: str,
    data: dict,
):
    path = job_json_path(
        job_id
    )

    temp = Path(
        f"{path}.tmp"
    )

    temp.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temp,
        path,
    )


def write_job(
    job_id: str,
    data: dict,
):
    _write_job_file(
        job_id,
        data,
    )

    jobs_volume.commit()


async def write_job_async(
    job_id: str,
    data: dict,
):
    _write_job_file(
        job_id,
        data,
    )

    await jobs_volume.commit.aio()


def read_job(
    job_id: str,
):
    jobs_volume.reload()

    path = job_json_path(
        job_id
    )

    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


async def read_job_async(
    job_id: str,
):
    """
    Async Modal Volume access.

    Prevents:

    AsyncUsageWarning:
    blocking Modal interface used in async context
    """

    await jobs_volume.reload.aio()

    path = job_json_path(
        job_id
    )

    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )



async def reconcile_modal_call(job: dict) -> dict:
    """
    Reconcile a non-terminal job with its underlying Modal FunctionCall.

    This catches failures that happen before VideoGenerator.generate() starts,
    including GPU scheduling/startup, snapshot restore, container bootstrap,
    and @modal.enter() failures.
    """
    if job.get("status") in {"completed", "failed"}:
        return job

    call_id = job.get("modal_call_id")
    if not call_id:
        return job

    try:
        call = modal.FunctionCall.from_id(call_id)
        await call.get.aio(timeout=0)
    except TimeoutError:
        return job
    except Exception as error:
        job.update(
            {
                "status": "failed",
                "stage": "worker_failed",
                "progress": 0,
                "error": str(error),
                "completed_at": int(time.time()),
            }
        )
        await write_job_async(job["id"], job)
        log_event(
            "modal_worker_failed",
            job=job["id"],
            call_id=call_id,
            error=repr(error),
        )

    return job


def duration_to_frames(
    duration: int,
) -> int:
    """
    SkyReels / Wan VAE works best with frame counts
    compatible with 4n + 1.

    1 second @ 24fps ~= 25 frames
    2 seconds ~= 49
    3 seconds ~= 73
    4 seconds ~= 97
    5 seconds ~= 121
    """

    target = duration * FPS

    n = round(
        (target - 1) / 4
    )

    frames = (
        4 * n + 1
    )

    max_frames = MAX_DURATION * FPS
    max_n = round((max_frames - 1) / 4)
    max_compatible_frames = 4 * max_n + 1

    return max(
        25,
        min(
            frames,
            max_compatible_frames,
        ),
    )


# ============================================================
# GPU WORKER
# ============================================================

@app.cls(
    image=gpu_image,

    gpu="L40S",

    volumes={
        MODEL_CACHE: model_volume,
        JOBS_DIR: jobs_volume,
    },

    min_containers=0,

    max_containers=1,

    scaledown_window=KEEP_WARM_SECONDS,

    timeout=3600,

    startup_timeout=1800,

    enable_memory_snapshot=True,

    experimental_options={
        "enable_gpu_snapshot": True,
    },
)
class VideoGenerator:

    # ========================================================
    # MODEL LOAD
    # ========================================================

    @modal.enter(snap=True)
    def load_model(
        self,
    ):
        import torch

        from diffusers import (
            AutoencoderKLWan,
            SkyReelsV2Pipeline,
            UniPCMultistepScheduler,
        )

        from huggingface_hub import (
            snapshot_download,
        )

        startup_started = (
            time.time()
        )

        log_event(
            "worker_starting",
            model=MODEL_NAME,
            gpu="L40S",
        )

        model_path = (
            Path(MODEL_CACHE)
            / "skyreels-v2-14b"
        )

        sentinel = (
            model_path
            / ".download_complete"
        )

        # --------------------------------------------
        # Download only once
        # --------------------------------------------

        if sentinel.exists():

            log_event(
                "model_cache_hit"
            )

        else:

            log_event(
                "model_cache_miss"
            )

            model_path.mkdir(
                parents=True,
                exist_ok=True,
            )

            snapshot_download(
                repo_id=HF_MODEL_ID,
                local_dir=str(
                    model_path
                ),
            )

            sentinel.write_text(
                "complete",
                encoding="utf-8",
            )

            model_volume.commit()

        # --------------------------------------------
        # VAE
        # --------------------------------------------

        log_event(
            "model_loading",
            component="vae",
        )

        vae = (
            AutoencoderKLWan
            .from_pretrained(
                str(model_path),

                subfolder="vae",

                torch_dtype=(
                    torch.float32
                ),
            )
        )

        # --------------------------------------------
        # Pipeline
        # --------------------------------------------

        log_event(
            "model_loading",
            component="pipeline",
        )

        self.pipe = (
            SkyReelsV2Pipeline
            .from_pretrained(
                str(model_path),

                vae=vae,

                torch_dtype=(
                    torch.bfloat16
                ),
            )
        )

        # --------------------------------------------
        # Official T2V scheduler config
        # --------------------------------------------

        self.pipe.scheduler = (
            UniPCMultistepScheduler
            .from_config(
                self.pipe.scheduler.config,

                flow_shift=FLOW_SHIFT,
            )
        )

        # --------------------------------------------
        # Memory optimization
        #
        # A100 80 GB gives us substantial VRAM,
        # but SkyReels 14B is large.
        #
        # CPU offload greatly lowers OOM risk.
        # --------------------------------------------

        self.pipe.enable_model_cpu_offload()

        # VAE tiling prevents large 720p decode spikes
        try:
            self.pipe.vae.enable_tiling()
        except Exception:
            pass

        self.torch = torch

        load_seconds = (
            time.time()
            - startup_started
        )

        log_event(
            "model_ready",
            seconds=round(
                load_seconds,
                2,
            ),
        )

    # ========================================================
    # GENERATION
    # ========================================================

    @modal.method()
    def generate(
        self,
        job_id: str,
        prompt: str,
        model: str,
        duration: int,
        resolution: str,
        aspect_ratio: str,
        seed: int,
    ):

        from diffusers.utils import (
            export_to_video,
        )

        started_at = (
            time.time()
        )

        job = {
            "id": job_id,
            "status": "in_progress",
            "stage": "preparing",

            "model": model,

            "progress": 5,

            "prompt": prompt,

            "duration": duration,

            "resolution": resolution,

            "aspect_ratio": (
                aspect_ratio
            ),

            "seed": seed,

            "created_at": int(
                started_at
            ),

            "error": "",
        }

        write_job(
            job_id,
            job,
        )

        try:

            width, height = (
                SIZE_MAP[
                    (
                        resolution,
                        aspect_ratio,
                    )
                ]
            )

            frame_count = (
                duration_to_frames(
                    duration
                )
            )

            output_path = (
                job_video_path(
                    job_id
                )
            )

            log_event(
                "generation_started",
                job=job_id,
                width=width,
                height=height,
                frames=frame_count,
                steps=NUM_INFERENCE_STEPS,
                seed=seed,
            )

            # --------------------------------------------
            # UPDATE STATUS
            # --------------------------------------------

            job[
                "stage"
            ] = "inference"

            job[
                "progress"
            ] = 20

            write_job(
                job_id,
                job,
            )

            # --------------------------------------------
            # SEED
            # --------------------------------------------

            generator = (
                self.torch.Generator(
                    device="cpu"
                )
                .manual_seed(
                    seed
                )
            )

            # --------------------------------------------
            # GENERATE
            # --------------------------------------------

            inference_started = (
                time.time()
            )

            result = self.pipe(
                prompt=prompt,

                height=height,

                width=width,

                num_frames=(
                    frame_count
                ),

                num_inference_steps=(
                    NUM_INFERENCE_STEPS
                ),

                guidance_scale=(
                    GUIDANCE_SCALE
                ),

                generator=generator,
            )

            frames = (
                result.frames[0]
            )

            inference_seconds = (
                time.time()
                - inference_started
            )

            # --------------------------------------------
            # ENCODE
            # --------------------------------------------

            job[
                "stage"
            ] = "encoding"

            job[
                "progress"
            ] = 90

            job[
                "inference_seconds"
            ] = inference_seconds

            write_job(
                job_id,
                job,
            )

            encode_started = (
                time.time()
            )

            export_to_video(
                frames,
                str(output_path),

                fps=FPS,

                quality=8,
            )

            encode_seconds = (
                time.time()
                - encode_started
            )

            del frames
            del result

            self.torch.cuda.empty_cache()

            # --------------------------------------------
            # VERIFY FILE
            # --------------------------------------------

            if not output_path.exists():

                raise RuntimeError(
                    "Generated video "
                    "was not created"
                )

            file_size = (
                output_path
                .stat()
                .st_size
            )

            if file_size <= 0:

                raise RuntimeError(
                    "Generated MP4 is empty"
                )

            jobs_volume.commit()

            total_seconds = (
                time.time()
                - started_at
            )

            # --------------------------------------------
            # COST
            # --------------------------------------------

            usage_cost_usd = round(
                total_seconds
                * L40S_USD_PER_SECOND,

                6,
            )

            # --------------------------------------------
            # COMPLETE
            # --------------------------------------------

            job.update(
                {
                    "status":
                        "completed",

                    "stage":
                        "completed",

                    "progress":
                        100,

                    "filename":
                        output_path.name,

                    "size_bytes":
                        file_size,

                    "inference_seconds":
                        inference_seconds,

                    "encode_seconds":
                        encode_seconds,

                    "total_seconds":
                        total_seconds,

                    "usage_cost_usd":
                        usage_cost_usd,

                    "gpu_billed_seconds":
                        total_seconds,

                    "gpu_rate_usd_per_second":
                        L40S_USD_PER_SECOND,

                    "completed_at":
                        int(
                            time.time()
                        ),

                    "error":
                        "",
                }
            )

            write_job(
                job_id,
                job,
            )

            log_event(
                "generation_completed",
                job=job_id,

                inference_s=round(
                    inference_seconds,
                    2,
                ),

                total_s=round(
                    total_seconds,
                    2,
                ),

                cost_usd=(
                    usage_cost_usd
                ),
            )

        except Exception as error:

            traceback.print_exc()

            job.update(
                {
                    "status":
                        "failed",

                    "stage":
                        "failed",

                    "error":
                        str(error),

                    "completed_at":
                        int(
                            time.time()
                        ),
                }
            )

            write_job(
                job_id,
                job,
            )

            log_event(
                "generation_failed",
                job=job_id,
                error=repr(
                    error
                ),
            )

            raise


# ============================================================
# FASTAPI
# ============================================================

@app.function(
    image=api_image,
    volumes={
        JOBS_DIR: jobs_volume,
    },
    secrets=[
        api_secret,
    ],
    timeout=300,
)
@modal.asgi_app(
    label="skyreels"
)
def api():

    from fastapi import (
        FastAPI,
        HTTPException,
        Request,
        Response,
    )

    from fastapi.responses import (
        FileResponse,
    )

    # Modal emits its own edge/router request lines. Silence framework-level
    # access logging where possible; Retry-After lets clients back off.
    logging.getLogger("uvicorn.access").disabled = True
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)

    api_app = FastAPI(
        title="SkyReels Video API",
    )

    # ========================================================
    # AUTH
    # ========================================================

    def authenticate(request: Request):
        expected_key = os.environ.get("MODAL_VIDEO_API_KEY")
        if not expected_key:
            raise HTTPException(
                status_code=500,
                detail="Server API key is not configured",
            )

        authorization = request.headers.get("Authorization", "")
        expected = f"Bearer {expected_key}"

        if not secrets.compare_digest(authorization, expected):
            raise HTTPException(
                status_code=401,
                detail="Invalid API key",
                headers={
                    "WWW-Authenticate": "Bearer",
                },
            )

    # ========================================================
    # HEALTH
    # ========================================================

    @api_app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model": MODEL_ID,
        }

    # ========================================================
    # MODELS
    #
    # Same OpenRouter-style schema consumed by modal.go.
    # GET /api/v1/videos/models
    # ========================================================

    @api_app.get("/api/v1/videos/models")
    async def list_models(request: Request):
        authenticate(request)

        return {
            "data": [
                {
                    "id": MODEL_ID,
                    "name": MODEL_NAME,
                    "supported_durations": list(
                        range(MIN_DURATION, MAX_DURATION + 1)
                    ),
                    "supported_resolutions": SUPPORTED_RESOLUTIONS,
                    "supported_aspect_ratios": SUPPORTED_ASPECT_RATIOS,
                    "generate_audio": False,
                    "pricing_skus": {},
                }
            ]
        }

    # ========================================================
    # CREATE VIDEO
    #
    # OpenRouter-compatible request fields:
    # model, prompt, duration, resolution, aspect_ratio,
    # generate_audio. seed is accepted as an optional Modal extension.
    #
    # POST /api/v1/videos
    # ========================================================

    @api_app.post(
        "/api/v1/videos",
        status_code=202,
    )
    async def create_video(request: Request):
        authenticate(request)

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(
                status_code=400,
                detail="Invalid JSON",
            )

        model = body.get("model", MODEL_ID)
        if model != MODEL_ID:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported model: {model}",
            )

        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise HTTPException(
                status_code=400,
                detail="prompt must be a string",
            )

        prompt = prompt.strip()
        if not prompt:
            raise HTTPException(
                status_code=400,
                detail="prompt is required",
            )

        raw_duration = body.get("duration", 4)
        try:
            duration = int(raw_duration)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="duration must be an integer",
            )

        if duration < MIN_DURATION or duration > MAX_DURATION:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"duration must be between "
                    f"{MIN_DURATION} and {MAX_DURATION} seconds"
                ),
            )

        resolution = str(
            body.get("resolution", "480p")
        ).lower()

        if resolution not in SUPPORTED_RESOLUTIONS:
            raise HTTPException(
                status_code=400,
                detail="resolution must be '480p' or '720p'",
            )

        aspect_ratio = str(
            body.get("aspect_ratio", "9:16")
        )

        if aspect_ratio not in SUPPORTED_ASPECT_RATIOS:
            raise HTTPException(
                status_code=400,
                detail="aspect_ratio must be '9:16' or '16:9'",
            )

        generate_audio = body.get("generate_audio", False)
        if generate_audio:
            raise HTTPException(
                status_code=400,
                detail="SkyReels V2 T2V does not generate audio",
            )

        raw_seed = body.get("seed")
        if raw_seed is None:
            seed = secrets.randbelow(2**31 - 1)
        else:
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400,
                    detail="seed must be an integer",
                )

            if seed < 0:
                seed = secrets.randbelow(2**31 - 1)

        combination = (
            resolution,
            aspect_ratio,
        )
        if combination not in SIZE_MAP:
            raise HTTPException(
                status_code=400,
                detail="Unsupported resolution and aspect ratio",
            )

        job_id = "gen_" + uuid.uuid4().hex

        job = {
            "id": job_id,
            "status": "pending",
            "stage": "queued",
            "model": model,
            "progress": 0,
            "prompt": prompt,
            "duration": duration,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "seed": seed,
            "created_at": int(time.time()),
            "error": "",
        }

        await write_job_async(job_id, job)

        try:
            log_event(
                "gpu_dispatch_requested",
                job=job_id,
            )

            call = await (
                VideoGenerator()
                .generate
                .spawn.aio(
                    job_id=job_id,
                    prompt=prompt,
                    model=model,
                    duration=duration,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    seed=seed,
                )
            )

            job["modal_call_id"] = call.object_id
            job["stage"] = "dispatched"

            await write_job_async(
                job_id,
                job,
            )

            log_event(
                "gpu_job_dispatched",
                job=job_id,
                call_id=call.object_id,
            )

        except Exception as error:
            job.update(
                {
                    "status": "failed",
                    "stage": "dispatch_failed",
                    "error": str(error),
                    "completed_at": int(time.time()),
                }
            )

            await write_job_async(
                job_id,
                job,
            )

            log_event(
                "gpu_dispatch_failed",
                job=job_id,
                error=repr(error),
            )

            raise HTTPException(
                status_code=500,
                detail="Could not start video generation",
            )

        return {
            "id": job_id,
            "status": "pending",
            "stage": "dispatched",
            "model": model,
            "progress": 0,
            "modal_call_id": call.object_id,
            "poll_after_seconds": STATUS_POLL_RETRY_SECONDS,
        }

    # ========================================================
    # STATUS
    #
    # GET /api/v1/videos/{id}
    # ========================================================

    @api_app.get("/api/v1/videos/{job_id}")
    async def get_video(
        job_id: str,
        request: Request,
        response: Response,
    ):
        authenticate(request)

        job = await read_job_async(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Generation not found",
            )

        job = await reconcile_modal_call(job)

        if job.get("status") not in {"completed", "failed"}:
            response.headers["Retry-After"] = str(
                STATUS_POLL_RETRY_SECONDS
            )

        result = {
            "id": job["id"],
            "status": job["status"],
            "model": job.get("model", MODEL_ID),
            "progress": job.get("progress", 0),
            "stage": job.get("stage", "unknown"),
        }

        if job.get("status") not in {"completed", "failed"}:
            result["poll_after_seconds"] = STATUS_POLL_RETRY_SECONDS

        if job["status"] == "completed":
            base_url = str(
                request.base_url
            ).rstrip("/")

            content_url = (
                f"{base_url}"
                f"/api/v1/videos/"
                f"{job_id}"
                f"/content?index=0"
            )

            result["unsigned_urls"] = [
                content_url
            ]

            result["usage"] = {
                "cost": job.get(
                    "usage_cost_usd",
                    0.0,
                ),
                "currency": "USD",
                "gpu_seconds": job.get(
                    "gpu_billed_seconds"
                ),
                "gpu_rate_usd_per_second": job.get(
                    "gpu_rate_usd_per_second",
                    L40S_USD_PER_SECOND,
                ),
                "basis": "measured_generation_runtime",
            }

            result["timings"] = {
                "inference_seconds": job.get(
                    "inference_seconds"
                ),
                "encode_seconds": job.get(
                    "encode_seconds"
                ),
                "total_seconds": job.get(
                    "total_seconds"
                ),
            }

        if job["status"] == "failed":
            result["error"] = job.get(
                "error",
                "Generation failed",
            )

        return result

    # ========================================================
    # VIDEO CONTENT
    #
    # GET /api/v1/videos/{id}/content?index=0
    # ========================================================

    @api_app.get("/api/v1/videos/{job_id}/content")
    async def get_content(
        job_id: str,
        request: Request,
        index: int = 0,
    ):
        authenticate(request)

        if index != 0:
            raise HTTPException(
                status_code=404,
                detail="Only output index 0 exists",
            )

        job = await read_job_async(job_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="Generation not found",
            )

        job = await reconcile_modal_call(job)

        if job["status"] != "completed":
            if job["status"] == "failed":
                raise HTTPException(
                    status_code=409,
                    detail=job.get(
                        "error",
                        "Generation failed",
                    ),
                )

            raise HTTPException(
                status_code=409,
                detail="Generation is not completed yet",
            )

        path = job_video_path(job_id)

        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail="Generated video file not found",
            )

        return FileResponse(
            path=str(path),
            media_type="video/mp4",
            filename=f"{job_id}.mp4",
        )

    return api_app
