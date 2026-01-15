import os
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import typer
from fastapi import FastAPI
from loguru import logger
from typer import Option
from typer_config import use_json_config

from iopaint.const import *
from iopaint.config_loader import register_env_override
from iopaint.runtime import setup_model_dir, dump_environment_info, check_device
from iopaint.schema import InteractiveSegModel, Device, RealESRGANModel, RemoveBGModel

typer_app = typer.Typer(pretty_exceptions_show_locals=False, add_completion=False)


@typer_app.command(help="Install all plugins dependencies")
def install_plugins_packages():
    from iopaint.installer import install_plugins_package

    install_plugins_package()


@typer_app.command(help="Download SD/SDXL normal/inpainting model from HuggingFace")
def download(
    model: str = Option(
        ..., help="Model id on HuggingFace e.g: runwayml/stable-diffusion-inpainting"
    ),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import cli_download_model

    cli_download_model(model)


@typer_app.command(name="list", help="List downloaded models")
def list_model(
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import scan_models

    scanned_models = scan_models()
    for it in scanned_models:
        print(it.name)


@typer_app.command(help="Batch processing images")
def run(
    model: str = Option("lama"),
    device: Device = Option(Device.cpu),
    image: Path = Option(..., help="Image folders or file path"),
    mask: Path = Option(
        ...,
        help="Mask folders or file path. "
        "If it is a directory, the mask images in the directory should have the same name as the original image."
        "If it is a file, all images will use this mask."
        "Mask will automatically resize to the same size as the original image.",
    ),
    output: Path = Option(..., help="Output directory or file path"),
    config: Path = Option(
        None, help="Config file path. You can use dump command to create a base config."
    ),
    concat: bool = Option(
        False, help="Concat original image, mask and output images into one image"
    ),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        file_okay=False,
        callback=setup_model_dir,
    ),
):
    from iopaint.download import cli_download_model, scan_models

    scanned_models = scan_models()
    if model not in [it.name for it in scanned_models]:
        logger.info(f"{model} not found in {model_dir}, try to downloading")
        cli_download_model(model)

    from iopaint.batch_processing import batch_inpaint

    batch_inpaint(model, device, image, mask, output, config, concat)


@typer_app.command(help="Start IOPaint server")
@use_json_config()
def start(
    host: str = Option("127.0.0.1"),
    port: int = Option(8080),
    inbrowser: bool = Option(False, help=INBROWSER_HELP),
    model: str = Option(
        DEFAULT_MODEL,
        help=f"Erase models: [{', '.join(AVAILABLE_MODELS)}].\n"
        f"Diffusion models: [{', '.join(DIFFUSION_MODELS)}] or any SD/SDXL normal/inpainting models on HuggingFace.",
    ),
    model_dir: Path = Option(
        DEFAULT_MODEL_DIR,
        help=MODEL_DIR_HELP,
        dir_okay=True,
        file_okay=False,
        callback=setup_model_dir,
    ),
    low_mem: bool = Option(False, help=LOW_MEM_HELP),
    no_half: bool = Option(False, help=NO_HALF_HELP),
    cpu_offload: bool = Option(False, help=CPU_OFFLOAD_HELP),
    disable_nsfw_checker: bool = Option(False, help=DISABLE_NSFW_HELP),
    cpu_textencoder: bool = Option(False, help=CPU_TEXTENCODER_HELP),
    local_files_only: bool = Option(False, help=LOCAL_FILES_ONLY_HELP),
    device: Device = Option(Device.cpu),
    input: Optional[Path] = Option(None, help=INPUT_HELP),
    mask_dir: Optional[Path] = Option(
        None, help=MODEL_DIR_HELP, dir_okay=True, file_okay=False
    ),
    output_dir: Optional[Path] = Option(
        None, help=OUTPUT_DIR_HELP, dir_okay=True, file_okay=False
    ),
    quality: int = Option(100, help=QUALITY_HELP),
    enable_interactive_seg: bool = Option(False, help=INTERACTIVE_SEG_HELP),
    interactive_seg_model: InteractiveSegModel = Option(
        InteractiveSegModel.sam2_1_tiny, help=INTERACTIVE_SEG_MODEL_HELP
    ),
    interactive_seg_device: Device = Option(Device.cpu),
    enable_remove_bg: bool = Option(False, help=REMOVE_BG_HELP),
    remove_bg_device: Device = Option(Device.cpu, help=REMOVE_BG_DEVICE_HELP),
    remove_bg_model: RemoveBGModel = Option(RemoveBGModel.briaai_rmbg_1_4),
    enable_anime_seg: bool = Option(False, help=ANIMESEG_HELP),
    enable_realesrgan: bool = Option(False),
    realesrgan_device: Device = Option(Device.cpu),
    realesrgan_model: RealESRGANModel = Option(RealESRGANModel.realesr_general_x4v3),
    enable_gfpgan: bool = Option(False),
    gfpgan_device: Device = Option(Device.cpu),
    enable_restoreformer: bool = Option(False),
    restoreformer_device: Device = Option(Device.cpu),
    # OpenAI-compatible API options
    openai_api_key: Optional[str] = Option(
        None,
        envvar="AIE_OPENAI_API_KEY",
        help="API key for OpenAI-compatible image API. Required for openai-compat model.",
    ),
    openai_base_url: Optional[str] = Option(
        None,
        envvar="AIE_OPENAI_BASE_URL",
        help="Base URL for OpenAI-compatible API. Default: https://api.openai.com/v1",
    ),
    openai_model: Optional[str] = Option(
        None,
        envvar="AIE_OPENAI_MODEL",
        help="Default model for OpenAI image operations. Default: gpt-image-1",
    ),
    # Budget safety options
    budget_daily_cap: Optional[float] = Option(
        None,
        envvar="AIE_BUDGET_DAILY_CAP",
        help="Daily budget cap in USD. 0 = unlimited. Default: 10.0",
    ),
    budget_monthly_cap: Optional[float] = Option(
        None,
        envvar="AIE_BUDGET_MONTHLY_CAP",
        help="Monthly budget cap in USD. 0 = unlimited. Default: 100.0",
    ),
    budget_session_cap: Optional[float] = Option(
        None,
        envvar="AIE_BUDGET_SESSION_CAP",
        help="Per-session budget cap in USD. 0 = unlimited. Default: 5.0",
    ),
    budget_rate_limit: Optional[int] = Option(
        None,
        envvar="AIE_RATE_LIMIT_SECONDS",
        help="Min seconds between expensive operations. Default: 10",
    ),
    budget_data_dir: Optional[Path] = Option(
        None,
        envvar="AIE_DATA_DIR",
        help="Data directory for budget DB and cache. Default: ~/.iopaint/data",
    ),
):
    dump_environment_info()
    device = check_device(device)
    remove_bg_device = check_device(remove_bg_device)
    realesrgan_device = check_device(realesrgan_device)
    gfpgan_device = check_device(gfpgan_device)

    if input and not input.exists():
        logger.error(f"invalid --input: {input} not exists")
        exit(-1)
    if mask_dir and not mask_dir.exists():
        logger.error(f"invalid --mask-dir: {mask_dir} not exists")
        exit(-1)
    if input and input.is_dir() and not output_dir:
        logger.error(
            "invalid --output-dir: --output-dir must be set when --input is a directory"
        )
        exit(-1)
    if output_dir:
        output_dir = output_dir.expanduser().absolute()
        logger.info(f"Image will be saved to {output_dir}")
        if not output_dir.exists():
            logger.info(f"Create output directory {output_dir}")
            output_dir.mkdir(parents=True)
    if mask_dir:
        mask_dir = mask_dir.expanduser().absolute()

    model_dir = model_dir.expanduser().absolute()

    if local_files_only:
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    env_overrides = []

    # Set OpenAI environment variables from CLI flags if provided
    if openai_api_key:
        os.environ["AIE_OPENAI_API_KEY"] = openai_api_key
        env_overrides.append("AIE_OPENAI_API_KEY")
    if openai_base_url:
        os.environ["AIE_OPENAI_BASE_URL"] = openai_base_url
        env_overrides.append("AIE_OPENAI_BASE_URL")
    if openai_model:
        os.environ["AIE_OPENAI_MODEL"] = openai_model
        env_overrides.append("AIE_OPENAI_MODEL")

    # Set budget environment variables from CLI flags if provided
    if budget_daily_cap is not None:
        os.environ["AIE_BUDGET_DAILY_CAP"] = str(budget_daily_cap)
        env_overrides.append("AIE_BUDGET_DAILY_CAP")
    if budget_monthly_cap is not None:
        os.environ["AIE_BUDGET_MONTHLY_CAP"] = str(budget_monthly_cap)
        env_overrides.append("AIE_BUDGET_MONTHLY_CAP")
    if budget_session_cap is not None:
        os.environ["AIE_BUDGET_SESSION_CAP"] = str(budget_session_cap)
        env_overrides.append("AIE_BUDGET_SESSION_CAP")
    if budget_rate_limit is not None:
        os.environ["AIE_RATE_LIMIT_SECONDS"] = str(budget_rate_limit)
        env_overrides.append("AIE_RATE_LIMIT_SECONDS")
    if budget_data_dir is not None:
        os.environ["AIE_DATA_DIR"] = str(budget_data_dir.expanduser().absolute())
        env_overrides.append("AIE_DATA_DIR")

    if env_overrides:
        register_env_override(env_overrides)

    from iopaint.download import cli_download_model, scan_models

    scanned_models = scan_models()
    if model not in [it.name for it in scanned_models]:
        logger.info(f"{model} not found in {model_dir}, try to downloading")
        cli_download_model(model)

    from iopaint.api import Api
    from iopaint.schema import ApiConfig

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if inbrowser:
            webbrowser.open(f"http://localhost:{port}", new=0, autoraise=True)
        yield

    app = FastAPI(lifespan=lifespan)

    api_config = ApiConfig(
        host=host,
        port=port,
        inbrowser=inbrowser,
        model=model,
        no_half=no_half,
        low_mem=low_mem,
        cpu_offload=cpu_offload,
        disable_nsfw_checker=disable_nsfw_checker,
        local_files_only=local_files_only,
        cpu_textencoder=cpu_textencoder if device == Device.cuda else False,
        device=device,
        input=input,
        mask_dir=mask_dir,
        output_dir=output_dir,
        quality=quality,
        enable_interactive_seg=enable_interactive_seg,
        interactive_seg_model=interactive_seg_model,
        interactive_seg_device=interactive_seg_device,
        enable_remove_bg=enable_remove_bg,
        remove_bg_device=remove_bg_device,
        remove_bg_model=remove_bg_model,
        enable_anime_seg=enable_anime_seg,
        enable_realesrgan=enable_realesrgan,
        realesrgan_device=realesrgan_device,
        realesrgan_model=realesrgan_model,
        enable_gfpgan=enable_gfpgan,
        gfpgan_device=gfpgan_device,
        enable_restoreformer=enable_restoreformer,
        restoreformer_device=restoreformer_device,
    )
    print(api_config.model_dump_json(indent=4))
    api = Api(app, api_config)
    api.launch()


@typer_app.command(help="Start IOPaint web config page")
def start_web_config(
    config_file: Path = Option("config.json"),
):
    dump_environment_info()
    from iopaint.web_config import main

    main(config_file)
