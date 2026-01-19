def test_load_model():
    from iopaint.model_manager import ModelManager

    models = ["lama", "ldm", "zits", "mat", "fcf", "manga", "migan"]
    for m in models:
        ModelManager(
            name=m,
            device="cpu",
            no_half=False,
            disable_nsfw=False,
            sd_cpu_textencoder=True,
            cpu_offload=True,
        )
