"""Small runtime helpers; GPU capacity is diagnostic, not an arbitrary cutoff."""
import warnings


def configure_hf_token(secrets, environ):
    """Export the optional Kaggle secret before starting Hugging Face subprocesses."""
    try:
        token = secrets.get_secret("HF_TOKEN")
    except Exception:
        # Missing or ungranted secret: public model downloads can continue.
        # Never print exception details, which may contain authentication data.
        token = None
    if token and token.strip():
        environ["HF_TOKEN"] = token.strip()
    enabled = bool(environ.get("HF_TOKEN", "").strip())
    print("HF_TOKEN configured for model downloads." if enabled else
          "HF_TOKEN not configured; public model downloads will be unauthenticated. "
          "To authenticate, enable the HF_TOKEN secret in Kaggle Add-ons > Secrets.")
    return enabled


def report_gpu(torch):
    if not torch.cuda.is_available():
        raise RuntimeError("Enable a CUDA GPU in Kaggle Settings before running")
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, _ = torch.cuda.mem_get_info(index)
        info = {"index": index, "name": props.name,
                "total_gib": props.total_memory / 1024**3, "free_gib": free / 1024**3}
        devices.append(info)
        print(f"GPU {index}: {info['name']}; total={info['total_gib']:.2f} GiB; free={info['free_gib']:.2f} GiB")
    print("Experiments use GPU 0, batch_size=1 and grad_accum=8; GPU capacities are not combined.")
    if devices[0]["total_gib"] < 14:
        warnings.warn("GPU 0 has less than 14 GiB. E2/E3 may run out of memory; select a T4/P100 or larger GPU.")
    return devices
