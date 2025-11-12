"""
Deploy RAGEN with REAL WebArena benchmark on Modal
"""
import modal

# ----- BUILD THE IMAGE -----
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "curl", "ca-certificates")
    .pip_install(
        "numpy>=1.25.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "accelerate>=0.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "gym==0.24.0",
        "playwright",
        "beautifulsoup4",
        "requests",
        "scikit-learn",
        "pandas",
        "rich",
        "thefuzz",
        "python-Levenshtein",
    )
    .run_commands("playwright install")
)

app = modal.App("ragen-webarena", image=image)
volume = modal.Volume.from_name("ragen-outputs", create_if_missing=True)




# ----- LAUNCH WEBARENA WEBSITES -----
@app.function(timeout=3600, concurrency_limit=1)
def launch_webarena_sites():
    """Run the core WebArena Docker websites inside Modal"""
    import subprocess, time, os

    print("=" * 60, flush=True)
    print("🌐 Starting WebArena Websites inside Modal", flush=True)
    print("=" * 60, flush=True)

    os.makedirs("/root/webarena-sites", exist_ok=True)
    os.chdir("/root/webarena-sites")

    sites = {
        "shopping": (
            "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar",
            7770,
        ),
        "shopping_admin": (
            "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar",
            7780,
        ),
        "forum": (
            "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar",
            9999,
        ),
    }

    for name, (url, port) in sites.items():
        print(f"⬇️ Downloading {name} image...", flush=True)
        subprocess.run(f"wget -q -O {name}.tar {url}", shell=True, check=True)
        print(f"📦 Loading {name} into Docker...", flush=True)
        subprocess.run(f"docker load --input {name}.tar", shell=True, check=True)
        print(f"🚀 Running {name} on port {port}...", flush=True)
        subprocess.Popen(f"docker run --name {name} -p {port}:80 -d {name}", shell=True)

    print("⏳ Waiting 30 seconds for all sites to boot up...", flush=True)
    time.sleep(30)

    print("✅ All WebArena websites are running inside Modal!", flush=True)
    return {"status": "running"}


# ----- TRAIN FUNCTION -----
@app.function(
    gpu="H100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
)
def train(code_tar: bytes, config_yaml: str):
    """Run RAGEN training on WebArena"""
    import subprocess, sys, os, tarfile, io

    print("=" * 60, flush=True)
    print("🤖 RAGEN + WEBARENA BENCHMARK ON H100", flush=True)
    print("=" * 60, flush=True)

    work_dir = "/root/work"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    # ----- unpack your code -----
    print("\n📦 Extracting code...", flush=True)
    tar = tarfile.open(fileobj=io.BytesIO(code_tar))
    tar.extractall()
    tar.close()

    os.makedirs("configs", exist_ok=True)
    with open("configs/ragen_webarena.yaml", "w") as f:
        f.write(config_yaml)

    # ----- install your repo -----
    print("\n📥 Installing RAGEN...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    # ----- setup WebArena -----
    print("\n🌐 Setting up WebArena...", flush=True)
    if not os.path.exists("/root/webarena"):
        subprocess.run(["git", "clone", "https://github.com/web-arena-x/webarena.git", "/root/webarena"], check=True)

    os.chdir("/root/webarena")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    # Generate test data
    print("\n⚙️ Generating WebArena config files...", flush=True)
    subprocess.run([sys.executable, "scripts/generate_test_data.py"], check=True)

    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")

    # ----- Run RAGEN training -----
    os.chdir(work_dir)
    print("\n🚀 STARTING RAGEN TRAINING ON WEBARENA", flush=True)
    result = subprocess.run(
        [sys.executable, "-u", "-m", "ragen.train_ragen", "--config", "configs/ragen_webarena.yaml", "--output_dir", "/outputs", "--skip-initial-eval"]
    )

    if result.returncode != 0:
        print("❌ Training failed.", flush=True)
        return {"status": "failed", "exit_code": result.returncode}

    print("\n🎉 SUCCESS!", flush=True)
    volume.commit()
    return {"status": "completed", "exit_code": 0}


# ----- LOCAL ENTRY POINT -----
@app.local_entrypoint()
def main():
    import tarfile, io
    from pathlib import Path

    print("\n🌐 Launching WebArena websites inside Modal cloud...")
    launch_webarena_sites.remote()

    print("\n📦 PACKAGING CODE...")
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add("tinyzero", arcname="tinyzero")
        tar.add("ragen", arcname="ragen")
        tar.add("setup.py", arcname="setup.py")

    code_tar = tar_buffer.getvalue()
    print(f"✓ Package size: {len(code_tar)/1024/1024:.1f} MB")

    with open("configs/ragen_webarena.yaml") as f:
        config_yaml = f.read()

    print("\n🚀 DEPLOYING RAGEN TRAINING TO MODAL...")
    result = train.remote(code_tar, config_yaml)

    if result.get("status") == "completed":
        print("\n🎉 SUCCESS! Retrieve outputs with:")
        print("modal volume get ragen-outputs /outputs ./ragen_results")
    else:
        print(f"\n⚠️ Failed: {result}")



if __name__ == "__main__":
    main()
