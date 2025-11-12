"""
Deploy RAGEN with WebArena environment on Modal
"""

import modal

# ==========================================================
# === BUILD THE IMAGE (includes Java + WebArena deps) ===
# ==========================================================

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git",
        "wget",
        "curl",
        "ca-certificates",
        "software-properties-common",
        "gnupg",
    )
    .pip_install(
        # Core ML / RL stack
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        # WebArena + browser deps
        "gym==0.24.0",
        "beautifulsoup4",
        "flask==2.1.2",
        "Werkzeug==2.0.3",
        "requests",
        "rich",
        "pandas",
        "spacy>=3.6.0",
        "faiss-cpu",
        "pyserini",
        "thefuzz",
        "python-Levenshtein",
        "gdown",
        "beartype",
        "playwright",
        "browsergym[webarena]",
    )
    .run_commands(
        # Install Java 21 (required by pyserini)
        "mkdir -p /etc/apt/keyrings",
        "wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc",
        "echo 'deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb bookworm main' | tee /etc/apt/sources.list.d/adoptium.list",
        "apt-get update && apt-get install -y temurin-21-jdk",
        # Preinstall Playwright browser and SpaCy model
        "python -m playwright install --with-deps chromium",
        "python -m spacy download en_core_web_sm",
    )
)

app = modal.App("ragen-webarena", image=image)
volume = modal.Volume.from_name("ragen-outputs", create_if_missing=True)

# ==========================================================
# === TRAIN FUNCTION (runs in Modal container) ===
# ==========================================================

@app.function(
    gpu="H100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
)
def train(code_tar: bytes, config_yaml: str):
    """
    Run RAGEN training with WebArena environment inside Modal
    """
    import os, sys, io, tarfile, subprocess, shutil, json, re, time

    print("=" * 60)
    print("🤖 RAGEN + WebArena TRAINING SESSION")
    print("=" * 60)

    # --------------------------------------------------
    # Unpack the user's code
    # --------------------------------------------------
    work_dir = "/root/work"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)

    print("\n📦 Extracting your project...")
    tar = tarfile.open(fileobj=io.BytesIO(code_tar))
    tar.extractall()
    tar.close()

    # Write config YAML
    os.makedirs("configs", exist_ok=True)
    config_path = os.path.join(work_dir, "configs", "ragen_webarena.yaml")
    with open(config_path, "w") as f:
        f.write(config_yaml)
    print(f"✓ Config written to {config_path}")

    # --------------------------------------------------
    # Install your RAGEN package
    # --------------------------------------------------
    print("\n📥 Installing RAGEN...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)

    # --------------------------------------------------
    # WebArena setup (use bundled repo)
    # --------------------------------------------------
    webarena_dir = os.path.join(work_dir, "webarena")
    if not os.path.exists(webarena_dir):
        raise RuntimeError("❌ WebArena folder not found in code package!")

    sys.path.insert(0, webarena_dir)
    os.environ["PYTHONPATH"] = f"{webarena_dir}:{os.environ.get('PYTHONPATH', '')}"

    # Ensure WebArena config files exist
    cfg_dir = os.path.join(webarena_dir, "webarena", "config_files")
    if not os.path.exists(cfg_dir):
        alt = os.path.join(webarena_dir, "config_files")
        if os.path.exists(alt):
            shutil.copytree(alt, cfg_dir, dirs_exist_ok=True)
    print(f"✓ WebArena config files at: {cfg_dir}")

    # --------------------------------------------------
    # Patch URLs for localhost (in Modal)
    # --------------------------------------------------
    print("\n🩹 Updating config URLs to use 127.0.0.1 ...")
    for name in os.listdir(cfg_dir):
        if name.endswith(".json"):
            p = os.path.join(cfg_dir, name)
            try:
                text = open(p).read()
                text = re.sub(r"your_[a-zA-Z0-9_]+:([0-9/]+)", r"http://127.0.0.1:\1", text)
                open(p, "w").write(text)
            except Exception as e:
                print(f"⚠️ Could not rewrite {name}: {e}")
    print("✓ All config files updated.")

    # --------------------------------------------------
    # Launch WebArena sites locally
    # --------------------------------------------------
    print("\n🌐 Launching WebArena demo sites (localhost ports)...")

    ports = [7770, 7780, 8023, 8888, 9999, 3000, 4399]
    env_vars = os.environ.copy()
    env_vars.update({
        "SHOPPING": "http://127.0.0.1:7770",
        "SHOPPING_ADMIN": "http://127.0.0.1:7780/admin",
        "REDDIT": "http://127.0.0.1:9999",
        "GITLAB": "http://127.0.0.1:8023",
        "MAP": "http://127.0.0.1:3000",
        "WIKIPEDIA": "http://127.0.0.1:8888",
        "HOMEPAGE": "http://127.0.0.1:4399",
    })

    # Start Flask homepage as placeholder
    home_dir = os.path.join(webarena_dir, "webarena-homepage")
    if os.path.exists(home_dir):
        subprocess.Popen(["flask", "run", "--host=0.0.0.0", "--port=4399"], cwd=home_dir)
        time.sleep(3)

    print("✓ Local demo sites up (simulated)")

    # --------------------------------------------------
    # Run the actual training script
    # --------------------------------------------------
    print("\n🚀 STARTING RAGEN TRAINING ON WEBARENA")
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "-m",
            "ragen.train_ragen_webarena",
            "--config",
            config_path,
            "--output_dir",
            "/outputs",
            "--skip-initial-eval",
        ],
        cwd=webarena_dir,
        env=env_vars,
    )

    if result.returncode != 0:
        print(f"❌ Training failed with exit code {result.returncode}")
        return {"status": "failed", "exit_code": result.returncode}

    print("\n🎉 SUCCESS! Training completed.")
    volume.commit()
    return {"status": "completed", "exit_code": 0}


# ==========================================================
# === LOCAL ENTRYPOINT (packs your code + launches) ===
# ==========================================================

@app.local_entrypoint()
def main():
    import io, tarfile
    from pathlib import Path

    print("\n📦 PACKAGING PROJECT...")
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
        tar.add("tinyzero", arcname="tinyzero")
        tar.add("ragen", arcname="ragen")
        tar.add("webarena", arcname="webarena")
        tar.add("setup.py", arcname="setup.py")

    code_tar = tar_buffer.getvalue()
    print(f"✓ Package size: {len(code_tar)/1024/1024:.1f} MB")

    with open("configs/ragen_webarena.yaml") as f:
        config_yaml = f.read()

    print("\n🚀 DEPLOYING TO MODAL...")
    result = train.remote(code_tar, config_yaml)

    if result.get("status") == "completed":
        print("\n🎉 SUCCESS! Retrieve outputs with:")
        print("modal volume get ragen-outputs /outputs ./ragen_results")
    else:
        print(f"\n⚠️ Failed: {result}")
