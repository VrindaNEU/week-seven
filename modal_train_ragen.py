"""
Deploy RAGEN with REAL WebShop to Modal - JAVA 21 FIX
"""
import modal

# Complete image with Java 21
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "git", 
        "wget", 
        "gnupg",
        "software-properties-common",
        "ca-certificates",
        "curl"
    )
    .run_commands(
        # Add Adoptium repository and install Java 21
        "mkdir -p /etc/apt/keyrings",
        "wget -O - https://packages.adoptium.net/artifactory/api/gpg/key/public | tee /etc/apt/keyrings/adoptium.asc",
        "echo 'deb [signed-by=/etc/apt/keyrings/adoptium.asc] https://packages.adoptium.net/artifactory/deb bookworm main' | tee /etc/apt/sources.list.d/adoptium.list",
        "apt-get update",
        "apt-get install -y temurin-21-jdk",
    )
    .pip_install(
        # Core ML stack
        "numpy>=1.25.0",
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "accelerate>=0.24.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        
        # WebShop dependencies
        "faiss-cpu>=1.7.0",
        "pyserini",
        "gym==0.24.0",
        "spacy>=3.6.0",
        "flask==2.1.2",
        "Werkzeug==2.0.3",
        "beautifulsoup4",
        "rank-bm25",
        "nltk",
        "cleantext",
        "requests",
        "scikit-learn",
        "pandas",
        "selenium",
        "gdown",
        "rich",
        "thefuzz",
        "python-Levenshtein",
    )
    .run_commands(
        "python -m spacy download en_core_web_sm"
    )
)

app = modal.App("ragen-webshop-real", image=image)
volume = modal.Volume.from_name("ragen-outputs", create_if_missing=True)

@app.function(
    gpu="H100",
    timeout=14400,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/outputs": volume},
)
def train(code_tar: bytes, config_yaml: str):
    """Run RAGEN training"""
    import subprocess
    import sys
    import os
    import tarfile
    import io
    
    print("="*60, flush=True)
    print("🤖 RAGEN + REAL WEBSHOP ON H100", flush=True)
    print("="*60, flush=True)
    
    work_dir = "/root/work"
    os.makedirs(work_dir, exist_ok=True)
    os.chdir(work_dir)
    
    print("\n📦 Extracting code...", flush=True)
    tar = tarfile.open(fileobj=io.BytesIO(code_tar))
    tar.extractall()
    tar.close()
    
    os.makedirs("configs", exist_ok=True)
    with open("configs/ragen_webshop.yaml", "w") as f:
        f.write(config_yaml)
    
    print("\n📥 Installing RAGEN...", flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
    
    print("\n📥 Setting up WebShop...", flush=True)
    
    if not os.path.exists("/root/WebShop"):
        subprocess.run([
            "git", "clone", 
            "https://github.com/princeton-nlp/WebShop.git", 
            "/root/WebShop"
        ], check=True)
    
    os.chdir("/root/WebShop")
    os.makedirs("data", exist_ok=True)
    
    print("\n📥 Downloading WebShop data...", flush=True)
    
    data_files = [
        ("1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib", "data/items_shuffle_1000.json"),
        ("1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu", "data/items_ins_v2_1000.json"),
        ("14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O", "data/items_human_ins.json"),
    ]
    
    for file_id, output_path in data_files:
        if not os.path.exists(output_path):
            subprocess.run([
                sys.executable, "-m", "gdown", file_id, "-O", output_path
            ], check=False)
    
    print("\n📥 Setting up WebShop indexes...", flush=True)
    packaged_indexes = os.path.join(work_dir, "webshop_indexes")

    if os.path.exists(packaged_indexes):
        print("✓ Using packaged pre-built indexes", flush=True)
        os.makedirs("search_engine", exist_ok=True)
        subprocess.run(["cp", "-r", packaged_indexes, "search_engine/indexes_100"], check=True)
    else:
        print("⚠️ No pre-built indexes found, building them now...", flush=True)
        
        # Ensure we're in WebShop directory
        os.chdir("/root/WebShop")
        os.makedirs("search_engine", exist_ok=True)
        
        # Import WebShop utilities
        sys.path.insert(0, "/root/WebShop")
        from web_agent_site.utils import DEFAULT_FILE_PATH
        from web_agent_site.engine.engine import load_products
        import json
        
        print("  Loading products...", flush=True)
        all_products, *_ = load_products(
            filepath=DEFAULT_FILE_PATH,
            num_products=100,
            human_goals=True
        )
        
        print("  Creating index documents...", flush=True)
        os.makedirs("search_engine/resources_100", exist_ok=True)
        
        docs = []
        for p in all_products:
            option_texts = []
            options = p.get('options', {})
            for option_name, option_contents in options.items():
                option_contents_text = ', '.join(option_contents)
                option_texts.append(f'{option_name}: {option_contents_text}')
            option_text = ', and '.join(option_texts)
            
            doc = {
                'id': p['asin'],
                'contents': ' '.join([
                    p['Title'],
                    p['Description'],
                    p['BulletPoints'][0] if p.get('BulletPoints') else '',
                    option_text,
                ]).lower()
            }
            docs.append(doc)
        
        with open('search_engine/resources_100/documents.jsonl', 'w') as f:
            for doc in docs:
                f.write(json.dumps(doc) + '\n')
        
        print("  Building Lucene index (this may take a few minutes)...", flush=True)
        subprocess.run([
            sys.executable, "-m", "pyserini.index.lucene",
            "--collection", "JsonCollection",
            "--input", "search_engine/resources_100",
            "--index", "search_engine/indexes_100",
            "--generator", "DefaultLuceneDocumentGenerator",
            "--threads", "1",
            "--storePositions", "--storeDocvectors", "--storeRaw"
        ], check=True, cwd="/root/WebShop")
        
        print("✓ Indexes built successfully!", flush=True)
    
    # Verify indexes exist
    index_path = "/root/WebShop/search_engine/indexes_100"
    if not os.path.exists(index_path):
        print(f"❌ Index not found at {index_path}", flush=True)
        return {"status": "failed", "exit_code": -1}
    else:
        print(f"✓ Index verified at {index_path}", flush=True)

    # Continue with the rest...
    sys.path.insert(0, "/root/WebShop")
    os.environ['PYTHONPATH'] = "/root/WebShop:" + os.environ.get('PYTHONPATH', '')

    os.chdir(work_dir)
    
    # Verify Java version
    print("\n🔍 Verifying environment...", flush=True)
    result = subprocess.run(["java", "-version"], capture_output=True, text=True)
    print(f"Java: {result.stderr.split('version')[1].split()[0] if 'version' in result.stderr else 'unknown'}", flush=True)
    
    for mod in ["faiss", "numpy", "torch", "spacy", "pyserini"]:
        try:
            m = __import__(mod)
            print(f"✓ {mod}: {getattr(m, '__version__', 'OK')}", flush=True)
        except Exception as e:
            print(f"❌ {mod}: {e}", flush=True)
            return {"status": "failed", "exit_code": -1}
    
    print("✓ Ready!", flush=True)
    
    # Train
    print("\n" + "="*60, flush=True)
    print("🚀 STARTING TRAINING", flush=True)
    print("="*60, flush=True)
    
    result = subprocess.run([
        sys.executable, "-u", "-m", "ragen.train_ragen",
        "--config", "configs/ragen_webshop.yaml",
        "--output_dir", "/outputs",
        "--skip-initial-eval"
    ])
    
    if result.returncode != 0:
        return {"status": "failed", "exit_code": result.returncode}
    
    print("\n🎉 SUCCESS!", flush=True)
    volume.commit()
    
    return {"status": "completed", "exit_code": 0}


@app.local_entrypoint()
def main():
    import tarfile, io
    from pathlib import Path
    
    print("\n📦 PACKAGING")
    
    if not Path("ragen").exists():
        print("❌ Run from week_06/!")
        return
    
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode='w:gz') as tar:
        tar.add('tinyzero', arcname='tinyzero')
        tar.add('ragen', arcname='ragen')
        tar.add('setup.py', arcname='setup.py')
        
        idx = Path("../WebShop/search_engine/indexes_100")
        if idx.exists():
            tar.add(str(idx), arcname='webshop_indexes')
    
    code_tar = tar_buffer.getvalue()
    print(f"✓ {len(code_tar)/1024/1024:.1f} MB")
    
    with open("configs/ragen_webshop.yaml") as f:
        config_yaml = f.read()
    
    print("\n🚀 DEPLOYING...")
    result = train.remote(code_tar, config_yaml)
    
    if result.get("status") == "completed":
        print("\n🎉 SUCCESS!")
        print("Download: modal volume get ragen-outputs /outputs ./ragen_results")
    else:
        print(f"\n⚠️ Failed: {result}")


if __name__ == "__main__":
    main()