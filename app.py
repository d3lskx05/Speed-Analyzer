# app.py
import streamlit as st
import time
import os
import psutil
import requests
import json
import functools
import math
import gdown
import zipfile
import plotly.express as px

from sentence_transformers import SentenceTransformer
import torch
import plotly.express as px

# ---------- Helper utilities ----------

def sizeof_fmt(num, suffix="B"):
    # human readable
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

def get_ram_usage_mb():
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 2)

def path_size_bytes(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
            except Exception:
                pass
    return total

def try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

# ---------- HF API helpers ----------

def hf_model_info(model_name, token=None):
    """Try to get model metadata from HF api. Returns dict or None."""
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = f"https://huggingface.co/api/models/{model_name}"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

def hf_model_files(model_name, token=None):
    """Try to get list of repo files (and sizes) via HF API. Returns list or None."""
    # best-effort: use models/{model}/files endpoint (works for many public repos)
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        url = f"https://huggingface.co/api/models/{model_name}/revision/main/files"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ---------- Model loading (cached) ----------

@functools.lru_cache(maxsize=8)
def load_model_cached(path_or_name: str, from_hf: bool = True, hf_token: str = None):
    """Load SentenceTransformer model. from_hf True => HF repo id, else local folder path."""
    # If HF private token needed, pass use_auth_token in constructor
    if from_hf and hf_token:
        model = SentenceTransformer(path_or_name, use_auth_token=hf_token)
    else:
        model = SentenceTransformer(path_or_name)
    return model

# ---------- Introspection helpers ----------

def inspect_model_info(model, model_source_path=None):
    """
    Try to extract:
    - embedding dim
    - number of params
    - number of layers (if available)
    - batch optimization (True/False)
    - quantization support (bitsandbytes installed)
    - mixed precision support (fp16 GPU available)
    """
    info = {}
    # dim
    try:
        # do a single encode to get embedding dim if not obvious
        emb = model.encode("тест", convert_to_tensor=True)
        info["dim"] = int(emb.shape[-1])
    except Exception:
        info["dim"] = None

    # params
    try:
        total = 0
        for p in model.parameters():
            total += p.numel()
        info["params"] = int(total)
    except Exception:
        info["params"] = None

    # number of hidden layers - try to find transformer config
    num_layers = None
    try:
        # try several common attributes
        # some SentenceTransformer pipelines keep a huggingface model at model._first_module().auto_model
        first = None
        try:
            first = model._first_module()
        except Exception:
            # fallback: take first module of model._modules
            mods = list(model._modules.values())
            if mods:
                first = mods[0]
        if first is not None:
            auto_model = getattr(first, "auto_model", None) or getattr(first, "model", None) or getattr(first, "hf_model", None)
            if auto_model is not None:
                cfg = getattr(auto_model, "config", None)
                if cfg is not None:
                    num_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "num_layers", None)
    except Exception:
        num_layers = None
    info["num_layers"] = int(num_layers) if num_layers is not None else None

    # batch optimization - sentence-transformers supports batching
    info["batch_optimized"] = True

    # quantization support (bitsandbytes)
    bnb = try_import("bitsandbytes")
    info["quantization_bitsandbytes_available"] = bool(bnb)

    # mixed precision (fp16) support: check CUDA presence
    info["fp16_cuda_available"] = torch.cuda.is_available()

    # model local path (if loaded from local)
    try:
        local_dir = getattr(model, "cache_folder", None)
        info["model_cache_folder"] = local_dir
    except Exception:
        info["model_cache_folder"] = None

    return info

# ---------- Benchmarking ----------

def benchmark_model(model_name_or_path, source="HF", n_queries=10, text_length="short", hf_token=None):
    """Performs a set of measurements and returns a dict of metrics and metadata."""
    m = {}
    m["model_id_or_path"] = model_name_or_path
    m["source"] = source

    # HF metadata
    hf_meta = None
    if source == "HF":
        try:
            hf_meta = hf_model_info(model_name_or_path, token=hf_token)
        except Exception:
            hf_meta = None

    # start measuring load time and RAM
    t0 = time.time()
    try:
        model = load_model_cached(model_name_or_path, from_hf=(source=="HF"), hf_token=hf_token)
        load_time = time.time() - t0
        m["load_time_sec"] = round(load_time, 3)
    except Exception as e:
        m["error"] = f"Ошибка загрузки модели: {e}"
        return m

    # RAM after load
    try:
        m["ram_after_load_mb"] = round(get_ram_usage_mb(), 2)
    except Exception:
        m["ram_after_load_mb"] = None

    # model size (try local cache folder, else remote HF files)
    try:
        size_mb = None
        # prefer local model folder (if user gave local path or model cached)
        # if path exists and is dir:
        if source != "HF" and os.path.isdir(model_name_or_path):
            size_mb = path_size_bytes(model_name_or_path) / (1024 * 1024)
        else:
            # try to detect local cache folder reported by SentenceTransformer
            cache_folder = getattr(model, "cache_folder", None)
            if cache_folder:
                # construct probable subfolder name
                sub = model_name_or_path.replace("/", "_")
                candidate = os.path.join(cache_folder, sub)
                if os.path.isdir(candidate):
                    size_mb = path_size_bytes(candidate) / (1024 * 1024)
                else:
                    # maybe model placed directly in cache_folder
                    size_mb = path_size_bytes(cache_folder) / (1024 * 1024)
        # if still None, try via HF files API (best-effort)
        if size_mb is None and source == "HF":
            files = hf_model_files(model_name_or_path, token=hf_token)
            if isinstance(files, list):
                total = 0
                for f in files:
                    # file entries may include 'size' or not
                    sz = f.get("size", 0) if isinstance(f, dict) else 0
                    total += sz
                if total > 0:
                    size_mb = total / (1024 * 1024)
        m["model_size_mb"] = round(size_mb, 2) if size_mb else None
    except Exception:
        m["model_size_mb"] = None

    # inspect model internals
    try:
        info = inspect_model_info(model)
        m.update({
            "embedding_dim": info.get("dim"),
            "num_parameters": int(info.get("params")) if info.get("params") else None,
            "num_layers": info.get("num_layers"),
            "batch_optimized": info.get("batch_optimized"),
            "quantization_bitsandbytes_available": info.get("quantization_bitsandbytes_available"),
            "fp16_cuda_available": info.get("fp16_cuda_available")
        })
    except Exception:
        pass

    # HF meta fields (attempt)
    try:
        if hf_meta:
            m["hf_author"] = hf_meta.get("author")
            m["hf_lastModified"] = hf_meta.get("lastModified")
            # tags often contain languages or 'multilingual'
            tags = hf_meta.get("tags") or []
            # languages maybe present in 'pipeline_tag' or model card
            langs = hf_meta.get("languages") or []
            m["hf_tags"] = ", ".join(tags) if tags else None
            m["hf_languages"] = ", ".join(langs) if langs else None
    except Exception:
        pass

    # now perform encoding benchmarks
    if text_length == "short":
        sample_text = "Привет мир"
    elif text_length == "medium":
        sample_text = "Это тест для проверки скорости кодирования эмбеддингов моделью."
    else:
        sample_text = " ".join(["Длинный"] * 200)

    # warmup
    try:
        _ = model.encode("тёст", convert_to_tensor=True)
    except Exception:
        pass

    # single request
    try:
        t1 = time.time()
        _ = model.encode(sample_text, convert_to_tensor=True)
        t_single = time.time() - t1
        m["time_single_ms"] = round(t_single * 1000, 3)
    except Exception:
        m["time_single_ms"] = None

    # batch
    try:
        texts = [sample_text] * max(1, int(n_queries))
        t2 = time.time()
        _ = model.encode(texts, convert_to_tensor=True)
        t_batch = time.time() - t2
        m["time_batch_sec"] = round(t_batch, 3)
        m["avg_per_query_ms"] = round((t_batch / len(texts)) * 1000, 3)
    except Exception:
        m["time_batch_sec"] = None
        m["avg_per_query_ms"] = None

    # CPU load (short sample over encoding)
    try:
        cpu_before = psutil.cpu_percent(interval=None)
        # do a small encode
        _ = model.encode([sample_text] * 5, convert_to_tensor=True)
        cpu_after = psutil.cpu_percent(interval=None)
        m["cpu_percent_sample"] = round((cpu_before + cpu_after) / 2, 2)
    except Exception:
        m["cpu_percent_sample"] = None

    return m

# ---------- Optimization recommendations for Streamlit Free ----------

def optimization_tips(result):
    """
    Produce a text block of recommendations based on metrics and thresholds.
    """
    tips = []
    size = result.get("model_size_mb")
    params = result.get("num_parameters")
    ram = result.get("ram_after_load_mb")
    dim = result.get("embedding_dim")
    quant = result.get("quantization_bitsandbytes_available")
    fp16 = result.get("fp16_cuda_available")

    # thresholds suitable for Streamlit Free (approx)
    if size and size > 700:
        tips.append("• Модель большая (>700MB). На Streamlit Free возможны OOM/падения. Рекомендуется: distill/quantize/использовать более лёгкую модель.")
    elif size and size > 300:
        tips.append("• Модель среднего размера (300–700MB). Внимательно следи за RAM при загрузке и батчировании.")
    elif size:
        tips.append("• Модель компактная — должна загружаться и работать лучше на Free tier.")

    if params and params > 200_000_000:
        tips.append("• Много параметров (>200M) — высокая вероятность медленного инференса на CPU.")
    elif params and params > 80_000_000:
        tips.append("• Количество параметров умеренное (80–200M). Рекомендуется тестировать batch_size и throttle.")

    if ram and ram > 900:
        tips.append("• После загрузки модель занимает много RAM (>900MB). На Streamlit Free это критично — подумай о quantization или локальном хостинге.")
    if quant:
        tips.append("• bitsandbytes доступен — можно попробовать INT8/4bit квантование (уменьшит RAM и ускорит инференс).")
    else:
        tips.append("• bitsandbytes не установлен — квантование может быть недоступно. Для CPU можно попробовать `torch.quantization`/distillation.")

    if fp16:
        tips.append("• CUDA доступна — можно использовать mixed-precision (fp16) для ускорения на GPU.")
    else:
        tips.append("• CUDA недоступна — fp16 не поможет на этой машине.")

    if dim and dim >= 1024:
        tips.append("• Большой размер эмбеддинга (>=1024) — повышенная точность, но больше памяти в индексе (FAISS/pgvector).")
    elif dim and dim <= 384:
        tips.append("• Малый размер эмбеддинга (<=384) — экономит память и скорость, но точность может быть ниже.")

    # final advice
    if size and size > 700 or (ram and ram > 900) or (params and params > 200_000_000):
        tips.append("\nРекомендация: на бесплатном Streamlit лучше использовать модели < 500MB или использовать quantized/distilled версии. Также можно хранить эмбеддинги в persistent storage и не загружать модель на старте.")
    else:
        tips.append("\nРекомендация: модель подходит для Streamlit Free с осторожностью — протестируй нагрузку (N requests) и уменьшай batch_size при необходимости.")

    return "\n".join(tips)

# ---------- Streamlit UI ----------

st.set_page_config(layout="wide", page_title="Model Benchmark & Optimizer")

st.title("🔎 Model Benchmark & Optimizer (HF + GDrive)")

col1, col2 = st.columns([1, 2])

with col1:
    source = st.radio("Источник модели:", ["HuggingFace", "Local (GDrive / path)"])
    if source == "HuggingFace":
        model_input = st.text_input("HF model id:", value="deepvk/USER-bge-m3")
        use_token = st.checkbox("Приватный HF (use token from st.secrets['HUGGINGFACE_TOKEN'])", value=False)
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN") if use_token else None
    else:
        model_input = st.text_input("Путь к локальной модели (GDrive):", value="/content/drive/MyDrive/models/my_model")
        hf_token = None

    n_queries = st.number_input("Количество запросов (batch) для теста:", min_value=1, max_value=1000, value=10, step=1)
    text_len = st.selectbox("Длина текста:", ["short", "medium", "long"])
    run_btn = st.button("Запустить тест")

with col2:
    st.markdown("**Сохранённые результаты**")
    if "bench_results" not in st.session_state:
        st.session_state.bench_results = []

if run_btn:
    with st.spinner("Выполняется бенчмарк... Пожалуйста, подожди"):
        res = benchmark_model(model_input, source=("HF" if source=="HuggingFace" else "Local"), n_queries=n_queries, text_length=text_len, hf_token=hf_token)
    # append timestamp
    res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    st.session_state.bench_results.append(res)
    st.success("Тест завершён")
st.markdown("---")
st.subheader("⚖️ A/B тестирование моделей")

colA, colB = st.columns(2)

with colA:
    source_a = st.radio("Источник модели A:", ["HF", "GDrive"], key="sourceA")
    if source_a == "HF":
        modelA = st.text_input("Модель A (HF ID):", "deepvk/USER-bge-m3", key="modelA")
        hf_tokenA = st.text_input("HF Token для модели A (если приватная):", type="password", key="hfA")
    else:
        modelA_file_id = st.text_input("Google Drive File ID модели A:", "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf", key="gdriveA")

with colB:
    source_b = st.radio("Источник модели B:", ["HF", "GDrive"], key="sourceB")
    if source_b == "HF":
        modelB = st.text_input("Модель B (HF ID):", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", key="modelB")
        hf_tokenB = st.text_input("HF Token для модели B (если приватная):", type="password", key="hfB")
    else:
        modelB_file_id = st.text_input("Google Drive File ID модели B:", "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf", key="gdriveB")

n_queries_ab = st.number_input("Количество тестовых запросов для A/B:", min_value=1, max_value=500, value=10, key="n_queries_ab")
text_len_ab = st.selectbox("Длина текста для A/B:", ["short", "medium", "long"], key="text_len_ab")

# ---------- функции ----------
def download_gdrive_model(file_id, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "model.zip")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    return dest_folder

def normalize_result(res):
    """
    Приводим результат benchmark_model к плоскому словарю для DataFrame.
    Любые отсутствующие ключи заполняем None.
    """
    keys = ["load_time_sec","ram_after_load_mb","time_single_ms","time_batch_sec",
            "avg_per_query_ms","model_size_mb","embedding_dim","num_parameters"]
    if not res or "error" in res:
        return {k: None for k in keys}
    return {k: res.get(k, None) for k in keys}

# ---------- запуск ----------
if st.button("Запустить A/B тест"):
    with st.spinner("Выполняем A/B тест..."):
        # --- модель A ---
        try:
            if source_a == "GDrive":
                modelA_path = download_gdrive_model(modelA_file_id, "/tmp/modelA")
                resA = benchmark_model(modelA_path, source="Local", n_queries=n_queries_ab, text_length=text_len_ab)
            else:
                resA = benchmark_model(modelA, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenA)
        except Exception as e:
            resA = {"error": str(e)}

        # --- модель B ---
        try:
            if source_b == "GDrive":
                modelB_path = download_gdrive_model(modelB_file_id, "/tmp/modelB")
                resB = benchmark_model(modelB_path, source="Local", n_queries=n_queries_ab, text_length=text_len_ab)
            else:
                resB = benchmark_model(modelB, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenB)
        except Exception as e:
            resB = {"error": str(e)}

    # --- сохраняем в сессию ---
    st.session_state["AB_test"] = {"A": resA, "B": resB}

# ---------- вывод ----------
if st.session_state.get("AB_test"):
    resA_norm = normalize_result(st.session_state["AB_test"]["A"])
    resB_norm = normalize_result(st.session_state["AB_test"]["B"])
    
    df_ab = pd.DataFrame([resA_norm, resB_norm])
    df_ab.index = ["A","B"]
    st.markdown("### Результаты A/B теста")
    st.dataframe(df_ab)

    st.markdown("### Сравнение ключевых метрик")
    metrics = list(resA_norm.keys())
    diff = {}
    for m in metrics:
        a_val = resA_norm.get(m)
        b_val = resB_norm.get(m)
        diff[m] = {"A": a_val, "B": b_val, "diff (B-A)": (b_val - a_val) if a_val is not None and b_val is not None else None}
    st.dataframe(pd.DataFrame(diff).T)

    # Визуализация
    try:
        plot_df = pd.DataFrame([
            {"model":"A", **resA_norm},
            {"model":"B", **resB_norm}
        ])
        fig = px.bar(plot_df, x="model", y=["load_time_sec","time_batch_sec","ram_after_load_mb"],
                     barmode="group", title="Сравнение метрик моделей A vs B")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.write("Ошибка визуализации A/B:", e)

# display accumulated results
if st.session_state.get("bench_results"):
    df = st.session_state.bench_results.copy()

    # Normalize numeric fields for plotting (some may be None)
    for r in df:
        for k in ["load_time_sec", "model_size_mb", "ram_after_load_mb", "time_single_ms", "time_batch_sec", "avg_per_query_ms", "num_parameters"]:
            if k not in r:
                r[k] = None

    st.subheader("📋 Результаты тестов")
    st.dataframe(df)

    # Simple interactive filters
    st.subheader("📈 Графики")
    try:
        # build small DataFrame for plotting
        import pandas as pd
        plot_df = pd.DataFrame(df)
        # size vs batch time
        fig1 = px.scatter(plot_df, x="model_size_mb", y="time_batch_sec", size="ram_after_load_mb", color="model_id_or_path",
                          labels={"model_size_mb":"Model size (MB)", "time_batch_sec":f"Batch time (sec) for {n_queries} queries"},
                          title="Размер модели vs Время обработки батча")
        st.plotly_chart(fig1, use_container_width=True)

        # params vs single latency
        if "num_parameters" in plot_df.columns:
            fig2 = px.scatter(plot_df, x="num_parameters", y="time_single_ms", color="model_id_or_path",
                              labels={"num_parameters":"# parameters", "time_single_ms":"Single request (ms)"},
                              title="Параметры модели vs латентность одного запроса")
            st.plotly_chart(fig2, use_container_width=True)
    except Exception as e:
        st.write("Ошибка построения графиков:", e)

    # Show detailed info + optimization tips per model
    st.subheader("🛠 Оптимизация для Streamlit Free (рекомендации)")
    for r in df:
        st.markdown(f"### {r.get('model_id_or_path')}  —  {r.get('timestamp')}")
        # pretty printing selected metrics
        st.write({
            "Load time (s)": r.get("load_time_sec"),
            "Model size (MB)": r.get("model_size_mb"),
            "RAM after load (MB)": r.get("ram_after_load_mb"),
            "Embedding dim": r.get("embedding_dim"),
            "Num params": r.get("num_parameters"),
            "Num layers": r.get("num_layers"),
            "Time single (ms)": r.get("time_single_ms"),
            "Time batch (s)": r.get("time_batch_sec"),
            "Quantization available (bitsandbytes)": r.get("quantization_bitsandbytes_available"),
            "FP16 CUDA available": r.get("fp16_cuda_available"),
            "HF tags / languages": r.get("hf_tags", r.get("hf_languages"))
        })
        st.markdown("**Рекомендации:**")
        st.code(optimization_tips(r))

st.markdown("---")
st.markdown("### 📝 Примечания и советы")
st.markdown("""
- Для приватных HF моделей добавь токен в `st.secrets['HUGGINGFACE_TOKEN']` и поставь галочку 'Приватный HF'.
- Для Streamlit Free лучше использовать модели **< 500MB** (или quantized/distilled версии).  
- Если модель большая, лучше **предварительно вычислить эмбеддинги** для корпуса и хранить их в persistent storage (S3/DB/pgvector), чтобы не держать модель постоянно в памяти.
- Если у тебя нет GPU, устанавливать `bitsandbytes` не всегда полезно — но INT8-quantized версии иногда заметно снижают RAM.
- Тестируй разные `n_queries` и `text length` — поведение может меняться.
""")
