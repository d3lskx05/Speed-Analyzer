# app.py
import streamlit as st
import time
import os
import psutil
import requests
import functools
import shutil
import gdown
import zipfile
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import torch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ---------- Helper utilities ----------
def sizeof_fmt(num, suffix="B"):
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
    if from_hf and hf_token:
        model = SentenceTransformer(path_or_name, use_auth_token=hf_token)
    else:
        model = SentenceTransformer(path_or_name)
    return model

# ---------- Introspection helpers ----------
def inspect_model_info(model):
    info = {}
    try:
        emb = model.encode("тест", convert_to_tensor=True)
        info["dim"] = int(emb.shape[-1])
    except Exception:
        info["dim"] = None
    try:
        total = 0
        for p in model.parameters():
            total += p.numel()
        info["params"] = int(total)
    except Exception:
        info["params"] = None
    num_layers = None
    try:
        first = None
        try:
            first = model._first_module()
        except Exception:
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
    info["batch_optimized"] = True
    bnb = try_import("bitsandbytes")
    info["quantization_bitsandbytes_available"] = bool(bnb)
    info["fp16_cuda_available"] = torch.cuda.is_available()
    try:
        info["model_cache_folder"] = getattr(model, "cache_folder", None)
    except Exception:
        info["model_cache_folder"] = None
    return info

# ---------- Benchmarking ----------
def benchmark_model(model_name_or_path, source="HF", n_queries=10, text_length="short", hf_token=None):
    m = {}
    m["model_id_or_path"] = model_name_or_path
    m["source"] = source
    hf_meta = None
    if source == "HF":
        try:
            hf_meta = hf_model_info(model_name_or_path, token=hf_token)
        except Exception:
            hf_meta = None
    t0 = time.time()
    try:
        model = load_model_cached(model_name_or_path, from_hf=(source=="HF"), hf_token=hf_token)
        load_time = time.time() - t0
        m["load_time_sec"] = round(load_time, 3)
    except Exception as e:
        m["error"] = f"Ошибка загрузки модели: {e}"
        return m
    try:
        m["ram_after_load_mb"] = round(get_ram_usage_mb(), 2)
    except Exception:
        m["ram_after_load_mb"] = None
    try:
        size_mb = None
        if source != "HF" and os.path.isdir(model_name_or_path):
            size_mb = path_size_bytes(model_name_or_path) / (1024 * 1024)
        else:
            cache_folder = getattr(model, "cache_folder", None)
            if cache_folder:
                sub = model_name_or_path.replace("/", "_")
                candidate = os.path.join(cache_folder, sub)
                if os.path.isdir(candidate):
                    size_mb = path_size_bytes(candidate) / (1024 * 1024)
                else:
                    size_mb = path_size_bytes(cache_folder) / (1024 * 1024)
        if size_mb is None and source == "HF":
            files = hf_model_files(model_name_or_path, token=hf_token)
            if isinstance(files, list):
                total = 0
                for f in files:
                    sz = f.get("size", 0) if isinstance(f, dict) else 0
                    total += sz
                if total > 0:
                    size_mb = total / (1024 * 1024)
        m["model_size_mb"] = round(size_mb, 2) if size_mb else None
    except Exception:
        m["model_size_mb"] = None
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
    try:
        if hf_meta:
            m["hf_author"] = hf_meta.get("author")
            m["hf_lastModified"] = hf_meta.get("lastModified")
            tags = hf_meta.get("tags") or []
            langs = hf_meta.get("languages") or []
            m["hf_tags"] = ", ".join(tags) if tags else None
            m["hf_languages"] = ", ".join(langs) if langs else None
    except Exception:
        pass
    if text_length == "short":
        sample_text = "Привет мир"
    elif text_length == "medium":
        sample_text = "Это тест для проверки скорости кодирования эмбеддингов моделью."
    else:
        sample_text = " ".join(["Длинный"] * 200)
    try:
        _ = model.encode("тёст", convert_to_tensor=True)
    except Exception:
        pass
    try:
        t1 = time.time()
        _ = model.encode(sample_text, convert_to_tensor=True)
        t_single = time.time() - t1
        m["time_single_ms"] = round(t_single * 1000, 3)
    except Exception:
        m["time_single_ms"] = None
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
    try:
        cpu_before = psutil.cpu_percent(interval=None)
        _ = model.encode([sample_text] * 5, convert_to_tensor=True)
        cpu_after = psutil.cpu_percent(interval=None)
        m["cpu_percent_sample"] = round((cpu_before + cpu_after) / 2, 2)
    except Exception:
        m["cpu_percent_sample"] = None
    return m

# ---------- Optimization recommendations ----------
def optimization_tips(result):
    tips = []
    size = result.get("model_size_mb")
    params = result.get("num_parameters")
    ram = result.get("ram_after_load_mb")
    dim = result.get("embedding_dim")
    quant = result.get("quantization_bitsandbytes_available")
    fp16 = result.get("fp16_cuda_available")
    if size and size > 700:
        tips.append("• Модель большая (>700MB).")
    elif size and size > 300:
        tips.append("• Модель среднего размера (300–700MB).")
    elif size:
        tips.append("• Модель компактная — лучше на Free tier.")
    if params and params > 200_000_000:
        tips.append("• Много параметров (>200M).")
    elif params and params > 80_000_000:
        tips.append("• Количество параметров умеренное (80–200M).")
    if ram and ram > 900:
        tips.append("• После загрузки модель занимает много RAM (>900MB).")
    if quant:
        tips.append("• bitsandbytes доступен — можно INT8/4bit квантование.")
    else:
        tips.append("• bitsandbytes не установлен — квантование может быть недоступно.")
    if fp16:
        tips.append("• CUDA доступна — можно использовать mixed-precision (fp16).")
    else:
        tips.append("• CUDA недоступна — fp16 не поможет.")
    if dim and dim >= 1024:
        tips.append("• Большой размер эмбеддинга (>=1024).")
    elif dim and dim <= 384:
        tips.append("• Малый размер эмбеддинга (<=384).")
    if size and size > 700 or (ram and ram > 900) or (params and params > 200_000_000):
        tips.append("\nРекомендация: использовать модели <500MB или quantized/distilled.")
    return "\n".join(tips)

# ---------- GDrive download ----------
def download_gdrive_model(file_id, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "model.zip")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    return dest_folder

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Model Benchmark & Optimizer")
st.title("🔎 Model Benchmark & Optimizer (HF + GDrive)")

# ---------- режимы ----------
mode = st.radio("Режим работы:", ["Single", "A/B"], index=0)
if "last_mode" not in st.session_state or st.session_state["last_mode"] != mode:
    st.session_state.bench_results = []
    st.session_state.AB_test = {}
st.session_state["last_mode"] = mode

# ---------- Single ----------
if mode == "Single":
    st.subheader("⚙️ Single тест модели")
    col1, col2 = st.columns([1,2])
    with col1:
        source = st.radio("Источник модели:", ["HuggingFace", "GDrive"])
        if source == "HuggingFace":
            model_input = st.text_input("HF model id:", value="deepvk/USER-bge-m3")
            use_token = st.checkbox("Приватный HF (использовать токен)", value=False)
            hf_token = st.secrets.get("HUGGINGFACE_TOKEN") if use_token else None
        else:
            model_input = st.text_input("GDrive File ID модели:", value="1bZoykt0Sj2GRPvLRC_3Z7Wt8g4AGT34R")
            hf_token = None
        n_queries = st.number_input("Количество запросов (batch):", min_value=1, max_value=1000, value=10, step=1)
        text_len = st.selectbox("Длина текста:", ["short","medium","long"])
        run_btn = st.button("Запустить тест")
    with col2:
        st.markdown("**История результатов**")
    if run_btn:
        st.session_state.bench_results = []
        with st.spinner("Выполняется Single тест..."):
            if source=="GDrive":
                model_path = download_gdrive_model(model_input, f"/tmp/{model_input}")
                res = benchmark_model(model_path, source="Local", n_queries=n_queries, text_length=text_len)
            else:
                res = benchmark_model(model_input, source="HF", n_queries=n_queries, text_length=text_len, hf_token=hf_token)
        res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        res["mode"] = "Single"
        st.session_state.bench_results.append(res)
        st.success("Single тест завершён")

# ---------- A/B ----------
else:
    st.subheader("⚖️ A/B тест моделей")
    colA,colB = st.columns(2)
    with colA:
        source_a = st.radio("Источник модели A:", ["HF","GDrive"], key="sourceA")
        if source_a=="HF":
            modelA = st.text_input("Модель A (HF ID):", "deepvk/USER-bge-m3", key="modelA")
            hf_tokenA = st.text_input("HF Token модели A:", type="password", key="hfA")
        else:
            modelA_file_id = st.text_input("GDrive File ID модели A:", "1bZoykt0Sj2GRPvLRC_3Z7Wt8g4AGT34R", key="gdriveA")
    with colB:
        source_b = st.radio("Источник модели B:", ["HF","GDrive"], key="sourceB")
        if source_b=="HF":
            modelB = st.text_input("Модель B (HF ID):", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", key="modelB")
            hf_tokenB = st.text_input("HF Token модели B:", type="password", key="hfB")
        else:
            modelB_file_id = st.text_input("GDrive File ID модели B:", "1bZoykt0Sj2GRPvLRC_3Z7Wt8g4AGT34R", key="gdriveB")
    n_queries_ab = st.number_input("Количество тестовых запросов:", min_value=1, max_value=500, value=10, key="n_queries_ab")
    text_len_ab = st.selectbox("Длина текста:", ["short","medium","long"], key="text_len_ab")
    run_ab = st.button("Запустить A/B тест")
    if run_ab:
        st.session_state.bench_results = []
        try:
            if source_a=="GDrive":
                modelA_path = download_gdrive_model(modelA_file_id, f"/tmp/{modelA_file_id}")
                resA = benchmark_model(modelA_path, source="Local", n_queries=n_queries_ab, text_length=text_len_ab)
            else:
                resA = benchmark_model(modelA, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenA)
        except Exception as e:
            resA = {"error": str(e)}
        try:
            if source_b=="GDrive":
                modelB_path = download_gdrive_model(modelB_file_id, f"/tmp/{modelB_file_id}")
                resB = benchmark_model(modelB_path, source="Local", n_queries=n_queries_ab, text_length=text_len_ab)
            else:
                resB = benchmark_model(modelB, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenB)
        except Exception as e:
            resB = {"error": str(e)}
        resA["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        resB["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        resA["mode"]="A"
        resB["mode"]="B"
        st.session_state.bench_results.extend([resA,resB])
        st.success("A/B тест завершён")

# ---------- Отображение истории + CSV/PDF ----------
if st.session_state.get("bench_results"):
    st.subheader("📋 История результатов (краткая сводка)")
    df_summary = pd.DataFrame(st.session_state.bench_results)
    display_cols = ["mode","model_id_or_path","timestamp","load_time_sec","time_batch_sec","ram_after_load_mb","embedding_dim","num_parameters"]
    st.dataframe(df_summary[display_cols])
    csv_bytes = df_summary.to_csv(index=False).encode()
    st.download_button("Скачать CSV", csv_bytes, file_name="bench_results.csv")
    if st.button("Очистить историю"):
        st.session_state.bench_results = []
    if st.button("Скачать PDF"):
        pdf_path = "/tmp/bench_report.pdf"
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        elements=[]
        styles = getSampleStyleSheet()
        elements.append(Paragraph("Benchmark Report", styles['Title']))
        elements.append(Spacer(1,12))
        for r in st.session_state.bench_results:
            elements.append(Paragraph(f"Модель: {r.get('model_id_or_path')} ({r.get('mode')})", styles['Heading2']))
            for k,v in r.items():
                elements.append(Paragraph(f"{k}: {v}", styles['Normal']))
            tips = optimization_tips(r)
            elements.append(Paragraph("Рекомендации:", styles['Heading3']))
            elements.append(Paragraph(tips, styles['Normal']))
            elements.append(Spacer(1,12))
        doc.build(elements)
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        st.download_button("Скачать PDF с аналитикой", pdf_bytes, file_name="bench_report.pdf")
