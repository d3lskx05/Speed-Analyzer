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
import pandas as pd
import shutil

from sentence_transformers import SentenceTransformer
import torch

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

# ---------- Introspection ----------
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
        pass
    info["num_layers"] = int(num_layers) if num_layers is not None else None
    info["batch_optimized"] = True
    info["quantization_bitsandbytes_available"] = bool(try_import("bitsandbytes"))
    info["fp16_cuda_available"] = torch.cuda.is_available()
    try:
        local_dir = getattr(model, "cache_folder", None)
        info["model_cache_folder"] = local_dir
    except Exception:
        info["model_cache_folder"] = None
    return info

# ---------- Benchmark ----------
def benchmark_model(model_name_or_path, source="HF", n_queries=10, text_length="short", hf_token=None):
    m = {}
    m["model_id_or_path"] = model_name_or_path
    m["source"] = source
    hf_meta = None
    if source == "HF":
        hf_meta = hf_model_info(model_name_or_path, token=hf_token)
    t0 = time.time()
    try:
        model = load_model_cached(model_name_or_path, from_hf=(source=="HF"), hf_token=hf_token)
        m["load_time_sec"] = round(time.time()-t0, 3)
    except Exception as e:
        m["error"] = str(e)
        return m
    try:
        m["ram_after_load_mb"] = round(get_ram_usage_mb(), 2)
    except Exception:
        m["ram_after_load_mb"] = None
    # model size
    try:
        size_mb = None
        if source != "HF" and os.path.isdir(model_name_or_path):
            size_mb = path_size_bytes(model_name_or_path) / (1024 * 1024)
        else:
            cache_folder = getattr(model, "cache_folder", None)
            if cache_folder:
                size_mb = path_size_bytes(cache_folder) / (1024*1024)
        if size_mb is None and source=="HF":
            files = hf_model_files(model_name_or_path, token=hf_token)
            if isinstance(files, list):
                total = sum(f.get("size",0) if isinstance(f,dict) else 0 for f in files)
                size_mb = total/(1024*1024)
        m["model_size_mb"] = round(size_mb,2) if size_mb else None
    except Exception:
        m["model_size_mb"] = None
    # model internals
    try:
        info = inspect_model_info(model)
        m.update({
            "embedding_dim": info.get("dim"),
            "num_parameters": info.get("params"),
            "num_layers": info.get("num_layers"),
            "batch_optimized": info.get("batch_optimized"),
            "quantization_bitsandbytes_available": info.get("quantization_bitsandbytes_available"),
            "fp16_cuda_available": info.get("fp16_cuda_available")
        })
    except Exception:
        pass
    if hf_meta:
        try:
            m["hf_author"] = hf_meta.get("author")
            m["hf_lastModified"] = hf_meta.get("lastModified")
            m["hf_tags"] = ", ".join(hf_meta.get("tags") or [])
            m["hf_languages"] = ", ".join(hf_meta.get("languages") or [])
        except Exception:
            pass
    sample_text = {"short":"Привет мир", "medium":"Это тест для проверки скорости кодирования эмбеддингов моделью.", "long":" ".join(["Длинный"]*200)}.get(text_length,"Привет мир")
    try: _ = model.encode("тест", convert_to_tensor=True)
    except: pass
    try:
        t1 = time.time()
        _ = model.encode(sample_text, convert_to_tensor=True)
        m["time_single_ms"] = round((time.time()-t1)*1000,3)
    except: m["time_single_ms"] = None
    try:
        texts = [sample_text]*max(1,int(n_queries))
        t2 = time.time()
        _ = model.encode(texts, convert_to_tensor=True)
        t_batch = time.time()-t2
        m["time_batch_sec"] = round(t_batch,3)
        m["avg_per_query_ms"] = round((t_batch/len(texts))*1000,3)
    except:
        m["time_batch_sec"]=None
        m["avg_per_query_ms"]=None
    try:
        cpu_before = psutil.cpu_percent(interval=None)
        _ = model.encode([sample_text]*5, convert_to_tensor=True)
        cpu_after = psutil.cpu_percent(interval=None)
        m["cpu_percent_sample"] = round((cpu_before+cpu_after)/2,2)
    except:
        m["cpu_percent_sample"] = None
    return m

# ---------- Optimization tips ----------
def optimization_tips(result):
    tips=[]
    size = result.get("model_size_mb")
    params = result.get("num_parameters")
    ram = result.get("ram_after_load_mb")
    dim = result.get("embedding_dim")
    quant = result.get("quantization_bitsandbytes_available")
    fp16 = result.get("fp16_cuda_available")
    if size and size>700: tips.append("• Модель >700MB. На Streamlit Free возможны OOM.")
    elif size and size>300: tips.append("• Модель 300–700MB, следи за RAM.")
    else: tips.append("• Модель компактная, подходит для Free tier.")
    if params and params>200_000_000: tips.append("• >200M параметров, инференс медленный на CPU.")
    elif params and params>80_000_000: tips.append("• 80–200M параметров, тестируй batch_size.")
    if ram and ram>900: tips.append("• RAM после загрузки >900MB, критично для Free.")
    if quant: tips.append("• bitsandbytes доступен — можно использовать INT8/4bit квантование.")
    if fp16: tips.append("• CUDA доступна — mixed-precision ускорит.")
    return "\n".join(tips)

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Model Benchmark & Optimizer")
st.title("🔎 Model Benchmark & Optimizer (HF + GDrive)")

mode = st.radio("Выберите режим работы:", ["Single", "A/B"])

# ---------- Shared inputs ----------
if mode=="Single":
    st.subheader("⚡ Single model test")
    source = st.radio("Источник модели:", ["HuggingFace", "GDrive/Local"])
    if source=="HuggingFace":
        model_input = st.text_input("HF model id:", value="deepvk/USER-bge-m3")
        use_token = st.checkbox("Приватный HF (use token from st.secrets['HUGGINGFACE_TOKEN'])", value=False)
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN") if use_token else None
    else:
        model_input = st.text_input("Путь к локальной модели / GDrive:", value="/content/drive/MyDrive/models/my_model")
        hf_token=None
    n_queries = st.number_input("Количество запросов (batch) для теста:", min_value=1, max_value=1000, value=10)
    text_len = st.selectbox("Длина текста:", ["short","medium","long"])
    run_btn = st.button("Запустить тест Single")

    if run_btn:
        with st.spinner("Выполняется тест..."):
            res = benchmark_model(model_input, source=("HF" if source=="HuggingFace" else "Local"), n_queries=n_queries, text_length=text_len, hf_token=hf_token)
            res["timestamp"]=time.strftime("%Y-%m-%d %H:%M:%S")
            if "bench_results" not in st.session_state:
                st.session_state.bench_results=[]
            st.session_state.bench_results.append(res)
            st.success("Тест завершён")

if mode=="A/B":
    st.subheader("⚖️ A/B тестирование моделей")
    colA,colB = st.columns(2)
    with colA:
        source_a = st.radio("Источник модели A:", ["HF","GDrive"], key="sourceA")
        if source_a=="HF":
            modelA = st.text_input("Модель A (HF ID):","deepvk/USER-bge-m3", key="modelA")
            hf_tokenA = st.text_input("HF Token модели A (если приватная):", type="password", key="hfA")
        else:
            modelA_file_id = st.text_input("Google Drive File ID модели A:", "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf", key="gdriveA")
    with colB:
        source_b = st.radio("Источник модели B:", ["HF","GDrive"], key="sourceB")
        if source_b=="HF":
            modelB = st.text_input("Модель B (HF ID):","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", key="modelB")
            hf_tokenB = st.text_input("HF Token модели B (если приватная):", type="password", key="hfB")
        else:
            modelB_file_id = st.text_input("Google Drive File ID модели B:", "1RR15OMLj9vfSrVa1HN-dRU-4LbkdbRRf", key="gdriveB")
    n_queries_ab = st.number_input("Количество тестовых запросов:", min_value=1,max_value=500,value=10,key="n_queries_ab")
    text_len_ab = st.selectbox("Длина текста:", ["short","medium","long"], key="text_len_ab")
    if st.button("Запустить A/B тест"):
        with st.spinner("Выполняем A/B тест..."):
            def download_gdrive_model(file_id,dest_folder):
                os.makedirs(dest_folder, exist_ok=True)
                zip_path=os.path.join(dest_folder,"model.zip")
                url=f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, output=zip_path, quiet=False)
                with zipfile.ZipFile(zip_path,'r') as zip_ref:
                    zip_ref.extractall(dest_folder)
                return dest_folder
            def normalize_result(res):
                keys=["load_time_sec","ram_after_load_mb","time_single_ms","time_batch_sec","avg_per_query_ms","model_size_mb","embedding_dim","num_parameters"]
                if not res or "error" in res:
                    return {k:None for k in keys}
                return {k:res.get(k,None) for k in keys}
            def cleanup_tmp():
                for folder in ["/tmp/modelA","/tmp/modelB"]:
                    if os.path.exists(folder):
                        size = path_size_bytes(folder)/(1024*1024)
                        t0=time.time()
                        try:
                            shutil.rmtree(folder)
                            st.info(f"Папка {folder} удалена за {round(time.time()-t0,2)}s, освободилось {round(size,2)} MB")
                        except Exception as e:
                            st.warning(f"Не удалось удалить {folder}: {e}")
            # модель A
            try:
                if source_a=="GDrive":
                    modelA_path=download_gdrive_model(modelA_file_id,"/tmp/modelA")
                    resA=benchmark_model(modelA_path, source="Local", n_queries=n_queries_ab, text_length=text_len_ab)
                else:
                    resA=benchmark_model(modelA, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenA)
            except Exception as e:
                resA={"error":str(e)}
            # модель B
            try:
                if source_b=="GDrive":
                    modelB_path=download_gdrive_model(modelB_file_id,"/tmp/modelB")
                    resB=benchmark_model(modelB_path, source="Local", n_queries=n_queries_ab, text_length=text_len_ab)
                else:
                    resB=benchmark_model(modelB, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenB)
            except Exception as e:
                resB={"error":str(e)}
            st.session_state["AB_test"]={"A":resA,"B":resB}
            cleanup_tmp()

# ---------- Display accumulated results ----------
st.subheader("📋 Сохранённые результаты")
all_results=[]
if "bench_results" in st.session_state:
    all_results.extend(st.session_state.bench_results)
if "AB_test" in st.session_state:
    all_results.append(st.session_state.AB_test["A"])
    all_results.append(st.session_state.AB_test["B"])
if all_results:
    st.dataframe(all_results)
