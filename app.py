# app.py
import streamlit as st
import time
import os
import psutil
import requests
import functools
import gdown
import zipfile
import shutil
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import torch

# Словарь с русскими названиями колонок
COLUMN_NAMES_RU = {
    "model_id_or_path": "Модель",
    "source": "Источник",
    "load_time_sec": "Время загрузки (сек)",
    "ram_after_load_mb": "RAM после загрузки (МБ)",
    "model_size_mb": "Размер модели (МБ)",
    "embedding_dim": "Размер эмбеддинга",
    "num_parameters": "Параметров",
    "num_layers": "Слоёв",
    "batch_optimized": "Оптимизация батчей",
    "quantization_bitsandbytes_available": "Доступно квантование",
    "fp16_cuda_available": "FP16 CUDA",
    "hf_author": "Автор HF",
    "hf_lastModified": "Последнее изменение HF",
    "hf_tags": "Теги HF",
    "hf_languages": "Языки HF",
    "time_single_ms": "Время 1 запроса (мс)",
    "time_batch_sec": "Время батча (сек)",
    "avg_per_query_ms": "Среднее время (мс)",
    "cpu_percent_sample": "CPU (%)",
    "timestamp": "Время теста"
}
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
        hf_meta = hf_model_info(model_name_or_path, token=hf_token)
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
            size_mb = path_size_bytes(model_name_or_path) / (1024*1024)
        else:
            cache_folder = getattr(model, "cache_folder", None)
            if cache_folder:
                sub = model_name_or_path.replace("/", "_")
                candidate = os.path.join(cache_folder, sub)
                if os.path.isdir(candidate):
                    size_mb = path_size_bytes(candidate) / (1024*1024)
                else:
                    size_mb = path_size_bytes(cache_folder) / (1024*1024)
        if size_mb is None and source == "HF":
            files = hf_model_files(model_name_or_path, token=hf_token)
            if isinstance(files, list):
                total = sum([f.get("size",0) if isinstance(f, dict) else 0 for f in files])
                size_mb = total / (1024*1024) if total>0 else None
        m["model_size_mb"] = round(size_mb,2) if size_mb else None
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

    if hf_meta:
        try:
            m["hf_author"] = hf_meta.get("author")
            m["hf_lastModified"] = hf_meta.get("lastModified")
            tags = hf_meta.get("tags") or []
            langs = hf_meta.get("languages") or []
            m["hf_tags"] = ", ".join(tags) if tags else None
            m["hf_languages"] = ", ".join(langs) if langs else None
        except Exception:
            pass

    if text_length=="short":
        sample_text="Привет мир"
    elif text_length=="medium":
        sample_text="Это тест для проверки скорости кодирования эмбеддингов моделью."
    else:
        sample_text=" ".join(["Длинный"]*200)

    try:
        _ = model.encode("тёст", convert_to_tensor=True)
    except Exception:
        pass
    try:
        t1 = time.time()
        _ = model.encode(sample_text, convert_to_tensor=True)
        t_single = time.time() - t1
        m["time_single_ms"] = round(t_single*1000,3)
    except Exception:
        m["time_single_ms"] = None

    try:
        texts=[sample_text]*max(1,int(n_queries))
        t2=time.time()
        _=model.encode(texts, convert_to_tensor=True)
        t_batch=time.time()-t2
        m["time_batch_sec"]=round(t_batch,3)
        m["avg_per_query_ms"]=round((t_batch/len(texts))*1000,3)
    except Exception:
        m["time_batch_sec"]=None
        m["avg_per_query_ms"]=None

    try:
        cpu_before=psutil.cpu_percent(interval=None)
        _ = model.encode([sample_text]*5, convert_to_tensor=True)
        cpu_after=psutil.cpu_percent(interval=None)
        m["cpu_percent_sample"]=round((cpu_before+cpu_after)/2,2)
    except Exception:
        m["cpu_percent_sample"]=None

    return m

# ---------- Optimization tips ----------

def optimization_tips(result):
    tips=[]
    size=result.get("model_size_mb")
    params=result.get("num_parameters")
    ram=result.get("ram_after_load_mb")
    dim=result.get("embedding_dim")
    quant=result.get("quantization_bitsandbytes_available")
    fp16=result.get("fp16_cuda_available")

    if size and size>700:
        tips.append("• Модель большая (>700MB). На Streamlit Free возможны OOM/падения.")
    elif size and size>300:
        tips.append("• Модель среднего размера (300–700MB). Внимательно следи за RAM.")
    elif size:
        tips.append("• Модель компактная — должна загружаться и работать лучше на Free tier.")

    if params and params>200_000_000:
        tips.append("• Много параметров (>200M) — высокая вероятность медленного инференса на CPU.")
    elif params and params>80_000_000:
        tips.append("• Количество параметров умеренное (80–200M). Тестируй batch_size.")

    if ram and ram>900:
        tips.append("• После загрузки модель занимает много RAM (>900MB).")
    if quant:
        tips.append("• bitsandbytes доступен — можно попробовать INT8/4bit квантование.")
    else:
        tips.append("• bitsandbytes не установлен — квантование может быть недоступно.")
    if fp16:
        tips.append("• CUDA доступна — можно использовать mixed-precision (fp16).")
    else:
        tips.append("• CUDA недоступна — fp16 не поможет на этой машине.")
    if dim and dim>=1024:
        tips.append("• Большой размер эмбеддинга (>=1024) — повышенная точность, больше памяти.")
    elif dim and dim<=384:
        tips.append("• Малый размер эмбеддинга (<=384) — экономит память и скорость, точность ниже.")

    if size and size>700 or (ram and ram>900) or (params and params>200_000_000):
        tips.append("\nРекомендация: использовать модели <500MB или quantized/distilled версии.")
    else:
        tips.append("\nРекомендация: модель подходит для Streamlit Free с осторожностью.")

    return "\n".join(tips)

# ---------- Download GDrive ----------

def download_gdrive_model(file_id, dest_folder):
    dest_folder = f"/tmp/{file_id}"  # исправлено
    os.makedirs(dest_folder, exist_ok=True)
    zip_path=os.path.join(dest_folder,"model.zip")
    url=f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    return dest_folder

# ---------- Cleanup tmp with logging ----------

def cleanup_tmp(folder_list):
    for folder in folder_list:
        if os.path.exists(folder):
            start=time.time()
            size_before=path_size_bytes(folder)
            try:
                shutil.rmtree(folder)
                elapsed=time.time()-start
                st.info(f"Очистка {folder}: {sizeof_fmt(size_before)}, время {elapsed:.2f}s")
            except Exception as e:
                st.warning(f"Не удалось удалить {folder}: {e}")

# ---------- Streamlit UI ----------

st.set_page_config(layout="wide", page_title="Model Benchmark & Optimizer")
st.title("🔎 Model Benchmark & Optimizer (HF + GDrive)")

# Только Single режим
col1,col2=st.columns([1,2])
with col1:
    source=st.radio("Источник модели:", ["HuggingFace","GDrive"])
    if source=="HuggingFace":
        model_input=st.text_input("HF model id:", value="deepvk/USER-bge-m3")
        use_token=st.checkbox("Приватный HF (use token from st.secrets['HUGGINGFACE_TOKEN'])", value=False)
        hf_token=st.secrets.get("HUGGINGFACE_TOKEN") if use_token else None
    else:
        model_input=st.text_input("GDrive File ID модели:", value="1bZoykt0Sj2GRPvLRC_3Z7Wt8g4AGT34R")
        hf_token=None
    n_queries=st.number_input("Количество запросов (batch) для теста:", min_value=1,max_value=1000,value=10,step=1)
    text_len=st.selectbox("Длина текста:", ["short","medium","long"])
    run_btn=st.button("Запустить тест")

with col2:
    st.markdown("**Сохранённые результаты**")
    if "bench_results" not in st.session_state:
        st.session_state.bench_results=[]

if run_btn:
    # НЕ очищаем историю, просто добавляем новый результат
    if "bench_results" not in st.session_state:
        st.session_state.bench_results = []

    with st.spinner("Выполняется бенчмарк..."):
        try:
            start_ram = get_ram_usage_mb()
            start_time = time.time()

            if source=="GDrive":
                model_path=download_gdrive_model(model_input,f"/tmp/{model_input}")
                res=benchmark_model(model_path, source="Local", n_queries=n_queries, text_length=text_len)
            else:
                res=benchmark_model(model_input, source="HF", n_queries=n_queries, text_length=text_len, hf_token=hf_token)

            end_time = time.time()
            end_ram = get_ram_usage_mb()

            res["timestamp"]=time.strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.bench_results.append(res)

            st.success("Тест завершён")
            st.info(f"RAM до: {start_ram:.2f} MB, после: {end_ram:.2f} MB, рост: {end_ram - start_ram:.2f} MB, время: {end_time-start_time:.2f}s")
            cleanup_tmp([f"/tmp/{model_input}"])
        except Exception as e:
            st.error(f"Ошибка бенчмарка: {e}")

# ---------- Display results ----------
if "bench_results" in st.session_state and st.session_state.bench_results:
    st.subheader("📋 Результаты Single тестов")
    df=pd.DataFrame(st.session_state.bench_results)
    df_display = df.rename(columns=COLUMN_NAMES_RU)
    st.dataframe(df_display)

    # Выбор тестов для сравнения
    selected_rows = st.multiselect(
        "Выберите тесты для сравнения:",
        options=df.index.tolist(),
        format_func=lambda x: f"{df.loc[x,'model_id_or_path']} - {df.loc[x,'timestamp']}"
    )

    if st.button("Сравнить выбранные") and len(selected_rows) >= 2:
        compare_df = df.loc[selected_rows]
        st.dataframe(compare_df)
        try:
            fig=px.bar(compare_df, x="model_id_or_path", y=["load_time_sec","time_batch_sec","ram_after_load_mb"],
                       barmode="group", title="Сравнение выбранных моделей")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Ошибка построения графика: {e}")

    # Рекомендации по каждой модели
    st.subheader("🛠 Рекомендации")
    for r in st.session_state.bench_results:
        st.markdown(f"### {r.get('model_id_or_path')} — {r.get('timestamp')}")
        st.write({COLUMN_NAMES_RU.get(k, k): v for k, v in r.items()})
        st.markdown("**Рекомендации:**")
        st.code(optimization_tips(r))
