# app.py
import streamlit as st
import time
import os
import psutil
import requests
import functools
import gdown
import zipfile
import pandas as pd
import plotly.express as px
from sentence_transformers import SentenceTransformer
import torch
import shutil

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

def try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None

# ---------- Model loading (cached) ----------

@functools.lru_cache(maxsize=8)
def load_model_cached(path_or_name: str, hf_token: str = None):
    if hf_token:
        model = SentenceTransformer(path_or_name, use_auth_token=hf_token)
    else:
        model = SentenceTransformer(path_or_name)
    return model

# ---------- Benchmarking ----------

def benchmark_model(model_name_or_path, source="HF", n_queries=10, text_length="short", hf_token=None):
    m = {}
    m["model_id_or_path"] = model_name_or_path
    m["source"] = source

    t0 = time.time()
    try:
        model = load_model_cached(model_name_or_path, hf_token=hf_token if source=="HF" else None)
        load_time = time.time() - t0
        m["load_time_sec"] = round(load_time,3)
    except Exception as e:
        m["error"] = str(e)
        return m

    m["ram_after_load_mb"] = round(get_ram_usage_mb(),2)
    try:
        sample_text = {"short":"Привет мир", "medium":"Это тест для проверки скорости кодирования эмбеддингов моделью.", "long":" ".join(["Длинный"]*200)}.get(text_length, "Привет мир")
        _ = model.encode(sample_text, convert_to_tensor=True)
        t1 = time.time()
        _ = model.encode(sample_text, convert_to_tensor=True)
        m["time_single_ms"] = round((time.time()-t1)*1000,3)
        texts = [sample_text]*max(1,int(n_queries))
        t2 = time.time()
        _ = model.encode(texts, convert_to_tensor=True)
        t_batch = time.time()-t2
        m["time_batch_sec"] = round(t_batch,3)
        m["avg_per_query_ms"] = round((t_batch/len(texts))*1000,3)
    except Exception:
        m["time_single_ms"]=m["time_batch_sec"]=m["avg_per_query_ms"]=None

    # embedding dim and params
    try:
        emb = model.encode("тест", convert_to_tensor=True)
        m["embedding_dim"] = int(emb.shape[-1])
    except: m["embedding_dim"] = None

    try:
        total = 0
        for p in model.parameters(): total+=p.numel()
        m["num_parameters"]=int(total)
    except: m["num_parameters"]=None

    # quantization support
    m["quantization_bitsandbytes_available"] = bool(try_import("bitsandbytes"))
    m["fp16_cuda_available"] = torch.cuda.is_available()

    return m

# ---------- GDrive helper ----------

def download_gdrive_model(file_id, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    zip_path = os.path.join(dest_folder, "model.zip")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=zip_path, quiet=False)
    with zipfile.ZipFile(zip_path,'r') as zip_ref:
        zip_ref.extractall(dest_folder)
    return dest_folder

# ---------- Cleanup with logging ----------

def cleanup_tmp_with_logging(folders=["/tmp/modelA","/tmp/modelB"]):
    for folder in folders:
        if os.path.exists(folder):
            try:
                start_ram = psutil.virtual_memory().used
                t0 = time.time()
                shutil.rmtree(folder)
                t1 = time.time()
                end_ram = psutil.virtual_memory().used
                freed = abs(end_ram - start_ram)
                st.info(f"Удалена папка {folder}. Время: {round(t1-t0,2)}s, освобождено памяти: {sizeof_fmt(freed)}")
            except Exception as e:
                st.warning(f"Не удалось удалить временную папку {folder}: {e}")

# ---------- Streamlit UI ----------

st.set_page_config(layout="wide", page_title="Model Benchmark & Optimizer")
st.title("🔎 Model Benchmark & Optimizer (HF + GDrive)")

mode = st.radio("Режим работы:", ["Single", "A/B тест"])

if "bench_results" not in st.session_state:
    st.session_state.bench_results = []

if mode=="Single":
    source = st.radio("Источник модели:", ["HuggingFace", "GDrive"])
    if source=="HuggingFace":
        model_input = st.text_input("HF model id:", value="deepvk/USER-bge-m3")
        use_token = st.checkbox("Приватный HF (use token from st.secrets['HUGGINGFACE_TOKEN'])", value=False)
        hf_token = st.secrets.get("HUGGINGFACE_TOKEN") if use_token else None
    else:
        model_input = st.text_input("Google Drive File ID модели:", "1bZoykt0Sj2GRPvLRC_3Z7Wt8g4AGT34R")
        hf_token = None

    n_queries = st.number_input("Количество запросов (batch) для теста:", min_value=1,max_value=1000,value=10,step=1)
    text_len = st.selectbox("Длина текста:", ["short","medium","long"])
    run_btn = st.button("Запустить тест")

    if run_btn:
        with st.spinner("Выполняется бенчмарк..."):
            if source=="GDrive":
                model_path = download_gdrive_model(model_input,"/tmp/modelSingle")
                res = benchmark_model(model_path, source="Local", n_queries=n_queries, text_length=text_len)
                cleanup_tmp_with_logging(["/tmp/modelSingle"])
            else:
                res = benchmark_model(model_input, source="HF", n_queries=n_queries, text_length=text_len, hf_token=hf_token)
            res["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            st.session_state.bench_results.append(res)
        st.success("Тест завершён")

elif mode=="A/B тест":
    colA,colB = st.columns(2)
    with colA:
        source_a = st.radio("Источник модели A:", ["HF","GDrive"], key="sourceA")
        if source_a=="HF":
            modelA = st.text_input("Модель A (HF ID):","deepvk/USER-bge-m3", key="modelA")
            hf_tokenA = st.text_input("HF Token модели A (если приватная):", type="password", key="hfA")
        else:
            modelA_file_id = st.text_input("Google Drive File ID модели A:","1bZoykt0Sj2GRPvLRC_3Z7Wt8g4AGT34R", key="gdriveA")

    with colB:
        source_b = st.radio("Источник модели B:", ["HF","GDrive"], key="sourceB")
        if source_b=="HF":
            modelB = st.text_input("Модель B (HF ID):","sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", key="modelB")
            hf_tokenB = st.text_input("HF Token модели B (если приватная):", type="password", key="hfB")
        else:
            modelB_file_id = st.text_input("Google Drive File ID модели B:","1bZoykt0Sj2GRPvLRC_3Z7Wt8g4AGT34R", key="gdriveB")

    n_queries_ab = st.number_input("Количество тестовых запросов:", min_value=1,max_value=500,value=10,key="n_queries_ab")
    text_len_ab = st.selectbox("Длина текста:", ["short","medium","long"], key="text_len_ab")
    run_ab_btn = st.button("Запустить A/B тест")

    if run_ab_btn:
        with st.spinner("Выполняем A/B тест..."):
            # --- модель A ---
            try:
                if source_a=="GDrive":
                    modelA_path = download_gdrive_model(modelA_file_id,"/tmp/modelA")
                    resA = benchmark_model(modelA_path, source="Local", n_queries=n_queries_ab, text_length=text_len_ab)
                else:
                    resA = benchmark_model(modelA, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenA)
            except Exception as e:
                resA = {"error": str(e)}

            # --- модель B ---
            try:
                if source_b=="GDrive":
                    modelB_path = download_gdrive_model(modelB_file_id,"/tmp/modelB")
                    resB = benchmark_model(modelB_path, source="Local", n_queries=n_queries_ab, text_length=text_len_ab)
                else:
                    resB = benchmark_model(modelB, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenB)
            except Exception as e:
                resB = {"error": str(e)}

            # append results
            for r in [resA,resB]:
                r["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                st.session_state.bench_results.append(r)

            cleanup_tmp_with_logging(["/tmp/modelA","/tmp/modelB"])
        st.success("A/B тест завершён")

# ---------- вывод всех результатов ----------
if st.session_state.get("bench_results"):
    df = pd.DataFrame(st.session_state.bench_results)
    st.subheader("📋 Результаты тестов")
    st.dataframe(df)

    # простые графики
    try:
        fig1 = px.scatter(df, x="num_parameters", y="time_single_ms", color="model_id_or_path",
                          labels={"num_parameters":"# parameters","time_single_ms":"Single request (ms)"},
                          title="Параметры модели vs латентность одного запроса")
        st.plotly_chart(fig1,use_container_width=True)
    except Exception as e:
        st.write("Ошибка построения графиков:",e)

st.markdown("---")
st.markdown("### 📝 Примечания")
st.markdown("""
- Для приватных HF моделей добавь токен в `st.secrets['HUGGINGFACE_TOKEN']`.
- Для GDrive достаточно File ID.
- Все результаты Single и A/B теста сохраняются в одном списке.
""")
