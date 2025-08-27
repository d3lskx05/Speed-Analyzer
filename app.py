# app.py
import streamlit as st
import time, os, shutil, zipfile, requests, functools
import pandas as pd
import plotly.express as px
import psutil
from sentence_transformers import SentenceTransformer
import torch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

# ---------- Helper utils ----------
def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"

def get_ram_usage_mb():
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)

def path_size_bytes(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except Exception:
                pass
    return total

def try_import(name):
    try: return __import__(name)
    except Exception: return None

# ---------- HF helpers ----------
def hf_model_info(model_name, token=None):
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(f"https://huggingface.co/api/models/{model_name}", headers=headers, timeout=15)
        if r.status_code == 200: return r.json()
    except: pass
    return None

def hf_model_files(model_name, token=None):
    try:
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        r = requests.get(f"https://huggingface.co/api/models/{model_name}/revision/main/files", headers=headers, timeout=15)
        if r.status_code == 200: return r.json()
    except: pass
    return None

# ---------- Cached model loader ----------
@functools.lru_cache(maxsize=8)
def load_model_cached(path_or_name: str, from_hf=True, hf_token=None):
    if from_hf and hf_token: return SentenceTransformer(path_or_name, use_auth_token=hf_token)
    return SentenceTransformer(path_or_name)

# ---------- Benchmark ----------
def benchmark_model(model_name_or_path, source="HF", n_queries=10, text_length="short", hf_token=None):
    m={}
    m["model_id_or_path"]=model_name_or_path
    m["source"]=source
    hf_meta = hf_model_info(model_name_or_path, token=hf_token) if source=="HF" else None
    t0 = time.time()
    try:
        model = load_model_cached(model_name_or_path, from_hf=(source=="HF"), hf_token=hf_token)
        m["load_time_sec"]=round(time.time()-t0,3)
    except Exception as e:
        m["error"]=f"Ошибка загрузки: {e}"; return m

    m["ram_after_load_mb"]=round(get_ram_usage_mb(),2)
    # model size
    try:
        size_mb=None
        if source!="HF" and os.path.isdir(model_name_or_path):
            size_mb=path_size_bytes(model_name_or_path)/(1024*1024)
        else:
            cache_folder=getattr(model,"cache_folder",None)
            if cache_folder: size_mb=path_size_bytes(cache_folder)/(1024*1024)
        if size_mb is not None: m["model_size_mb"]=round(size_mb,2)
    except: m["model_size_mb"]=None

    # embedding dim & params
    try:
        m["embedding_dim"]=int(model.encode("тест", convert_to_tensor=True).shape[-1])
    except: m["embedding_dim"]=None
    try:
        total=0
        for p in model.parameters(): total+=p.numel()
        m["num_parameters"]=int(total)
    except: m["num_parameters"]=None
    # batch timings
    sample_text = {"short":"Привет мир","medium":"Тест скорости кодирования эмбеддингов.","long":" ".join(["Длинный"]*200)}.get(text_length,"Привет")
    try:
        t1=time.time(); model.encode(sample_text, convert_to_tensor=True); m["time_single_ms"]=round((time.time()-t1)*1000,3)
    except: m["time_single_ms"]=None
    try:
        texts=[sample_text]*max(1,int(n_queries))
        t2=time.time(); model.encode(texts, convert_to_tensor=True)
        t_batch=time.time()-t2
        m["time_batch_sec"]=round(t_batch,3); m["avg_per_query_ms"]=round((t_batch/len(texts))*1000,3)
    except: m["time_batch_sec"]=m["avg_per_query_ms"]=None
    # CPU
    try: cpu_before=psutil.cpu_percent(interval=None); model.encode([sample_text]*5, convert_to_tensor=True); cpu_after=psutil.cpu_percent(interval=None); m["cpu_percent_sample"]=round((cpu_before+cpu_after)/2,2)
    except: m["cpu_percent_sample"]=None
    # HF metadata
    if hf_meta:
        m["hf_author"]=hf_meta.get("author"); m["hf_lastModified"]=hf_meta.get("lastModified")
        m["hf_tags"]=", ".join(hf_meta.get("tags",[])); m["hf_languages"]=", ".join(hf_meta.get("languages",[]))
    m["timestamp"]=time.strftime("%Y-%m-%d %H:%M:%S")
    return m

# ---------- Optimization tips ----------
def optimization_tips(result):
    tips=[]
    if result.get("model_size_mb",0)>700: tips.append("• Модель >700MB — осторожно на Free tier.")
    elif result.get("model_size_mb",0)>300: tips.append("• Модель 300–700MB — умеренно.")
    else: tips.append("• Компактная модель — быстро.")
    if result.get("num_parameters",0)>200_000_000: tips.append("• >200M параметров — возможен медленный инференс.")
    elif result.get("num_parameters",0)>80_000_000: tips.append("• 80–200M параметров — средняя нагрузка.")
    if result.get("ram_after_load_mb",0)>900: tips.append("• RAM после загрузки >900MB.")
    if result.get("embedding_dim",0)>=1024: tips.append("• Большой размер эмбеддингов — точность выше.")
    elif result.get("embedding_dim",0)<=384: tips.append("• Малый размер эмбеддингов — экономия памяти.")
    return "\n".join(tips)

# ---------- GDrive download ----------
def download_gdrive_model(file_id, dest_folder):
    import gdown; os.makedirs(dest_folder, exist_ok=True)
    zip_path=os.path.join(dest_folder,"model.zip")
    url=f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output=zip_path, quiet=True)
    with zipfile.ZipFile(zip_path,'r') as zip_ref: zip_ref.extractall(dest_folder)
    return dest_folder

# ---------- Cleanup ----------
def cleanup_tmp(folder_list):
    for folder in folder_list:
        if os.path.exists(folder):
            try: shutil.rmtree(folder)
            except: pass

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Model Benchmark & Optimizer")
st.title("🔎 Model Benchmark & Optimizer (HF + GDrive)")

mode=st.radio("Режим работы:", ["Single","A/B тест"])
# Reset results when mode changes
if st.session_state.get("last_mode")!=mode:
    st.session_state.bench_results=[]
    st.session_state.AB_test={}
st.session_state.last_mode=mode

# ---------- Single ----------
if mode=="Single":
    col1,col2=st.columns([1,2])
    with col1:
        source=st.radio("Источник модели:", ["HuggingFace","GDrive"])
        if source=="HuggingFace":
            model_input=st.text_input("HF model id:", value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            use_token=st.checkbox("Приватная HF", value=False)
            hf_token=st.secrets.get("HUGGINGFACE_TOKEN") if use_token else None
        else:
            model_input=st.text_input("GDrive File ID модели:", value="1bZoykt0Sj2GRPvLRC_3Z7Wt8g4AGT34R")
            hf_token=None
        n_queries=st.number_input("Количество запросов:", min_value=1,max_value=1000,value=10)
        text_len=st.selectbox("Длина текста:", ["short","medium","long"])
        run_btn=st.button("Запустить тест")

    if run_btn:
        with st.spinner("Выполняется бенчмарк..."):
            cleanup_tmp(["/tmp/model_single"])
            try:
                if source=="GDrive":
                    model_path=download_gdrive_model(model_input,f"/tmp/{model_input}")
                    res=benchmark_model(model_path, source="Local", n_queries=n_queries, text_length=text_len)
                else:
                    res=benchmark_model(model_input, source="HF", n_queries=n_queries, text_length=text_len, hf_token=hf_token)
                st.session_state.bench_results.append(res)
                st.success("Тест завершён")
            except Exception as e: st.error(f"Ошибка бенчмарка: {e}")

# ---------- A/B ----------
if mode=="A/B тест":
    colA,colB=st.columns(2)
    with colA:
        source_a=st.radio("Источник модели A:", ["HF","GDrive"], key="sourceA")
        modelA=st.text_input("Модель A:", value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", key="modelA")
        hf_tokenA=st.text_input("HF Token A:", type="password", key="hfA") if source_a=="HF" else None
    with colB:
        source_b=st.radio("Источник модели B:", ["HF","GDrive"], key="sourceB")
        modelB=st.text_input("Модель B:", value="sentence-transformers/all-MiniLM-L6-v2", key="modelB")
        hf_tokenB=st.text_input("HF Token B:", type="password", key="hfB") if source_b=="HF" else None
    n_queries_ab=st.number_input("Количество запросов:", min_value=1,max_value=500,value=10,key="n_queries_ab")
    text_len_ab=st.selectbox("Длина текста:", ["short","medium","long"], key="text_len_ab")
    if st.button("Запустить A/B тест"):
        with st.spinner("Выполняем A/B тест..."):
            cleanup_tmp([f"/tmp/{modelA}", f"/tmp/{modelB}"])
            try:
                resA = benchmark_model(download_gdrive_model(modelA,f"/tmp/{modelA}"), source="Local", n_queries=n_queries_ab, text_length=text_len_ab) if source_a=="GDrive" else benchmark_model(modelA, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenA)
                resB = benchmark_model(download_gdrive_model(modelB,f"/tmp/{modelB}"), source="Local", n_queries=n_queries_ab, text_length=text_len_ab) if source_b=="GDrive" else benchmark_model(modelB, source="HF", n_queries=n_queries_ab, text_length=text_len_ab, hf_token=hf_tokenB)
                st.session_state.AB_test={"A":resA,"B":resB}
                st.success("A/B тест завершён")
            except Exception as e: st.error(f"Ошибка A/B: {e}")

# ---------- Display results ----------
def display_results(df, title="Результаты тестов"):
    st.subheader(title)
    st.dataframe(df)
    try:
        fig1=px.scatter(df,x="model_size_mb",y="time_batch_sec",size="ram_after_load_mb",color="model_id_or_path",
                        labels={"model_size_mb":"Model size (MB)","time_batch_sec":"Batch time (s)"}, title="Размер модели vs Время обработки батча")
        st.plotly_chart(fig1,use_container_width=True)
        fig2=px.scatter(df,x="num_parameters",y="time_single_ms",color="model_id_or_path",
                        labels={"num_parameters":"# parameters","time_single_ms":"Single request (ms)"}, title="Параметры модели vs Single latency")
        st.plotly_chart(fig2,use_container_width=True)
    except: pass
    st.subheader("🛠 Рекомендации")
    for r in df.to_dict(orient="records"):
        st.markdown(f"### {r.get('model_id_or_path')} — {r.get('timestamp')}")
        st.code(optimization_tips(r))

if st.session_state.bench_results:
    display_results(pd.DataFrame(st.session_state.bench_results), "📋 История Single тестов")
if st.session_state.get("AB_test"):
    resA=st.session_state["AB_test"]["A"]; resB=st.session_state["AB_test"]["B"]
    df_ab=pd.DataFrame([resA,resB]); df_ab.index=["A","B"]
    st.subheader("📋 История A/B тестов"); st.dataframe(df_ab)

# ---------- CSV / PDF download ----------
if st.session_state.bench_results or st.session_state.get("AB_test"):
    col_down1,col_down2=st.columns(2)
    with col_down1:
        if st.button("Скачать CSV"):
            all_res=st.session_state.bench_results[:]
            if st.session_state.get("AB_test"): all_res+=[st.session_state["AB_test"]["A"],st.session_state["AB_test"]["B"]]
            pd.DataFrame(all_res).to_csv("benchmark_results.csv", index=False, encoding="utf-8-sig")
            st.download_button("Скачать CSV", data=open("benchmark_results.csv","rb").read(), file_name="benchmark_results.csv")
    with col_down2:
        if st.button("Скачать PDF"):
            doc=SimpleDocTemplate("benchmark_report.pdf", pagesize=A4)
            styles=getSampleStyleSheet(); elems=[]
            if st.session_state.bench_results:
                for r in st.session_state.bench_results:
                    elems.append(Paragraph(f"{r.get('model_id_or_path')} — {r.get('timestamp')}", styles["Heading3"]))
                    elems.append(Paragraph(optimization_tips(r), styles["Normal"]))
                    elems.append(Spacer(1,12))
            if st.session_state.get("AB_test"):
                for label,res in st.session_state["AB_test"].items():
                    elems.append(Paragraph(f"{label}: {res.get('model_id_or_path')}", styles["Heading3"]))
                    elems.append(Paragraph(optimization_tips(res), styles["Normal"]))
                    elems.append(Spacer(1,12))
            doc.build(elems)
            st.download_button("Скачать PDF", data=open("benchmark_report.pdf","rb").read(), file_name="benchmark_report.pdf")
