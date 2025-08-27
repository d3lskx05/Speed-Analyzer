import os
import shutil
import tempfile
import time
import psutil
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import gdown
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Model Performance Analyzer", layout="wide")

# -------------------------------
# Инициализация сессии
# -------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# -------------------------------
# Функции
# -------------------------------
def download_model_from_gdrive(file_id: str) -> str:
    """Загрузка модели с Google Drive по ID"""
    model_path = f"/tmp/{file_id}"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        url = f"https://drive.google.com/uc?id={file_id}"
        output = os.path.join(model_path, "model.zip")
        gdown.download(url, output, quiet=False)
        shutil.unpack_archive(output, model_path)
        os.remove(output)
    return model_path

def load_model(source: str, model_id: str) -> SentenceTransformer:
    """Загрузка модели: HuggingFace или GDrive"""
    if source == "HuggingFace":
        return SentenceTransformer(model_id)
    elif source == "Google Drive":
        return SentenceTransformer(download_model_from_gdrive(model_id))
    else:
        raise ValueError("Неверный источник модели")

def measure_performance(model, sentences):
    """Вычисление скорости, памяти и точности"""
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_mem = process.memory_info().rss / 1024 / 1024  # MB

    embeddings = model.encode(sentences, convert_to_tensor=False)
    end_time = time.time()
    end_mem = process.memory_info().rss / 1024 / 1024

    duration = end_time - start_time
    mem_used = end_mem - start_mem
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return {
        "Время (сек)": round(duration, 4),
        "Память (MB)": round(mem_used, 4),
        "Сходство": round(similarity, 4)
    }

def plot_metric(history_df, metric_name):
    """График изменения метрики"""
    fig, ax = plt.subplots()
    ax.plot(history_df["Дата"], history_df[metric_name], marker="o")
    ax.set_title(f"{metric_name} по тестам")
    ax.set_xlabel("Дата")
    ax.set_ylabel(metric_name)
    st.pyplot(fig)

def clear_tmp_dir(model_path):
    """Удаление временной папки модели и логирование"""
    if os.path.exists(model_path):
        size_before = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(model_path) for f in filenames)
        shutil.rmtree(model_path)
        st.write(f"🗑 Папка {model_path} удалена, освобождено {size_before/1024/1024:.2f} MB")

def save_pdf_report(result, recommendations, img_paths):
    """Генерация PDF отчета"""
    file_path = "/tmp/report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Отчет по тесту модели", styles["Title"]), Spacer(1, 12)]

    # Таблица с результатами
    data = [["Метрика", "Значение"]] + [[k, v] for k, v in result.items()]
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 12))

    # Рекомендации
    story.append(Paragraph("Рекомендации:", styles["Heading2"]))
    for rec in recommendations:
        story.append(Paragraph(f"- {rec}", styles["Normal"]))

    doc.build(story)
    return file_path

# -------------------------------
# Интерфейс
# -------------------------------
st.title("🔍 Анализ производительности модели")

# Очистка предыдущей аналитики при новом тесте
st.subheader("1. Настройка теста")
source = st.selectbox("Источник модели", ["HuggingFace", "Google Drive"])
model_id = st.text_input("ID модели или GDrive File ID")
sentences = st.text_area("Введите 2 предложения через Enter", "Привет мир\nЗдравствуйте мир")

if st.button("🚀 Запустить тест"):
    st.session_state["analytics"] = {}  # Очистка старой аналитики

    with st.spinner("Загрузка модели..."):
        model = load_model(source, model_id)

    st.success("✅ Модель загружена!")

    with st.spinner("Выполняется тестирование..."):
        lines = sentences.split("\n")
        result = measure_performance(model, lines[:2])

    # Добавляем в историю
    st.session_state["history"].append({
        "Дата": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Модель": model_id,
        **result
    })

    # Аналитика
    st.markdown("### 📊 Результаты теста")
    st.write(result)

    # Графики (если есть история)
    if len(st.session_state["history"]) > 1:
        history_df = pd.DataFrame(st.session_state["history"])
        for metric in ["Время (сек)", "Память (MB)", "Сходство"]:
            plot_metric(history_df, metric)

    # Рекомендации
    recommendations = []
    if result["Время (сек)"] > 2:
        recommendations.append("Модель слишком медленная для Streamlit Free.")
    if result["Память (MB)"] > 200:
        recommendations.append("Высокое потребление памяти — попробуйте quantization.")
    if not recommendations:
        recommendations.append("Модель подходит для использования.")

    st.markdown("### ✅ Рекомендации")
    for rec in recommendations:
        st.write(f"- {rec}")

    # Кнопка скачать PDF
    pdf_path = save_pdf_report(result, recommendations, [])
    with open(pdf_path, "rb") as f:
        st.download_button("📥 Скачать отчет (PDF)", f, file_name="report.pdf")

    # Очистка памяти
    if source == "Google Drive":
        clear_tmp_dir(f"/tmp/{model_id}")

# -------------------------------
# История тестов
# -------------------------------
st.markdown("### 🗂 История тестов")
if st.session_state["history"]:
    df_history = pd.DataFrame(st.session_state["history"])
    st.dataframe(df_history)
