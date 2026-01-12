# 🎥 Context-Aware Video Intelligence using Retrieval-Augmented Generation (RAG)

A **Video Intelligence system** that enables **grounded question answering and summarization** over both **local videos** and **YouTube videos** by combining **video transcription, semantic retrieval, and large language models** using a **Retrieval-Augmented Generation (RAG)** pipeline.

This project ensures **hallucination-free responses** by strictly generating answers **only from retrieved video context with timestamps**.

---

## 🚀 Key Features

* 📹 **Supports both Local Videos & YouTube Videos**
* 🗣️ **Automatic transcript extraction with timestamps**
* 🔎 **Semantic search over video content using FAISS**
* 🧠 **RAG-based question answering & summarization**
* ⏱️ **Timestamp-grounded answers for verification**
* 🚫 **Strict hallucination control via prompt constraints**

---

## 🧩 Project Architecture (High-Level)

```
Video / YouTube URL
        ↓
Transcript Extraction (Whisper / YouTube API)
        ↓
Timestamp-aware Chunking
        ↓
Embedding Generation
        ↓
FAISS Vector Store
        ↓
Retriever (Top-K Relevant Chunks)
        ↓
LLM (Context-Constrained Generation)
        ↓
Answer + Summary + Timestamps
```

---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Frameworks:** LangChain
* **Transcription:** Faster-Whisper, YouTube Transcript API
* **Embeddings:** Sentence-Transformers (all-MiniLM-L6-v2)
* **Vector Store:** FAISS
* **LLMs:** Google Gemma 2B / Flan-T5
* **Video Processing:** FFmpeg
* **Environment:** Google Colab / Local Python

---

## 📂 Repository Structure

```
RAG/
│
├── RAG-General Video Project.py        # RAG pipeline for local video files
├── rag_using_langchain.ipynb           # YouTube video RAG (LangChain-based)
├── youtube_rag.ipynb                   # Timestamp-aware YouTube QA
├── general_video_rag.ipynb             # Local video QA & summarization
├── README.md                           # Project documentation
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/shivavadtyavath/RAG.git
cd RAG
```

### 2️⃣ Install Dependencies

```bash
pip install langchain langchain-community langchain-huggingface \
            faiss-cpu sentence-transformers \
            faster-whisper youtube-transcript-api \
            transformers torch ffmpeg-python
```

> ⚠️ Ensure **FFmpeg** is installed on your system.

---

## ▶️ How It Works

### 🔹 Step 1: Transcript Extraction

* **Local videos:** Audio extracted using FFmpeg → transcribed with Faster-Whisper
* **YouTube videos:** Captions fetched using YouTube Transcript API

Each transcript segment retains:

```json
{
  "text": "...",
  "start": 120.5,
  "end": 134.8
}
```

---

### 🔹 Step 2: Chunking

* Transcript is split into **overlapping semantic chunks**
* Each chunk preserves **start & end timestamps**

---

### 🔹 Step 3: Vector Indexing

* Text chunks converted into embeddings
* Stored in **FAISS vector database** for fast similarity search

---

### 🔹 Step 4: Retrieval

* User query is embedded
* Top-K most relevant chunks retrieved based on similarity

---

### 🔹 Step 5: Generation (RAG)

* Retrieved chunks passed as **strict context** to the LLM
* Prompt enforces:

  * ✅ Use only retrieved context
  * ❌ No external knowledge
  * ⏱️ Output timestamps

---

## 💬 Example Query

**Input**

```text
Is quantization discussed in the video? Summarize it.
```

**Output**

```text
YES.
Quantization is explained as a method to reduce model precision while
maintaining performance.

Timestamps:
01:19 – 02:40
02:34 – 03:15
```

---

## 🎯 Use Cases

* 🎓 Educational video understanding
* 🧑‍💻 Technical interview preparation
* 📚 Long lecture summarization
* 🔍 Timestamp-based video search
* 🤖 Building AI video assistants


## 🔮 Future Enhancements

* Streamlit / Web UI
* Multi-video indexing
* Speaker diarization
* Cross-video question answering
* GPU-optimized embedding & retrieval

🔗 GitHub: [https://github.com/shivavadtyavath](https://github.com/shivavadtyavath)



