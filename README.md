# 🧠 Adaptive Learning Tutor (LangGraph + RAG)

An intelligent, multi-turn tutoring system built with LangGraph, Retrieval-Augmented Generation (RAG), and LLM-driven reasoning. The system diagnoses a student’s misunderstanding level and dynamically adapts follow-up questions to guide learning.

Sample Langsmith Tracing can be found here: https://smith.langchain.com/public/89bae199-713c-4195-b499-c2631e877573/r

Simple LangGraph Visualisation:

<img width="421" height="447" alt="image" src="https://github.com/user-attachments/assets/f6c6f157-815f-47a7-9710-163e083382e6" />

---

## 🚀 Key Features

### 🔹 1. Misunderstanding Classification

* Uses lecture-note retrieval (RAG) to compare student questions against course material.
* Classifies understanding as high, mid, or low.

### 🔹 2. Adaptive Follow-Up Questioning

* Generates follow-up questions based on the student’s level.
* Uses interrupt() to pause execution and collect real student replies.
* Reclassifies answers across multiple turns to measure improvement.

### 🔹 3. Multi-Turn Workflow with LangGraph

Structured graph of nodes:

* classify_initial_question
* generate_followup_question
* wait_for_answer
* classify_followup_answer
* final_output

Flow loops until:
✔️ understanding improves,
✔️ stagnates after several turns, or
✔️ gets worse → session ends.

### 🔹 4. Retrieval-Augmented Generation (RAG)

* Loads and chunks lecture PDFs.
* Embeds chunks and stores them in a vector database.
* Retrieves the most relevant excerpts to ground classification and feedback.

### 🔹 5. Personalized Final Feedback

* Summarizes student’s learning progress.
* Recommends specific lecture excerpts to review.
* Provides tailored encouragement based on their improvement trajectory.

---

## 🛠 Tech Stack

* LangGraph — multi-turn agent state machine
* LangChain / Tools — tool execution and retrieval
* Vector Store — similarity search over lecture notes
* LLM (ChatOpenAI) — classification + natural language generation
* PDF Loader + RecursiveCharacterTextSplitter — RAG preprocessing

---

## 🎯 What This Project Demonstrates

* Human-in-the-loop AI tutoring
* Adaptive multi-turn reasoning
* State-based LLM workflows using LangGraph
* Practical RAG pipeline for academic content
* Intelligent follow-up question generation
* End-to-end design for an AI tutor system

---
