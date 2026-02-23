<div align="center">
  <img src="assets/cover.png" alt="Datapizza AI Cover" width="100%">
  
  # 🍕 Datapizza-AI Workspace
  
  **Un buon punto di partenza per iniziare a sviluppare applicazioni basate su intelligenza artificiale con il framework `datapizza-ai`.**
  
  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Framework: Datapizza](https://img.shields.io/badge/Framework-Datapizza-FFD700.svg)](https://datapizza.tech)
</div>

---

### 🚀 Key Features

- 🤖 **AI-Agent Ready**: Workspace ottimizzato per lavorare con agenti AI come **Cursor**, **Antigravity** e simili. Fornisce il contesto necessario per permettere all'IA di scrivere codice utilizzando il framework `datapizza-ai`.
- 📚 **Documentazione Integrata**: Tutta la documentazione tecnica è disponibile nella cartella `docs/`, ideale da "dare in pasto" all'agente per spiegare il funzionamento del framework.
- 🎥 **Esempi da YouTube**: Include implementazioni pratiche (cartella `examples/`) e pattern di utilizzo estratti dai tutorial di Datapizza, utili per mostrare all'agente come integrare la libreria.
- ⚡ **Context Injection**: Il punto di partenza perfetto per istruire tool di AI Coding su una libreria che potrebbero non conoscere nativamente, garantendo suggerimenti precisi.


## 🛠️ Installazione

Configura il tuo ambiente in pochi secondi:

### 1. Crea un Virtual Environment
```bash
python -m venv datapizza_venv
source datapizza_venv/bin/activate  # Su Windows: .\datapizza_venv\Scripts\activate
```

### 2. Installa il Framework
```bash
pip install datapizza-ai
```

### 3. Installa i Provider (Opzionale)
Installa solo quello che ti serve:
```bash
pip install datapizza-ai-clients-openai
pip install datapizza-ai-clients-google
pip install datapizza-ai-clients-anthropic
```

### 4. Installa le Dipendenze
```bash
pip install -r requirements.txt
```

---

## 🔐 Configurazione

Crea un file `.env` nella root del progetto e aggiungi le tue API Key:

```env
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

---

## 📚 Esempi

Trovi esempi completi nella cartella `examples/`:
- 🤖 `1_simple_chatbot`: Chatbot essenziale.
- 🌀 `2_streaming_structure_multimodality`: Feature avanzate.
- 🏭 `3_clientFactory`: Gestione dinamica dei client.
- 🔍 `4_webSearch_functionCalling`: Tool use e ricerca web.

---

