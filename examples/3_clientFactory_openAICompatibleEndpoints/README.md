# Ollama

1. Install Ollama from https://ollama.com/
2. Run the Ollama server locally with the command below.

```bash
ollama serve
```

3. Download a model that you like

```bash
ollama pull qwen3:4b
```

it will start downloading the model and once it's done you can use it in the example below by changing the model name to "qwen3:4b"

4. download the appropriate package for datapizza-ai

```bash
pip install datapizza-ai-clients-openai-like
```

This because ollama exposes openAI compatible endpoints so we can use the openAI client to connect to it.

5. Move to main.py to see how to use one of these clients to connect to the Ollama server and use the model you downloaded.

> [!IMPORTANT]
> Check datapizza-ai documentation for further instructions