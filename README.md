
# Virtual environment
```bash
python -m venv datapizza_venv

# Activate the environment
source datapizza_venv/bin/activate

```

# datapizza-ai framework

```bash
pip install datapizza-ai

```

specific providers (optional)

```bash
pip install datapizza-ai-clients-openai
pip install datapizza-ai-clients-google
pip install datapizza-ai-clients-anthropic
```

# install other dependencies

```bash
pip install -r requirements.txt
```

# create .env file

create a .env file in the root directory of the project and add the following lines:

```bash
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
ANTHROPIC_API_KEY=your_anthropic_api_key
```



