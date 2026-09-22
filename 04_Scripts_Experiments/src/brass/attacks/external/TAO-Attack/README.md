# TAO-Attack

This repository contains the official implementation of **TAO-Attack**, a novel optimization-based jailbreak attack for large language models (LLMs).  
The attack integrates a two-stage loss function and a direction–priority token optimization (DPTO) strategy, achieving higher attack success rates and improved efficiency across open- and closed-source LLMs.

---

## Installation

Clone the repository and install the dependencies:

```bash
cd TAO-Attack
pip install -r requirements.txt
```

You may also install it as a package:

```bash
pip install .
```

---

## Quick Start

Run the main attack implementation:

```bash
python attack.py
```


---

## Project Structure

```
LLM-ATTACKS/
├── api_experiments/       
├── data/                  # Data resources (e.g., evaluation prompts)
├── experiments/           
├── llm_attacks/           
├── attack.py              # Implementation of TAO-Attack
├── check_openai.py        # Helper script for evaluation with OpenAI APIs
├── requirements.txt       # Python dependencies
├── setup.py               # Installation script
├── LICENSE                # License file (MIT)
└── README.md              # Project description (this file)
```

---

## License

This project is licensed under the terms of the MIT License.
See the [LICENSE](LICENSE) file for details.
