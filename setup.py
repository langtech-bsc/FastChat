from setuptools import setup, find_packages

setup(
    name="fschatbsc",
    version="0.1.0",
    description="A platform to finetune models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
    python_requires=">=3.8,<3.13",
    packages=find_packages(include=["fastchat", "fastchat.*"]),
    install_requires=[
        "aiohttp",
        "fastapi",
        "httpx",
        "markdown2[all]",
        "nh3",
        "numpy",
        "prompt_toolkit>=3.0.0",
        "pydantic<3,>=2.0.0",
        "pydantic-settings",
        "psutil",
        "requests",
        "rich>=10.0.0",
        "shortuuid",
        "tiktoken",
        "uvicorn",
        "packaging",
        "wheel",
        "deepsee",
        "sentencepiece",
        "protobuf"
    ],
    extras_require={
        "model_worker": ["accelerate>=0.21", "peft==0.11", "sentencepiece", "torch", "transformers==4.45", "protobuf"],
        "webui": ["gradio>=4.10"],
        "train": ["einops", "deepspeed==0.14.4", "wandb", "mlflow", "flash-attn>=2.0"],
        "llm_judge": ["openai<1", "anthropic>=0.3", "ray"],
        "dev": ["black==23.3.0", "pylint==2.8.2"],
    },
    include_package_data=True,
    package_data={
        'fastchat': ['deepspeed_configs/*.json'],
    },
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'train=fastchat.train.train:train',
            'train_baichuan=fastchat.train.train_baichuan:train_baichuan',
            'train_lora=fastchat.train.train_lora:train_lora',
            'train_mem=fastchat.train.train_mem:train_mem',
            'train_with_template=fastchat.train.train_with_template:train_with_template',
            'train_yuan2=fastchat.train.train_yuan2:train_yuan2',
            'train_flant5=fastchat.train.train_flant5:train_flant5',
            'train_lora_t5=fastchat.train.train_lora_t5:train_lora_t5',
            'train_xformers=fastchat.train.train_xformers:train_xformers',
        ],
    },
)

#pip install git+https://github.com/langtech-bsc/FastChat.git#egg=fschatbsc[model_worker,train]
