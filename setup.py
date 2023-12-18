import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tts-scores",
    packages=["tts_scores"],
    version="1.0.1",
    author="James",
    author_email="james@adamant.ai",
    description="A library for computing performance metrics for text-to-speech systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neonbjb/tts-scores",
    project_urls={},
    install_requires=[
        'tqdm',
        'scipy',
        'torch>=1.8',
        'torchaudio>0.9',
        'transformers',
        'tokenizers',
        'requests',
        'ffmpeg',
        'unidecode',
        'inflect',
        'pytorch_fid',
        'einops'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    download_url = 'https://github.com/neonbjb/tts-scores/archive/refs/tags/1.0.0.tar.gz',
    python_requires=">=3.6",
)
