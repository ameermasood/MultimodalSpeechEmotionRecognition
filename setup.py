from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent


def read_requirements() -> list[str]:
    requirements = ROOT / "requirements.txt"
    if not requirements.is_file():
        return []
    return [
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


setup(
    name="multimodal-speech-emotion-recognition",
    version="0.1.0",
    description="Multimodal speech emotion recognition with Voxtral and parameter-efficient fine-tuning.",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["mer", "mer.*"]),
    install_requires=read_requirements(),
    entry_points={"console_scripts": ["mer=mer.cli:main"]},
)
