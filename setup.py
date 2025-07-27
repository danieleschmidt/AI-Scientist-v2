#!/usr/bin/env python3

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="ai-scientist-v2",
    version="2.0.0",
    author="Terragon Labs",
    author_email="contact@terragonlabs.com",
    description="Autonomous scientific research system via agentic tree search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SakanaAI/AI-Scientist-v2",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
            "myst-parser>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-scientist=launch_scientist_bfts:main",
            "ai-ideation=ai_scientist.perform_ideation_temp_free:main",
            "ai-writeup=ai_scientist.perform_writeup:main",
        ],
    },
    scripts=[
        "scripts/security_scan.py",
        "run_tests.py",
        "metrics_reporter.py",
    ],
    include_package_data=True,
    package_data={
        "ai_scientist": [
            "blank_icbinb_latex/*",
            "blank_icml_latex/*",
            "fewshot_examples/*",
            "treesearch/utils/viz_templates/*",
        ],
    },
)