from setuptools import setup, find_packages

setup(
    name="alpha_drug_discovery",
    version="1.0.0",
    author="David Xu",
    description="AI-driven drug discovery platform integrating multi-omics data.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/alpha_drug_discovery",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'torch',
        'tensorflow',
        'plotly',
        'pymol',
        'dash',
        'dash-bootstrap-components',
        'optuna',
        'reportlab',
        'biopython',
        'cryptography',
        'joblib'
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'alpha_drug_discovery=alpha_drug_discovery.main:main',
        ],
    },
)
