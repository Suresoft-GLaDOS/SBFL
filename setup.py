import io
from setuptools import find_packages, setup
setup(
    name             = 'sbfl-engine',
    version          = '1.0',
    description      = 'An engine for spectrum-based fault localization',
    author           = 'Gabin An',
    author_email     = 'agb94@kaist.ac.kr',
    url              = 'https://github.com/Suresoft-GLaDOS/SBFL',
    download_url     = 'https://github.com/Suresoft-GLaDOS/SBFL',
    install_requires = ['numpy==1.20.2', 'pandas==1.2.4', 'scikit-learn==0.24.2', 'scipy==1.6.3', 'tqdm==4.60.0'],
    packages         = find_packages(),
    keywords         = ['sbfl'],
    python_requires  = '>=3.8',
    package_data={},
    zip_safe=True,
)