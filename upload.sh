#!/bin/bash

# 设置用户名
TWINE_USERNAME="u03013112"

# 使用keyring检索密码
TWINE_PASSWORD=$(python -c "import keyring; print(keyring.get_password('pypi', '$TWINE_USERNAME'))")

# 构建分发包
python setup.py sdist bdist_wheel

# 上传到PyPI
TWINE_USERNAME=$TWINE_USERNAME TWINE_PASSWORD=$TWINE_PASSWORD twine upload dist/*
