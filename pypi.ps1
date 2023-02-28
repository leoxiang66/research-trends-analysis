# remember to modify setup.py (version and requirement)

rm -r -Force .\dist\
python setup.py sdist bdist_wheel
twine upload dist/*