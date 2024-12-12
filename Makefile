.PHONY: run clean
run:
	python main.py
install:
	pip install -r requirements.txt
clean:
	rm -rf results
	rm -rf agents/__pycache__
