.PHONY: run clean
run:
	python main.py
clean:
	rm -rf results
	rm -rf agents/__pycache__
