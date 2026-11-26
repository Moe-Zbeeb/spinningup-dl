VENV := .venv
BIN := $(VENV)/bin
SRC := docs
OUT := docs/_build/html

PAGE ?= index
HEIGHT ?= 3000

.PHONY: setup html lint check serve shot clean

setup:
	python3 -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

html:
	$(BIN)/sphinx-build -b html $(SRC) $(OUT)

lint:
	$(BIN)/python tools/lint_content.py

check: lint
	$(BIN)/sphinx-build -W --keep-going -b html $(SRC) $(OUT)

serve:
	$(BIN)/sphinx-autobuild $(SRC) $(OUT) --port 8000 --open-browser

shot:
	$(BIN)/python tools/screenshot.py $(PAGE) $(HEIGHT)

clean:
	rm -rf docs/_build
