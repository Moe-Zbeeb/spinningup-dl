VENV := .venv
BIN := $(VENV)/bin
SRC := docs
OUT := docs/_build/html

.PHONY: setup html check serve clean

setup:
	python3 -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

html:
	$(BIN)/sphinx-build -b html $(SRC) $(OUT)

check:
	$(BIN)/sphinx-build -W --keep-going -b html $(SRC) $(OUT)

serve:
	$(BIN)/sphinx-autobuild $(SRC) $(OUT) --port 8000 --open-browser

clean:
	rm -rf docs/_build
