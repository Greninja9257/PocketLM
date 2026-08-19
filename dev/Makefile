# Dev pipeline. Every path here points inside dev/ on purpose — see dev/README.md.
.PHONY: all data tokenizer train chat clean
PY    ?= ../.venv/bin/python
MODEL ?= 50k
VOCAB ?= 512
ROOT  := ..

all: data tokenizer train

data:
	cd $(ROOT) && $(PY) dev/fetch_hf.py --target-mb 14
	cd $(ROOT) && $(PY) dev/augment.py --rate 0.6
	cd $(ROOT) && $(PY) scripts/build_corpus.py --out dev/data

tokenizer:
	cd $(ROOT) && $(PY) scripts/train_tokenizer.py --data dev/data --out-dir dev/checkpoints

train:
	cd $(ROOT) && $(PY) train.py --model $(MODEL) --curriculum dev \
		--data dev/data --tokenizer dev/checkpoints/tokenizer-$(VOCAB).json \
		--out dev/checkpoints/$(MODEL)-dev.pt

chat:
	cd $(ROOT) && $(PY) chat.py --checkpoint dev/checkpoints/$(MODEL)-dev.pt

clean:
	rm -rf data checkpoints
