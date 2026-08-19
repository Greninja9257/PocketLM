.PHONY: all data tokenizer train train-family eval bakeoff chat export clean
PY    ?= .venv/bin/python
MODEL ?= 50k
FAMILY = 1k 5k 10k 50k 100k 500k 1m

all: data tokenizer train-family eval-family

data:
	$(PY) scripts/build_corpus.py
	$(PY) eval/make_evalset.py

tokenizer:
	$(PY) scripts/train_tokenizer.py

train:
	$(PY) train.py --model $(MODEL)

# The whole family, smallest first.
train-family:
	@for m in $(FAMILY); do echo "=== $$m"; $(PY) train.py --model $$m || exit 1; done

# Architecture bake-off at the 50K budget: transformer vs MQA vs GRU vs hybrid.
train-variants:
	@for m in 50k-mqa 50k-gru 50k-hybrid; do $(PY) train.py --model $$m || exit 1; done

eval:
	$(PY) eval/evaluate.py --model $(MODEL)

eval-family:
	$(PY) eval/evaluate.py --checkpoint checkpoints/1k.pt --compare \
		checkpoints/5k.pt checkpoints/10k.pt checkpoints/50k.pt \
		checkpoints/100k.pt checkpoints/500k.pt checkpoints/1m.pt

bakeoff:
	$(PY) eval/evaluate.py --checkpoint checkpoints/50k.pt --compare \
		checkpoints/50k-mqa.pt checkpoints/50k-gru.pt checkpoints/50k-hybrid.pt \
		--dump eval/results/blind.json

chat:
	$(PY) chat.py --model $(MODEL)

export:
	$(PY) export.py --model $(MODEL)

clean:
	rm -f checkpoints/*.pt checkpoints/*.log
