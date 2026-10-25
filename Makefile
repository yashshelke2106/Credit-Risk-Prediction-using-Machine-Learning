.PHONY: eval audit scorecard evaluate verify app api

eval:        ## regenerate every number in README.md and MODEL_CARD.md
	python run_eval.py

audit:
	python leakage_audit.py

scorecard:
	python build_scorecard.py

evaluate:
	python evaluate.py

verify:
	python verify_readme.py

app:
	streamlit run app1.py

api:
	uvicorn api:app --port 8000
