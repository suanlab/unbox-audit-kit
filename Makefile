# Unbox reproduction + build targets.
# Usage:
#   make reproduce-tables   # offline: verify every paper number against stored artifacts
#   make test               # unit tests (unittest-based; no pytest required)
#   make paper              # compile paper/unbox_arr.tex (pdflatex+bibtex+pdflatex x2)
#   make supplement         # build anonymized release zip using .releaseignore
#   make clean-paper        # remove LaTeX build artifacts
#   make help

PY := .venv/bin/python
PAPER_DIR := paper
PAPER := unbox_arr
RELEASE_NAME := unbox_supplement_anonymized

.PHONY: help reproduce-tables test paper supplement clean-paper verify-supplement camera-ready-check

help:
	@echo "Unbox Makefile targets:"
	@echo "  make reproduce-tables  Verify paper numbers against stored artifacts (no API)"
	@echo "  make test              Run unit tests"
	@echo "  make paper             Compile paper/$(PAPER).tex (pdflatex+bibtex+pdflatex x2)"
	@echo "  make supplement        Build anonymized release zip at $(RELEASE_NAME).zip"
	@echo "  make verify-supplement Build supplement zip and assert forbidden paths are absent"
	@echo "  make camera-ready-check Assert no placeholders / anonymity left in the paper"
	@echo "  make clean-paper       Remove LaTeX build artifacts"

reproduce-tables:
	@$(PY) scripts/reproduce_paper_tables.py

test:
	@$(PY) -m unittest discover -s tests -v

paper:
	@cd $(PAPER_DIR) && rm -f $(PAPER).aux $(PAPER).bbl && \
	 pdflatex -interaction=nonstopmode $(PAPER).tex > /tmp/$(PAPER)_p1.log 2>&1 && \
	 bibtex $(PAPER) > /tmp/$(PAPER)_bib.log 2>&1 && \
	 pdflatex -interaction=nonstopmode $(PAPER).tex > /tmp/$(PAPER)_p2.log 2>&1 && \
	 pdflatex -interaction=nonstopmode $(PAPER).tex > /tmp/$(PAPER)_p3.log 2>&1 && \
	 echo "Compile succeeded. PDF: $(PAPER_DIR)/$(PAPER).pdf"

supplement: reproduce-tables
	@command -v rsync >/dev/null || { echo "rsync required"; exit 1; }
	@rm -rf $(RELEASE_NAME) $(RELEASE_NAME).zip
	@rsync -a --exclude-from=.releaseignore --exclude=.git ./ $(RELEASE_NAME)/
	@command -v zip >/dev/null && (cd $(RELEASE_NAME)/.. && zip -r $(RELEASE_NAME).zip $(RELEASE_NAME) > /dev/null) && echo "Zip: $(RELEASE_NAME).zip" || echo "Dir: $(RELEASE_NAME)/ (zip not installed)"

verify-supplement: supplement
	@echo "Verifying supplement zip is clean..."
	@FORBIDDEN="/\.env$$ /\.venv/ /__pycache__/ /experiments/failed_runs/ /paper/drafts/ /\.internal/ /\.sisyphus/ /experiments/retrospective_full/ /AUDIT\.md$$ /CHECK\.md$$ /ARR_SUBMISSION_CHECK\.md$$ /ARR_SUBMISSION_FORM\.md$$ _complete\.csv$$ /recruit_template\.md$$ /reinvite_template\.md$$ /Unbox-[0-9]"; \
	ZIP=$(RELEASE_NAME).zip; \
	if [ ! -f "$$ZIP" ]; then echo "ERROR: $$ZIP not found"; exit 1; fi; \
	FAILED=0; \
	for pattern in $$FORBIDDEN; do \
		if unzip -l "$$ZIP" | awk '{print $$NF}' | grep -qE "$$pattern"; then \
			echo "FAIL: supplement contains forbidden path matching: $$pattern"; \
			unzip -l "$$ZIP" | awk '{print $$NF}' | grep -E "$$pattern" | head -3 | sed 's/^/  /'; \
			FAILED=1; \
		fi; \
	done; \
	if [ "$$FAILED" -eq 1 ]; then exit 1; fi; \
	echo "OK: supplement zip contains no forbidden paths."

clean-paper:
	@cd $(PAPER_DIR) && rm -f *.aux *.log *.fls *.fdb_latexmk *.out *.synctex.gz

# Camera-ready gate: fail loudly on the two things easiest to forget.
camera-ready-check:
	@fail=0; \
	if grep -q "RESULTS-PENDING" $(PAPER_DIR)/$(PAPER).tex; then \
		echo "FAIL: RESULTS-PENDING placeholder still in the paper:"; \
		grep -n "RESULTS-PENDING" $(PAPER_DIR)/$(PAPER).tex | sed 's/^/  /'; \
		fail=1; \
	fi; \
	if grep -q "Anonymous ACL submission" $(PAPER_DIR)/$(PAPER).tex; then \
		echo "FAIL: author block is still anonymized (camera-ready must be de-anonymized)."; \
		fail=1; \
	fi; \
	if grep -q "with all artifacts anonymized" $(PAPER_DIR)/$(PAPER).tex; then \
		echo "FAIL: abstract still says \"artifacts anonymized\" (should be \"released\")."; \
		fail=1; \
	fi; \
	if [ -f $(PAPER_DIR)/$(PAPER).pdf ] && pdftotext $(PAPER_DIR)/$(PAPER).pdf - 2>/dev/null | grep -q "??"; then \
		echo "FAIL: unresolved LaTeX reference (\"??\") in the compiled PDF."; \
		fail=1; \
	fi; \
	if [ "$$fail" -eq 1 ]; then exit 1; fi; \
	echo "OK: camera-ready checks pass."
