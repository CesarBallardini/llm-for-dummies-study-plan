# Complete Citations Index for LLM for Dummies Study Plan

**Generated: 2026-10-09**

## Summary
- **Total citations**: 300+ (books, papers, articles, courses)
- **From STUDY_PLAN.md**: 204 citations
- **From docs/ chapters**: 96 citations
- **Free access resources**: ~120+

## Structure of Citations

### Books (with metadata)
Format: Author(s) — Title (Edition, Publisher, Year) — [Link](URL) — Access status

### Papers/Articles
Format: Author(s) — [Title](URL) — Conference/Journal, Year

## Download Instructions

To download all free PDF resources:
```bash
cd books/free/
# Run download script (to be generated)
bash download_free_resources.sh
```

## Citation Categories

1. **Mathematics Foundations** (Chapters 1-5)
   - Algebra, Calculus, Linear Algebra, Probability, Optimization
   - ~40+ citations

2. **Programming & Tools** (Chapters 6-7)
   - Python, PyTorch, GPU, Scientific Stack
   - ~20+ citations

3. **Machine Learning Foundations** (Chapters 8-10)
   - ML basics, Deep Learning, Backpropagation
   - ~25+ citations

4. **Transformers & LLMs** (Chapters 11-16)
   - NLP, Transformer Architecture, Pretraining, Distributed Training
   - ~60+ citations

5. **Post-Training & Deployment** (Chapters 17-22)
   - Fine-tuning, Alignment, Reasoning, Evaluation, Inference, Operations
   - ~70+ citations

6. **Advanced Topics** (Chapters 23-24)
   - Capstone projects, Research reading
   - ~10+ citations

## Free Resources Already in books/free/

- `an_introduction_to_statistical_learning_python_2023.pdf`
- `linear_algebra_done_right_4e_2026.pdf`
- `mml-book_2024.pdf` (Mathematics for Machine Learning)
- `speech_and_language_processing_3e_2026.pdf` (Jurafsky & Martin)
- `understanding_deep_learning_2026.pdf` (Simon Prince)
- Plus 4 supplementary files

## To Complete the Download

Run this to extract and download all free PDFs:
```bash
python3 extract_and_download_citations.py
```

This will:
1. Parse STUDY_PLAN.md for all citations
2. Extract metadata (author, title, year, URL)
3. Identify PDF links marked as "free"
4. Download to books/free/
5. Create CITATIONS_FULL.csv with complete metadata

## Notes on Licensing

- Always confirm license before using any resource
- Most university courses are CC licensed (check individual pages)
- OpenStax books are free CC BY licensed
- arXiv papers are preprints
- Some resources marked "free" require verification of academic status
