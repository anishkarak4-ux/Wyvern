# Contributing to Wyvern

Thank you for your interest in contributing to **Wyvern: Large-Scale Multilingual Business Entity Resolution**!

We welcome community contributions, bug reports, and optimizations to enhance the entity resolution pipeline.

---

## Code of Conduct

Please maintain an inclusive, respectful, and constructive environment for all collaborators and contributors.

---

## Development Setup

1. **Fork and Clone the Repository:**
   ```bash
   git clone https://github.com/anishkarak4-ux/Wyvern.git
   cd Wyvern
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Installation:**
   ```bash
   pytest tests/
   ```

---

## Project Structure & Architecture Guidelines

- **`src/preprocessing/`**: Text cleaning, case folding, Unicode NFKC handling, and address normalization.
  - *Rule:* Ensure multilingual character integrity is preserved (do not apply destructive accent stripping on non-Latin or French characters).
- **`src/blocking/`**: Deterministic inverted-index blocking rules (Portfolio 2).
  - *Rule:* Country isolation must be strictly maintained (`US`, `India`, `France`).
- **`src/features/`**: Dynamic pairwise feature extraction.
  - *Rule:* Feature array length must strictly equal 46 and handle null/missing fields gracefully.
- **`src/models/`**: XGBoost training, refitting, inference, and post-processing (Rule F).
- **`utils/`**: Submission and formatting verification scripts (`validate_submission.py`).

---

## Testing Guidelines

- Write unit tests for new features in `tests/`.
- Run the test suite before submitting pull requests:
  ```bash
  pytest -v --cov=src tests/
  ```

---

## Pull Request Guidelines

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Keep commits concise and use conventional commit messages:
   - `feat(...)`: New feature or capability
   - `fix(...)`: Bug fix
   - `docs(...)`: Documentation update
   - `refactor(...)`: Code refactoring without behavior change
   - `test(...)`: Adding or updating tests
3. Ensure CI checks pass on GitHub Actions.
4. Submit a Pull Request describing your changes, motivation, and verification steps.
