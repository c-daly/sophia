# Contributing to Sophia

Thank you for your interest in contributing to Sophia! This document provides guidelines for contributing to the project.

## Development Setup

### Prerequisites

- Python >=3.11
- Poetry (recommended)

### Setup with Poetry

1. Clone the repository:
   ```bash
   git clone https://github.com/c-daly/sophia.git
   cd sophia
   ```

2. Install Poetry (if not already installed):
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   # or
   pip install poetry
   ```

3. Install dependencies:
   ```bash
   poetry install --with dev
   ```

4. Run tests to verify your setup:
   ```bash
   poetry run pytest
   ```

## CI Parity: Running All Checks Locally

Before opening a pull request, run these commands to mirror the GitHub Actions CI pipeline:

```bash
poetry install --with dev
poetry run ruff check src tests
poetry run black --check src tests
poetry run mypy src
poetry run pytest tests/ -v -m "not integration" --cov=sophia --cov-report=term --cov-report=xml
```

All checks must pass for your PR to be merged. Note: Integration tests requiring Neo4j/Milvus are excluded from CI using the `-m "not integration"` marker.

## Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes following the code style guidelines below.

3. Write tests for your changes.

4. Run the test suite:
   ```bash
   poetry run pytest
   ```

5. Format and lint your code (see CI Parity section above for all checks):
   ```bash
   poetry run black src tests
   poetry run ruff check src tests
   poetry run mypy src
   ```

6. Commit your changes with a clear commit message:
   ```bash
   git commit -m "Add feature: description of your feature"
   ```

7. Push your branch and create a pull request.

## Code Style

- Follow PEP 8 guidelines
- Use type hints for all function signatures
- Format code with `black`
- Lint code with `ruff`
- Maximum line length: 88 characters
- Write docstrings for all public modules, classes, and functions

## Testing

- Write unit tests for all new functionality
- Aim for high test coverage (>90%)
- Use pytest for all tests
- Place tests in the `tests/` directory mirroring the source structure
- Run tests with: `poetry run pytest`

## Managing Dependencies

- Add runtime dependencies: `poetry add package-name`
- Add development dependencies: `poetry add --group dev package-name`
- Update dependencies: `poetry update`
- Check for outdated dependencies: `poetry show --outdated`

## Documentation

- Update README.md if adding new features
- Write clear docstrings following Google style
- Add examples for new functionality

## Pull Request Process

1. Ensure all tests pass
2. Update documentation as needed
3. Add a clear description of your changes in the PR
4. Link any related issues
5. Wait for review and address any feedback

## Questions?

If you have questions, please open an issue for discussion.
