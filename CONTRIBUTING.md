# Contributing to PDetective

Thank you for your interest in contributing to PDetective! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)

### Suggesting Enhancements

We welcome suggestions for new features or improvements. Please open an issue with:
- A clear description of the enhancement
- Use cases and benefits
- Any relevant examples or references

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code style guidelines
3. **Add tests** if applicable
4. **Update documentation** as needed
5. **Ensure tests pass** before submitting
6. **Submit a pull request** with a clear description of your changes

## Code Style Guidelines

- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Comment complex logic

## Code Formatting

We use the following tools for code quality:
- `black` for code formatting
- `flake8` for linting
- `isort` for import sorting

Run these before submitting:
```bash
black .
isort .
flake8 .
```

## Testing

- Write unit tests for new functionality
- Ensure existing tests pass
- Run tests with: `pytest tests/`

## Documentation

- Update README.md if you add new features
- Add docstrings following Google or NumPy style
- Update relevant documentation in `docs/`

## Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add feature X", "Fix bug Y")
- Reference issues when applicable (e.g., "Fixes #123")

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help create a welcoming environment for all contributors

## Questions?

Feel free to open an issue for any questions about contributing!

---

Thank you for contributing to PDetective! 🎉
