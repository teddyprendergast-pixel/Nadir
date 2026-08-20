# Contributing to Nadir

Thank you for your interest in contributing to the **Nadir Bipedal Walker** project! 🦿

## 📌 Development Workflow

1. **Fork the Repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/Nadir.git
   cd Nadir
   ```
3. **Create a branch** for your feature or bugfix:
   ```bash
   git checkout -b feature/dynamic-gait-optimization
   ```
4. **Make changes** and ensure coding standards are maintained.
5. **Test your changes**:
   - Verify environment loading: `python -c "import sim.env"`
   - Test stepping logic and visualization: `python inference/visualize.py`
6. **Commit with clean commit messages**:
   ```bash
   git commit -m "feat(sim): add foot contact sensors to observation space"
   ```
7. **Push to your fork** and submit a **Pull Request (PR)**.

## 🛠️ Code Guidelines

- Follow PEP 8 style guidelines.
- Add clear docstrings explaining kinematics, reward mathematics, or sensor communication.
- When adding physical components or modifying dimensions, update both `models/nadir.xml` and `docs/kinematics.md`.

## 💬 Community & Questions

If you encounter issues or have design proposals, feel free to open a GitHub Issue or Discussion.
