# GitHub Upload Notes

Use these commands from the project folder when you are ready to publish.

```bash
git init
git add README.md LICENSE pyproject.toml requirements.txt src tests docs .gitignore
git commit -m "Add Taxi-v3 Q-learning project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

The `myProjk` environment, training artifacts, caches, package metadata, and `docs/line_by_line_explanation.md` are ignored so they do not get uploaded.

`docs/line_by_line_explanation.md` is a local study guide. Keep it for yourself, but leave it out of the public GitHub repository.
