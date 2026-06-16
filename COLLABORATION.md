# Collaboration Guidelines

Welcome to the team! Since we are all working on the same repository, please follow this workflow to keep our progress organized.

## Your Daily Workflow
1. **Sync your work**: Every morning, ensure you have the latest updates:

`git checkout main`

`git pull origin main`



2. **Create your branch**: Always work on your own branch. Do not push directly to `main`. 

`git checkout -b student-name-branch`


3. **Save your progress**: After making changes or adding scripts:

`git add .`

`git commit -m "Describe your changes here (e.g., Added calibration script)"`

`git push -u origin student-name-branch`


## Sharing and Merging
To hit the ground running, please review these key concepts:

- **Pull Requests** (PRs): When you are ready for your code to be reviewed or merged into the main project, open a Pull Request on the GitHub repository page.

- **Review**: As project lead, I will review your PR, provide feedback, and merge your work into the main branch.

- **Tip**: Remember to `git pull origin main` frequently to avoid conflicts!
