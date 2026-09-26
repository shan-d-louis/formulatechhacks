---
name: evgs-jupyter-notebook-editing
description: Use when working on Jupyter Notebooks.
---
## Clean Code Patching
If you try to modify or add a cell, leave the patch cleanly without a trailing `\n` newline on the last line of the cell.

Try your best to keep minimal patches of codes on each cell. This is for the sake of readability and maintainability. Reuse variables and logic whereever possible. If modularizing is possible, try to put common helpers in a separate accessible Python file at a reasonable location. Avoid duplicating code across cells.

## Testing Preference
Do not add or commit notebook static tests unless the user explicitly requests them.
For notebook validation, prefer temporary local/sandbox-only inspection scripts and keep
committed tests focused on reusable Python helpers.