# Re-indexing the documentation

Follow the below instructions step by step to re-index the documentation

1. Check if docs/README.md exists and is current
2. List documentation files in ./docs subdirectories (architecture/, development/, deployment/, tools/)
3. For each document found:
   - Determine if it fits well in its current category
   - If a document doesn't fit the existing categories, suggest creating a new category folder
   - Ask the user if they approve the suggested categorization
   - If user approves: file the document in the suggested location
   - If user doesn't approve: discuss with the user what the preferred filing system would look like
   - Upon agreement: file the document according to the agreed structure
4. In CLAUDE.md under the heading `Useful documents`, ensure all relevant docs are mentioned with descriptions
5. List the files in ./docs/development/instructions
6. In CLAUDE.md under the heading `Instructions for common tasks`, ensure all instruction files are mentioned
7. Update docs/README.md if new categories were created

#### notes

- The main documentation index is now docs/README.md which provides navigation
- Current documentation categories: architecture/, development/, deployment/, tools/
- Ignore files in docs/internal/ - these are internal planning documents not useful for users
- Focus on user-facing documentation in the main categories
- Ensure CLAUDE.md references match the organized structure
- When suggesting new categories, consider: user-guides/, tutorials/, api/, troubleshooting/, etc.
- Always maintain logical grouping and clear navigation