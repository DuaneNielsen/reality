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
4. List the files in ./docs/development/instructions
5. In CLAUDE.md under the heading `Instructions for common tasks`, ensure all instruction files are mentioned
6. Update docs/README.md if new categories were created
7. **Verify document descriptions in README.md**:
   - Read each document listed in docs/README.md
   - Compare the document's actual content with its description in the README
   - If the description is inaccurate, incomplete, or misleading, improve it to accurately reflect the document's content
   - Ensure descriptions are concise but informative (1-2 sentences max)
   - Focus on what the document helps users accomplish, not just what it contains

#### notes

- The main documentation index is now docs/README.md which provides navigation
- Current documentation categories: architecture/, development/, deployment/, tools/
- Ignore files in docs/internal/ - these are internal planning documents not useful for users
- Focus on user-facing documentation in the main categories
- Ensure CLAUDE.md references match the organized structure
- When suggesting new categories, consider: user-guides/, tutorials/, api/, troubleshooting/, etc.
- Always maintain logical grouping and clear navigation
- Document descriptions should be outcome-focused (what users can achieve) rather than content-focused (what's inside)