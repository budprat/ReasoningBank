#!/bin/bash

# ReasoningBank Codebase Cleanup Script
# This script removes unnecessary files and organizes the codebase
# Run with: bash cleanup.sh

set -e  # Exit on error

echo "🧹 ReasoningBank Cleanup Script"
echo "================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Counters
files_removed=0
space_saved=0

# Function to safely remove files
safe_remove() {
    if [ -e "$1" ]; then
        size=$(du -sk "$1" 2>/dev/null | cut -f1)
        rm -rf "$1"
        files_removed=$((files_removed + 1))
        space_saved=$((space_saved + size))
        echo -e "${GREEN}✓${NC} Removed: $1 (${size}KB)"
    else
        echo -e "${YELLOW}⚠${NC} Not found: $1"
    fi
}

echo "Step 1: Removing Python cache files..."
echo "---------------------------------------"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
echo -e "${GREEN}✓${NC} Python cache cleaned"
echo ""

echo "Step 2: Removing macOS system files..."
echo "---------------------------------------"
find . -name ".DS_Store" -delete 2>/dev/null || true
echo -e "${GREEN}✓${NC} macOS files cleaned"
echo ""

echo "Step 3: Removing accidental files..."
echo "-------------------------------------"
safe_remove "reasoningbank/=0.3.0"
echo ""

echo "Step 4: Cleaning test data (15MB+)..."
echo "--------------------------------------"
if [ -d "data" ]; then
    safe_remove "data/embeddings.json"
    safe_remove "data/memory_bank.json"
    # Keep the data directory but add .gitkeep
    touch data/.gitkeep
    echo -e "${GREEN}✓${NC} Added .gitkeep to data/"
fi
echo ""

echo "Step 5: Removing duplicate files..."
echo "------------------------------------"
safe_remove "reasoningbank/.env.example"
safe_remove "example_agent.py"  # Keeping examples/basic_usage.py instead
echo ""

echo "Step 6: Moving misplaced files..."
echo "----------------------------------"
# Move .github to root if it's in the wrong place
if [ -d "reasoningbank/.github" ] && [ ! -d ".github" ]; then
    mv reasoningbank/.github .
    echo -e "${GREEN}✓${NC} Moved .github to root"
fi

# Move test file to proper location
if [ -f "test_live_agent.py" ]; then
    mkdir -p tests/integration
    mv test_live_agent.py tests/integration/
    echo -e "${GREEN}✓${NC} Moved test_live_agent.py to tests/integration/"
fi
echo ""

echo "Step 7: Archiving development documentation..."
echo "----------------------------------------------"
mkdir -p docs/archive

# Move development planning docs
for file in GAP_*.md TESTING_GAP_*.md CROSS_CHECK_*.md TEST_*_ANALYSIS.md ULTRATHINK_*.md TEST_STATUS_SUMMARY.md ARCHITECTURE_ANALYSIS.md; do
    if [ -f "$file" ]; then
        mv "$file" docs/archive/
        echo -e "${GREEN}✓${NC} Archived: $file"
    fi
done
echo ""

echo "Step 8: Adding .gitkeep to empty directories..."
echo "-----------------------------------------------"
if [ -d "logs" ]; then
    touch logs/.gitkeep
    echo -e "${GREEN}✓${NC} Added .gitkeep to logs/"
fi
echo ""

echo "Step 9: Creating comprehensive .gitignore..."
echo "--------------------------------------------"
cat > .gitignore.new << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environments
venv/
ENV/
env/
.venv/

# Test Coverage
htmlcov/
.tox/
.coverage
.coverage.*
.cache
.pytest_cache/
nosetests.xml
coverage.xml
*.cover

# Data and Logs
data/*.json
data/*.db
data/*.sqlite
data/embeddings*
data/memory_bank*
logs/*.log
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.project
.pydevproject
.settings/

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Environment
.env
.env.local
.env.*.local

# Development artifacts
=0.3.0
*.tmp
*.temp
*.cache
*.bak
*.backup
*.old

# Documentation builds
docs/_build/
docs/.doctrees/
site/

# Jupyter
.ipynb_checkpoints/

# MyPy
.mypy_cache/
.dmypy.json
dmypy.json

# Ruff
.ruff_cache/
EOF

if [ -f ".gitignore" ]; then
    echo -e "${YELLOW}⚠${NC} .gitignore already exists. New version saved as .gitignore.new"
    echo "   Review and merge manually with: diff .gitignore .gitignore.new"
else
    mv .gitignore.new .gitignore
    echo -e "${GREEN}✓${NC} Created comprehensive .gitignore"
fi
echo ""

echo "========================================"
echo -e "${GREEN}🎉 Cleanup Complete!${NC}"
echo "========================================"
echo "Files removed: $files_removed"
echo "Space saved: ~${space_saved}KB"
echo ""
echo "📋 Next Steps:"
echo "1. Review archived docs in docs/archive/"
echo "2. Run tests to ensure everything works:"
echo "   cd reasoningbank && python -m pytest tests/"
echo "3. Commit the cleanup:"
echo "   git add -A"
echo "   git commit -m 'chore: Clean development artifacts and optimize repository structure'"
echo ""
echo -e "${YELLOW}⚠ Important:${NC}"
echo "- Check .gitignore.new if .gitignore already existed"
echo "- Verify imports still work after moving files"
echo "- Some development docs were archived, not deleted"
echo ""