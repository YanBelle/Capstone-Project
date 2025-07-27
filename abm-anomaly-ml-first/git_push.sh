#!/bin/bash
echo "📤 Git Push Helper"
cd /home/yc/development/Capstone-Project/abm-anomaly-ml-first

echo "Current status:"
git status

echo ""
read -p "Enter commit message: " commit_msg

if [ -n "$commit_msg" ]; then
    git add .
    git commit -m "$commit_msg"
    git push origin main
    echo "✅ Changes pushed to GitHub"
    echo "🚀 GitHub Actions will deploy to production"
else
    echo "❌ No commit message provided"
fi
