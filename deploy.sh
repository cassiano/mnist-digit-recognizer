#!/bin/bash
set -e

npm run build
rm -rf /tmp/ghpages-deploy
cp -r dist /tmp/ghpages-deploy
cd /tmp/ghpages-deploy
git init
git checkout -b gh-pages
git add -A
git commit -m "Deploy $(date +%Y-%m-%d_%H:%M)"
git remote add github https://github.com/cassiano/mnist-digit-recognizer.git
git push -f github gh-pages
echo "Deployed to https://cassiano.github.io/mnist-digit-recognizer/"
