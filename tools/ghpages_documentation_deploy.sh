#!/bin/bash

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# Script for generating the sphinx documentation and deploying it in the
# Github Pages repository. Please note that this depends on having the
# following variable set on travis containing a valid token with permissions
# for pushing into the Github Pages repository:
# GH_TOKEN

# Non-travis variables used by this script.
TARGET_REPOSITORY_USER="Qiskit"
TARGET_REPOSITORY_NAME="qiskit.github.io"
TARGET_TEMP_REPOSITORY_NAME=temp_$TARGET_REPOSITORY_NAME
TARGET_DOC_DIR="documentation/aqua"
SOURCE_DOC_DIR="docs/_build/html"
SOURCE_DIR=`pwd`

# Build the documentation.
make doc

echo "Cloning the Github Pages repository ..."
cd ..
if [ "$TRAVIS" != "true" ]; then
    rm -rf $TARGET_TEMP_REPOSITORY_NAME
    rm -rf $TARGET_REPOSITORY_NAME
fi
git clone https://github.com/$TARGET_REPOSITORY_USER/$TARGET_REPOSITORY_NAME.git
cd $TARGET_REPOSITORY_NAME

echo "Replacing $TARGET_DOC_DIR with the new contents ..."
git rm -rf $TARGET_DOC_DIR/_* $TARGET_DOC_DIR/*.html
mkdir -p $TARGET_DOC_DIR
cp -r $SOURCE_DIR/$SOURCE_DOC_DIR/* $TARGET_DOC_DIR/
git add $TARGET_DOC_DIR

if [ "$TRAVIS" == "true" ]; then
    echo "Commiting and pushing changes ..."
    git commit -m "Automated Aqua documentation update from SDK" -m "Commit: $TRAVIS_COMMIT" -m "Travis build: https://travis-ci.com/$TRAVIS_REPO_SLUG/builds/$TRAVIS_BUILD_ID"
    git push --quiet https://$GH_TOKEN@github.com/$TARGET_REPOSITORY_USER/$TARGET_REPOSITORY_NAME.git > /dev/null 2>&1
else
    echo "At this point you can commit and push your changes to qiskit.org"
    cd ..
    mkdir $TARGET_TEMP_REPOSITORY_NAME
    cp -r $TARGET_REPOSITORY_NAME/* $TARGET_TEMP_REPOSITORY_NAME/
    cd $TARGET_TEMP_REPOSITORY_NAME
    npm install
    npm start
fi
