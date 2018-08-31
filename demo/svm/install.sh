#! /usr/bin/bash

INSTALL_DIR=$HOME/.demo/svm/

if ! [ -d $INSTALL_DIR ]; then
  mkdir $INSTALL_DIR -p
  tar -xf img/data.tar.gz -C $INSTALL_DIR
  echo "Installed OCR image files to $INSTALL_DIR"
fi
