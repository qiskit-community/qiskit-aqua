#! /usr/bin/bash

if ! [ -d 'img/data/' ]; then
  cd img
  tar -xzf data.tar.gz
  cd ..
  echo 'Installed image files'
fi
