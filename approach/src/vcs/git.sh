#!/usr/bin/env bash
git clone --bare $1 cache/$2/
cd cache/$2/
git log > ../$2.txt
cd ..
rm -rf $2/
exit 0;
