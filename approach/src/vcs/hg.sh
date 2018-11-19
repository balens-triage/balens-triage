#!/usr/bin/env bash
hg clone $1 cache/$2/
cd cache/$2/
hg log > ../$2.txt
cd ..
rm -rf $2/
exit 0;
